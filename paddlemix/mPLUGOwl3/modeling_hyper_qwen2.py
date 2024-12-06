# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys

sys.path.append("/home/aistudio/paddle_test/mPLUGOwl3/utils")
import paddle
import paddle_aux
import paddlenlp

""" PyTorch Qwen2 model."""
import inspect
import math
from typing import List, Optional, Tuple, Union

from einops import rearrange, repeat

from paddlemix.models.flash_attn_utils import (
    has_flash_attn_func,
    is_flash_attn_available,
)

from .activations import ACT2FN
from .bert_padding import index_first_axis, pad_input, unpad_input
from .configuration_hyper_qwen2 import HyperQwen2Config

if is_flash_attn_available():
    flash_attn_func, flash_attn_varlen_func = has_flash_attn_func()
    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)
# >>>>>>if transformers.utils.is_flash_attn_2_available():
#     pass
#     _flash_supports_window_size = 'window_size' in list(inspect.signature(
#         flash_attn_func).parameters)

from .x_sdpa import ScaleDotProductAttention

try:
    from einops import rearrange

    use_flash_rotary = True
    print("use flash_attn rotary")
except ImportError:
    use_flash_rotary = False
    print("import flash_attn rotary fail")
logger = paddle.utils.try_import("logging").getLogger(name=__name__)
_CHECKPOINT_FOR_DOC = "Qwen/Qwen2-7B-beta"
_CONFIG_FOR_DOC = "HyperQwen2Config"


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(axis=-1, dtype="int32")
    paddle.utils.try_import("warnings").warn("Now, the return shape is inconsistent with torch when as_tuple is True")
    indices = paddle.nonzero(x=attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = paddle.nn.functional.pad(
        x=paddle.cumsum(x=seqlens_in_batch, axis=0, dtype="int32"), pad=(1, 0), pad_from_left_axis=False
    )
    return indices, cu_seqlens, max_seqlen_in_batch


class Qwen2RMSNorm(paddle.nn.Layer):
    def __init__(self, hidden_size, eps=1e-06):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = paddle.base.framework.EagerParamBase.from_tensor(tensor=paddle.ones(shape=hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to("float32")
        variance = hidden_states.pow(y=2).mean(axis=-1, keepdim=True)
        hidden_states = hidden_states * paddle.rsqrt(x=variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Qwen2RotaryEmbedding(paddle.nn.Layer):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / self.base ** (
            paddle.arange(start=0, end=self.dim, step=2, dtype="int64").astype(dtype="float32").to(device) / self.dim
        )
        self.register_buffer(name="inv_freq", tensor=inv_freq, persistable=False)
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.place, dtype=paddle.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = paddle.arange(dtype="int64", end=self.max_seq_len_cached).astype(dtype=self.inv_freq.dtype)
        freqs = paddle.outer(x=t, y=self.inv_freq)
        emb = paddle.concat(x=(freqs, freqs), axis=-1)
        self.register_buffer(name="cos_cached", tensor=emb.cos().to(dtype), persistable=False)
        self.register_buffer(name="sin_cached", tensor=emb.sin().to(dtype), persistable=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.place, dtype=x.dtype)
        return self.cos_cached[:seq_len].to(dtype=x.dtype), self.sin_cached[:seq_len].to(dtype=x.dtype)


class RotaryEmbedding(paddle.nn.Layer):
    def __init__(self, dim, base=10000, use_fp32=False, use_outer_in_rope=False):
        super().__init__()
        self.dim = dim
        self.base = base
        self.use_fp32 = use_fp32
        if use_fp32:
            self.inv_freq = 1.0 / base ** (paddle.arange(start=0, end=dim, step=2).astype(dtype="float32") / dim)
        else:
            inv_freq = 1.0 / base ** (paddle.arange(start=0, end=dim, step=2).astype(dtype="float32") / dim)
            self.register_buffer(name="inv_freq", tensor=inv_freq)
        self._rotary_pos_emb_cache = None
        self._seq_len_cached = 0
        self.use_outer_in_rope = use_outer_in_rope
        self._ntk_alpha_cached = 1.0

    def update_rotary_pos_emb_cache(self, max_seq_len, offset=0, ntk_alpha=1.0):
        seqlen = max_seq_len + offset
        if seqlen > self._seq_len_cached or ntk_alpha != self._ntk_alpha_cached:
            base = self.base * ntk_alpha ** (self.dim / (self.dim - 2))
            self.inv_freq = 1.0 / base ** (
                paddle.arange(start=0, end=self.dim, step=2).astype(dtype="float32") / self.dim
            )
            self._seq_len_cached = seqlen
            self._ntk_alpha_cached = ntk_alpha
            seq = paddle.arange(end=seqlen)
            if self.use_outer_in_rope:
                freqs = paddle.outer(x=seq.astype(dtype=self.inv_freq.dtype), y=self.inv_freq)
            else:
                freqs = einsum("i , j -> i j", seq.astype(dtype=self.inv_freq.dtype), self.inv_freq)
            emb = paddle.concat(x=(freqs, freqs), axis=-1)
            from einops import rearrange

            self._rotary_pos_emb_cache = rearrange(emb, "n d -> n 1 1 d")

    def forward(self, max_seq_len, offset=0, ntk_alpha=1.0):
        self.update_rotary_pos_emb_cache(max_seq_len, offset, ntk_alpha)
        return self._rotary_pos_emb_cache[offset : offset + max_seq_len]


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : tuple(x.shape)[-1] // 2]
    x2 = x[..., tuple(x.shape)[-1] // 2 :]
    return paddle.concat(x=(-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(axis=unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(axis=unsqueeze_dim)
    # print(q.shape)
    # print('-----------------')
    # print((rotate_half(q) * sin).shape)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


# Copied from transformers.models.llama.modeling_llama._prepare_4d_causal_attention_mask_with_cache_position
def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: paddle.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: paddle.dtype,
    min_dtype: float,
    cache_position: paddle.Tensor,
    batch_size: int,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

    Args:
        attention_mask (`paddle.Tensor`):
            A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
        sequence_length (`int`):
            The sequence length being processed.
        target_length (`int`):
            The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
        dtype (`paddle.dtype`):
            The dtype to use for the 4D attention mask.
        min_dtype (`float`):
            The minimum value representable with the dtype `dtype`.
        cache_position (`paddle.Tensor`):
            Indices depicting the position of the input sequence tokens in the sequence.
        batch_size (`paddle.Tensor`):
            Batch size.
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        causal_mask = paddle.full([sequence_length, target_length], fill_value=min_dtype, dtype=dtype)
        if sequence_length != 1:
            causal_mask = paddle.triu(x=causal_mask, diagonal=1)
        bool_tensor = paddle.arange(target_length) > cache_position.reshape([-1, 1])
        float_tensor = float16_tensor = bool_tensor.astype(paddle.float16)
        causal_mask *= float_tensor
        causal_mask = causal_mask[None, None, :, :].expand(shape=[batch_size, 1, -1, -1])
        if attention_mask is not None:
            causal_mask = causal_mask.clone()
            mask_length = tuple(attention_mask.shape)[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                mask=padding_mask, value=min_dtype
            )
    return causal_mask


class Qwen2MLP(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = paddle.nn.Linear(
            in_features=self.hidden_size, out_features=self.intermediate_size, bias_attr=False
        )
        self.up_proj = paddle.nn.Linear(
            in_features=self.hidden_size, out_features=self.intermediate_size, bias_attr=False
        )
        self.down_proj = paddle.nn.Linear(
            in_features=self.intermediate_size, out_features=self.hidden_size, bias_attr=False
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def repeat_kv(hidden_states: paddle.Tensor, n_rep: int) -> paddle.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = tuple(hidden_states.shape)
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(shape=[batch, num_key_value_heads, n_rep, slen, head_dim])
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def make_t2v_mask(media_offset_line, num_images):
    assert len(tuple(media_offset_line.shape)) == 1
    # media_offset_line = media_offset_line.view(-1, 1)
    # visual_arange = paddle.arange(end=num_images).view(1, -1)
    media_offset_line = paddle.reshape(media_offset_line, [-1, 1])
    visual_arange = paddle.arange(end=num_images).reshape([1, -1])

    mask = media_offset_line <= visual_arange
    return mask


def select_query(media_offset, num_queries=None):
    query_indices = media_offset[:, :, 1] >= 0
    assert query_indices.sum().item() % num_queries == 0, query_indices.sum().item()
    query_indices = query_indices.nonzero()
    ptr = 0
    while ptr < tuple(query_indices.shape)[0]:
        first_query_index, last_query_index = query_indices[ptr], query_indices[ptr + num_queries - 1]
        assert (last_query_index[1] - first_query_index[1] + 1).item() == num_queries
        assert last_query_index[0].item() == first_query_index[0].item()
        batch_id, begin_i, end_i = (
            first_query_index[0].item(),
            first_query_index[1].item(),
            first_query_index[1].item() + num_queries,
        )
        yield batch_id, begin_i, end_i
        ptr += num_queries


def _rotate_half(x):
    """
    change sign so the last dimension becomes [-odd, +even]
    """
    from einops import rearrange

    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(axis=-2)
    return paddle.concat(x=(-x2, x1), axis=-1)


def apply_rotary_pos_emb_core(t, freqs, use_fp32=False, debug=False):
    """
    input tensor t is of shape [seq_length, ..., dim]
    rotary positional embeding tensor freqs is of shape [seq_length, ..., dim]
    check https://kexue.fm/archives/8265 for detailed formulas
    """
    if use_flash_rotary and use_fp32:
        t_ = rearrange(t, "s b ... -> b s ...").contiguous()
        if use_fp32:
            t_ = t_.astype(dtype="float32")
        freqs = freqs.squeeze(axis=1).squeeze(axis=1)
        cos = freqs[:, : tuple(freqs.shape)[-1] // 2].cos()
        sin = freqs[:, : tuple(freqs.shape)[-1] // 2].sin()
        output = paddle_aux.apply_rotary_emb_func(x=t_, cos=cos, sin=sin).astype(dtype=t.dtype)
        if debug:
            from icecream import ic

            ic(tuple(t_.shape), tuple(freqs.shape), tuple(cos.shape))
        return rearrange(output, "b s ... -> s b ...")
    rot_dim = tuple(freqs.shape)[-1]
    t_, t_pass_ = t[..., :rot_dim], t[..., rot_dim:]
    if use_fp32:
        t_ = t_.astype(dtype="float32")
        t_pass_ = t_pass_.astype(dtype="float32")
    t_ = t_ * freqs.cos() + _rotate_half(t_) * freqs.sin()
    return paddle.concat(x=(t_, t_pass_), axis=-1).astype(dtype=t.dtype)


class HyperQwen2Attention(paddle.nn.Layer):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: HyperQwen2Config, layer_idx: Optional[int] = None, is_hyper_enabed=False):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` when creating this class."
            )
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`: {self.num_heads})."
            )
        self.q_proj = paddle.nn.Linear(
            in_features=self.hidden_size, out_features=self.num_heads * self.head_dim, bias_attr=True
        )
        self.k_proj = paddle.nn.Linear(
            in_features=self.hidden_size, out_features=self.num_key_value_heads * self.head_dim, bias_attr=True
        )
        self.v_proj = paddle.nn.Linear(
            in_features=self.hidden_size, out_features=self.num_key_value_heads * self.head_dim, bias_attr=True
        )
        self.o_proj = paddle.nn.Linear(
            in_features=self.num_heads * self.head_dim, out_features=self.hidden_size, bias_attr=False
        )
        self.rotary_emb = Qwen2RotaryEmbedding(
            self.head_dim, max_position_embeddings=self.max_position_embeddings, base=self.rope_theta
        )
        self.rotary_emb_core = RotaryEmbedding(
            self.head_dim, base=self.rope_theta, use_fp32=True, use_outer_in_rope=True
        )
        self.is_hyper_enabed = is_hyper_enabed
        if self.is_hyper_enabed:
            self.v_kv_proj = paddle.nn.Linear(
                in_features=self.hidden_size, out_features=self.num_key_value_heads * self.head_dim * 2, bias_attr=True
            )
            self.gate = paddle.base.framework.EagerParamBase.from_tensor(tensor=paddle.zeros(shape=self.hidden_size))
            self.v_core_attention_sdpa = ScaleDotProductAttention(
                layer_number=-1, causal=False, attention_dropout=self.attention_dropout
            )
            self.visual_cache = {}

    def apply_mi_rope(self, key_layer, media_offset_line, length_each_img):
        key_layer = rearrange(key_layer, "b h s d -> s b h d")
        if self.rotary_emb_core.inv_freq.place != key_layer.place:
            self.rotary_emb_core.inv_freq = self.rotary_emb_core.inv_freq.to(key_layer.place)
        rotary_pos_emb_max_seq_len = self.config.max_position_embeddings
        ntk_alpha = 1
        rotary_pos_emb = self.rotary_emb_core(rotary_pos_emb_max_seq_len, ntk_alpha=ntk_alpha)
        assert rotary_pos_emb is not None
        if isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = rotary_pos_emb
        else:
            rotary_pos_emb = (rotary_pos_emb,) * 2
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb
            image_pos = (media_offset_line[1:] - media_offset_line[:-1]).nonzero().squeeze(axis=1) + 1
            k_pos_emb = repeat(k_pos_emb[image_pos], "N_img b h d -> (N_img L) b h d", L=length_each_img)
            key_layer = apply_rotary_pos_emb_core(key_layer, k_pos_emb, use_fp32=True)
        key_layer = rearrange(key_layer, "s b h d -> b h s d")
        return key_layer

    def crossattention(self, query_layer, vision_features, media_offset, context_layer):
        """
        query_layer: [s b h d]
        vision_features: [b' lv d]
        context_layer: s b d
        """
        if vision_features is None or self.is_hyper_enabed == False:
            return context_layer
        context_layer_clone = context_layer.clone()
        vision_features = vision_features.contiguous()
        vision_features = self.v_kv_proj(vision_features)
        length_each_img = tuple(vision_features.shape)[1]
        sequence_length = tuple(query_layer.shape)[0]
        if sequence_length == 1:
            completion_flag = True
            media_offset = media_offset[:, -1:]
        else:
            completion_flag = False
            self.visual_cache["media_offset"] = media_offset
            self.visual_cache["vision_features"] = vision_features
        query_layer = rearrange(query_layer, "L B H D -> B H L D")
        assert sequence_length == tuple(media_offset.shape)[1], (sequence_length, tuple(media_offset.shape))
        gate_value = paddle.nn.functional.sigmoid(x=self.gate)
        for batch_id, begin_i, end_i in select_query(media_offset, sequence_length):
            assert begin_i == 0
            assert end_i == sequence_length, (end_i, sequence_length)
            curr_offset = media_offset[batch_id, end_i - 1]
            if not completion_flag:
                re_to_zero_media_offset = (media_offset[batch_id, :, 1] - curr_offset[0]).to(query_layer.place)
                query_shift = re_to_zero_media_offset.nonzero()[0].item()
                curr_mask = make_t2v_mask(
                    re_to_zero_media_offset[query_shift:], num_images=curr_offset[1] - curr_offset[0]
                )
                curr_mask = repeat(curr_mask, "s_q s_k -> B H s_q (s_k img_l)", B=1, H=1, img_l=length_each_img)
            else:
                curr_mask = None
                query_shift = 0
            curr_query_tokens = query_layer[batch_id, :, query_shift:].unsqueeze(axis=0).clone().contiguous()
            assert curr_offset[0] < tuple(vision_features.shape)[0]
            assert curr_offset[1] <= tuple(vision_features.shape)[0]
            curr_vision_kv: paddle.Tensor = rearrange(
                vision_features[curr_offset[0] : curr_offset[1]].clone(),
                "BL Lv (H KV D) -> KV 1 H (BL Lv) D",
                KV=2,
                H=self.num_key_value_heads,
            )
            key_layer = curr_vision_kv[0].contiguous()
            value_layer = curr_vision_kv[1].contiguous()
            key_layer = self.apply_mi_rope(
                key_layer,
                media_offset_line=self.visual_cache["media_offset"][batch_id, :, 1] - curr_offset[0],
                length_each_img=length_each_img,
            )
            key_layer = repeat_kv(key_layer, self.num_key_value_groups)
            value_layer = repeat_kv(value_layer, self.num_key_value_groups)
            v_context_layer = self.v_core_attention_sdpa(
                curr_query_tokens, key_layer, value_layer, attn_mask=curr_mask, order="bhsd"
            ).squeeze(axis=1)
            context_layer_clone[query_shift:, batch_id] = (
                context_layer[query_shift:, batch_id].clone() * (1 - gate_value) + v_context_layer * gate_value
            )
        return context_layer_clone

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        image_embeds=None,
        media_offset=None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        raise NotImplementedError(
            'We do not support eager model yet. Use attn_implementation == "flash_attention_2" or attn_implementation == "sdpa".'
        )
        bsz, q_len, _ = tuple(hidden_states.shape)
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        # query_states = query_states.view(bsz, q_len, self.num_heads, self.
        #     head_dim).transpose(perm=paddle_aux.transpose_aux_func(
        #     query_states.view(bsz, q_len, self.num_heads, self.head_dim).
        #     ndim, 1, 2))
        query_states = paddle.reshape(query_states, [bsz, q_len, self.num_heads, self.head_dim])
        query_states = paddle.transpose(query_states, perm=[0, 2, 1, 3])  # 交换第1和第2维度

        # key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
        #     self.head_dim).transpose(perm=paddle_aux.transpose_aux_func(
        #     key_states.view(bsz, q_len, self.num_key_value_heads, self.
        #     head_dim).ndim, 1, 2))
        key_states = paddle.reshape(key_states, [bsz, q_len, self.num_key_value_heads, self.head_dim])
        key_states = paddle.transpose(key_states, perm=[0, 2, 1, 3])  # 交换第1和第2维度

        # value_states = value_states.view(bsz, q_len, self.
        #     num_key_value_heads, self.head_dim).transpose(perm=paddle_aux.
        #     transpose_aux_func(value_states.view(bsz, q_len, self.
        #     num_key_value_heads, self.head_dim).ndim, 1, 2))
        value_states = paddle.reshape(value_states, [bsz, q_len, self.num_key_value_heads, self.head_dim])
        value_states = paddle.transpose(value_states, perm=[0, 2, 1, 3])  # 交换第1和第2维度

        kv_seq_len = tuple(key_states.shape)[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} for auto-regressive decoding with k/v caching, please make sure to initialize the attention class with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = paddle.matmul(
            x=query_states, y=key_states.transpose(perm=paddle_aux.transpose_aux_func(key_states.ndim, 2, 3))
        ) / math.sqrt(self.head_dim)
        if tuple(attn_weights.shape) != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {bsz, self.num_heads, q_len, kv_seq_len}, but is {tuple(attn_weights.shape)}"
            )
        if attention_mask is not None:
            if tuple(attention_mask.shape) != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {bsz, 1, q_len, kv_seq_len}, but is {tuple(attention_mask.shape)}"
                )
            attn_weights = attn_weights + attention_mask
        attn_weights = paddle.nn.functional.softmax(x=attn_weights, axis=-1, dtype="float32").to(query_states.dtype)
        attn_weights = paddle.nn.functional.dropout(x=attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = paddle.matmul(x=attn_weights, y=value_states)
        if tuple(attn_output.shape) != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {bsz, self.num_heads, q_len, self.head_dim}, but is {tuple(attn_output.shape)}"
            )
        attn_output = attn_output.transpose(perm=paddle_aux.transpose_aux_func(attn_output.ndim, 1, 2)).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.crossattention(
            query_states.transpose(perm=[1, 0, 1, 3]),
            image_embeds,
            media_offset,
            attn_output.transpose(perm=[1, 0, 2]),
        )
        attn_output = attn_output.transpose(perm=[1, 0, 2])
        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value


class HyperQwen2FlashAttention2(HyperQwen2Attention):
    """
    Qwen2 flash attention module, following Qwen2 attention module. This module inherits from `Qwen2Attention`
    as the weights of the module stays untouched. The only required change would be on the forward pass
    where it needs to correctly call the public API of flash attention and deal with padding tokens
    in case the input contains any of them. Additionally, for sliding window attention, we apply SWA only to the bottom
    config.max_window_layers layers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # >>>>>>        self._flash_attn_uses_top_left_mask = (not transformers.utils.
    #             is_flash_attn_greater_or_equal_2_10())

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        image_embeds=None,
        media_offset=None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        bsz, q_len, _ = tuple(hidden_states.shape)
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        # query_states = query_states.view(bsz, q_len, self.num_heads, self.
        #     head_dim).transpose(perm=paddle_aux.transpose_aux_func(
        #     query_states.view(bsz, q_len, self.num_heads, self.head_dim).
        #     ndim, 1, 2))
        query_states = paddle.reshape(query_states, [bsz, q_len, self.num_heads, self.head_dim])
        query_states = paddle.transpose(query_states, perm=[0, 2, 1, 3])  # 交换第1和第2维度

        # key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
        #     self.head_dim).transpose(perm=paddle_aux.transpose_aux_func(
        #     key_states.view(bsz, q_len, self.num_key_value_heads, self.
        #     head_dim).ndim, 1, 2))
        key_states = paddle.reshape(key_states, [bsz, q_len, self.num_key_value_heads, self.head_dim])
        key_states = paddle.transpose(key_states, perm=[0, 2, 1, 3])  # 交换第1和第2维度

        # value_states = value_states.view(bsz, q_len, self.
        #     num_key_value_heads, self.head_dim).transpose(perm=paddle_aux.
        #     transpose_aux_func(value_states.view(bsz, q_len, self.
        #     num_key_value_heads, self.head_dim).ndim, 1, 2))
        value_states = paddle.reshape(value_states, [bsz, q_len, self.num_key_value_heads, self.head_dim])
        value_states = paddle.transpose(value_states, perm=[0, 2, 1, 3])  # 交换第1和第2维度

        kv_seq_len = tuple(key_states.shape)[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} for auto-regressive decoding with k/v caching, please make sure to initialize the attention class with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
        cos, sin = self.rotary_emb(value_states, seq_len=rotary_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        use_sliding_windows = (
            _flash_supports_window_size
            and getattr(self.config, "sliding_window", None) is not None
            and kv_seq_len > self.config.sliding_window
            and self.config.use_sliding_window
        )
        if not _flash_supports_window_size:
            logger.warning_once(
                "The current flash attention version does not support sliding window attention, for a more memory efficient implementation make sure to upgrade flash-attn library."
            )
        if past_key_value is not None:
            cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
            if (
                getattr(self.config, "sliding_window", None) is not None
                and kv_seq_len > self.config.sliding_window
                and cache_has_contents
            ):
                slicing_tokens = 1 - self.config.sliding_window
                past_key = past_key_value[self.layer_idx][0]
                past_value = past_key_value[self.layer_idx][1]
                past_key = past_key[:, :, slicing_tokens:, :].contiguous()
                past_value = past_value[:, :, slicing_tokens:, :].contiguous()
                if tuple(past_key.shape)[-2] != self.config.sliding_window - 1:
                    raise ValueError(
                        f"past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got {tuple(past_key.shape)}"
                    )
                if attention_mask is not None:
                    attention_mask = attention_mask[:, slicing_tokens:]
                    attention_mask = paddle.concat(
                        x=[attention_mask, paddle.ones_like(x=attention_mask[:, -1:])], axis=-1
                    )
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout
        input_dtype = query_states.dtype
        if input_dtype == "float32":
            # >>>>>>            if torch.is_autocast_enabled():
            # >>>>>>                target_dtype = torch.get_autocast_gpu_dtype()
            if paddle.amp.auto_cast_enabled():
                target_dtype = paddle.get_device("gpu").dtype
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype
            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in {target_dtype}."
            )
            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)
        query_states = query_states.transpose(perm=paddle_aux.transpose_aux_func(query_states.ndim, 1, 2))
        key_states = key_states.transpose(perm=paddle_aux.transpose_aux_func(key_states.ndim, 1, 2))
        value_states = value_states.transpose(perm=paddle_aux.transpose_aux_func(value_states.ndim, 1, 2))
        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            use_sliding_windows=use_sliding_windows,
        )
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.crossattention(
            query_states.transpose(perm=[1, 0, 2, 3]),
            image_embeds,
            media_offset,
            attn_output.transpose(perm=[1, 0, 2]),
        )
        attn_output = attn_output.transpose(perm=[1, 0, 2])
        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
        use_sliding_windows=False,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
            use_sliding_windows (`bool`, *optional*):
                Whether to activate sliding window attention.
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            causal = self.is_causal and query_length != 1
        if use_sliding_windows and self.layer_idx >= self.config.max_window_layers:
            use_sliding_windows = False
        if attention_mask is not None:
            batch_size = tuple(query_states.shape)[0]
            (query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens) = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )
            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens
            if not use_sliding_windows:
                # >>>>>>                attn_output_unpad = flash_attn.flash_attn_varlen_func(
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    # softmax_scale = softmax_scale, causal=causal)
                    scale=softmax_scale,
                    causal=causal,
                )[0]
            else:
                # >>>>>>
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=
                    # max_seqlen_in_batch_k, dropout_p=dropout, softmax_scale
                    max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    scale=softmax_scale,
                    causal=causal,
                    window_size=(self.config.sliding_window, self.config.sliding_window),
                )[0]
            # >>>>>>            attn_output = flash_attn.bert_padding.pad_input(attn_output_unpad,
            #                 indices_q, batch_size, query_length)
            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        elif not use_sliding_windows:
            # >>>>>>            attn_output = flash_attn.flash_attn_func(query_states,
            attn_output = flash_attn_func(
                query_states,
                key_states,
                # value_states, dropout, softmax_scale=softmax_scale,
                value_states,
                dropout,
                causal=causal,
            )[0]
        else:
            # >>>>>>            attn_output = flash_attn.flash_attn_func(query_states,
            attn_output = flash_attn.flash_attn_func(
                query_states,
                # key_states, value_states, dropout, softmax_scale=softmax_scale,
                key_states,
                value_states,
                dropout,
                causal=causal,
                window_size=(self.config.sliding_window, self.config.sliding_window),
            )[0]
        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        batch_size, kv_seq_len, num_heads, head_dim = tuple(key_layer.shape)
        if kv_seq_len != tuple(attention_mask.shape)[-1]:
            attention_mask_num_tokens = tuple(attention_mask.shape)[-1]
            attention_mask = attention_mask[:, attention_mask_num_tokens - kv_seq_len :]
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        # >>>>>>        key_layer = flash_attn.bert_padding.index_first_axis(key_layer.
        #             reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)
        key_layer = index_first_axis(key_layer.reshape([batch_size * kv_seq_len, num_heads, head_dim]), indices_k)
        # >>>>>>        value_layer = flash_attn.bert_padding.index_first_axis(value_layer.
        #             reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)
        value_layer = index_first_axis(value_layer.reshape([batch_size * kv_seq_len, num_heads, head_dim]), indices_k)
        if query_length == kv_seq_len:
            # >>>>>>            query_layer = flash_attn.bert_padding.index_first_axis(query_layer
            #                 .reshape(batch_size * kv_seq_len, num_heads, head_dim),
            #                 indices_k)
            query_layer = index_first_axis(
                query_layer.reshape([batch_size * kv_seq_len, num_heads, head_dim]), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = paddle.arange(dtype="int32", end=batch_size + 1)
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(axis=1)
        else:
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = (
                # flash_attn.bert_padding.unpad_input(query_layer,
                # attention_mask))
                unpad_input(query_states, attention_mask)
            )
        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class HyperQwen2SdpaAttention(HyperQwen2Attention):
    """
    Qwen2 attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `Qwen2Attention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        image_embeds=None,
        media_offset=None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        # print('*&'*100)
        # print('output_attentions:',output_attentions)
        # print('attention_mask:',attention_mask)#(1,1,1,60)
        if output_attentions:  # false
            logger.warning_once(
                'Qwen2Model is using Qwen2SdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
        print(hidden_states.shape)
        bsz, q_len, _ = tuple(hidden_states.shape)
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        # query_states = query_states.view(bsz, q_len, self.num_heads, self.
        #     head_dim).transpose(perm=paddle_aux.transpose_aux_func(
        #     query_states.view(bsz, q_len, self.num_heads, self.head_dim).
        #     ndim, 1, 2))
        print("bsz:", bsz)
        print("qlen:", q_len)
        print("num_heads:", self.num_heads)
        print("head_dim:", self.head_dim)
        query_states = paddle.reshape(query_states, [bsz, q_len, self.num_heads, self.head_dim])
        query_states = paddle.transpose(query_states, perm=[0, 2, 1, 3])  # 交换 1 和 2 维度

        # key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
        #     self.head_dim).transpose(perm=paddle_aux.transpose_aux_func(
        #     key_states.view(bsz, q_len, self.num_key_value_heads, self.
        #     head_dim).ndim, 1, 2))
        key_states = paddle.reshape(key_states, [bsz, q_len, self.num_key_value_heads, self.head_dim])
        key_states = paddle.transpose(key_states, perm=[0, 2, 1, 3])  # 交换 1 和 2 维度

        # value_states = value_states.view(bsz, q_len, self.
        #     num_key_value_heads, self.head_dim).transpose(perm=paddle_aux.
        #     transpose_aux_func(value_states.view(bsz, q_len, self.
        #     num_key_value_heads, self.head_dim).ndim, 1, 2))

        value_states = paddle.reshape(value_states, [bsz, q_len, self.num_key_value_heads, self.head_dim])
        value_states = paddle.transpose(value_states, perm=[0, 2, 1, 3])  # 交换第1和第2维度

        kv_seq_len = tuple(key_states.shape)[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        # print('2'*100)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if attention_mask is not None:  # (1,1,1,60)
            if tuple(attention_mask.shape) != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {bsz, 1, q_len, kv_seq_len}, but is {tuple(attention_mask.shape)}"
                )
        # if query_states.device.type == 'cuda' and attention_mask is not None:
        # query_states = query_states.contiguous()
        # key_states = key_states.contiguous()
        # value_states = value_states.contiguous()
        attn_output = paddle.nn.functional.scaled_dot_product_attention(
            query=query_states,
            key=key_states,
            value=value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )
        attn_output = attn_output.transpose(perm=paddle_aux.transpose_aux_func(attn_output.ndim, 1, 2)).contiguous()
        # attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        attn_output = paddle.reshape(attn_output, [bsz, q_len, self.hidden_size])
        attn_output = self.crossattention(
            query_states.transpose(perm=[2, 0, 1, 3]),
            image_embeds,
            media_offset,
            attn_output.transpose(perm=[1, 0, 2]),
        )
        attn_output = attn_output.transpose(perm=[1, 0, 2])
        attn_output = self.o_proj(attn_output)
        return attn_output, None, past_key_value


QWEN2_ATTENTION_CLASSES = {
    "eager": HyperQwen2Attention,
    "flash_attention_2": HyperQwen2FlashAttention2,
    "sdpa": HyperQwen2SdpaAttention,
}


class HyperQwen2DecoderLayer(paddle.nn.Layer):
    def __init__(self, config: HyperQwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        if config.use_sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; unexpected results may be encountered."
            )
        self.is_hyper_enabled = layer_idx + 1 in config.hyper_layers
        self.self_attn = QWEN2_ATTENTION_CLASSES[config._attn_implementation](
            config, layer_idx, is_hyper_enabed=self.is_hyper_enabled
        )
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        image_embeds=None,
        media_offset=None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[paddle.Tensor, Optional[Tuple[paddle.Tensor, paddle.Tensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if image_embeds is not None and self.is_hyper_enabled:
            image_embeds = self.input_layernorm(image_embeds)
        else:
            image_embeds = media_offset = None
        # print('*&'*100)
        # print('attention_mask:',attention_mask)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,  # (1,1,1,60)
            position_ids=position_ids,
            image_embeds=image_embeds,
            media_offset=media_offset,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


QWEN2_START_DOCSTRING = """
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`HyperQwen2Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


# >>>>>>@transformers.utils.add_start_docstrings(
#     'The bare Qwen2 Model outputting raw hidden-states without any specific head on top.'
#     , QWEN2_START_DOCSTRING)
class Qwen2PreTrainedModel(paddlenlp.transformers.model_utils.PretrainedModel):
    config_class = HyperQwen2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["HyperQwen2DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    # def _init_weights(self, module):
    #     std = self.config.initializer_range
    #     if isinstance(module, paddle.nn.Linear):
    #         module.weight.data.normal_(mean=0.0, std=std)
    #         if module.bias is not None:
    #             module.bias.data.zero_()
    #     elif isinstance(module, paddle.nn.Embedding):
    #         module.weight.data.normal_(mean=0.0, std=std)
    #         if module.padding_idx is not None:
    #             module.weight.data[module.padding_idx].zero_()

    def _init_weights(self, layer):
        std = self.config.initializer_range
        if isinstance(layer, (paddle.nn.Linear, paddle.nn.Conv3D)):
            paddle.nn.initializer.Normal(mean=0.0, std=std)(layer.weight)
            if layer.bias is not None:
                paddle.nn.initializer.Constant(0.0)(layer.bias)
        elif isinstance(layer, paddle.nn.Embedding):
            paddle.nn.initializer.Normal(mean=0.0, std=std)(layer.weight)
            if layer._padding_idx is not None:
                with paddle.no_grad():
                    layer.weight[layer._padding_idx] = 0.0


QWEN2_INPUTS_DOCSTRING = """
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


# >>>>>>@transformers.utils.add_start_docstrings(
#     'The bare Qwen2 Model outputting raw hidden-states without any specific head on top.'
#     , QWEN2_START_DOCSTRING)
class HyperQwen2Model(Qwen2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: HyperQwen2Config
    """

    def __init__(self, config: HyperQwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = paddle.nn.Embedding(
            num_embeddings=config.vocab_size, embedding_dim=config.hidden_size, padding_idx=self.padding_idx
        )
        self.layers = paddle.nn.LayerList(
            sublayers=[HyperQwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        # self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # >>>>>>    @transformers.utils.add_start_docstrings_to_model_forward(
    #         QWEN2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        image_embeds=None,
        media_offset=None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, paddlenlp.transformers.model_outputs.BaseModelOutputWithPast]:
        print("^()" * 100)
        print("attention_mask", attention_mask.shape)
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            # print("%"*100)
            # print(input_ids.shape)
            batch_size, seq_length = tuple(input_ids.shape)  # (1,60)
        elif inputs_embeds is not None:
            # print("tuple(inputs_embeds.shape):",inputs_embeds.shape)
            batch_size, seq_length, _ = tuple(inputs_embeds.shape)
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if past_key_values is None:
            cache_position = paddle.arange(input_ids.shape[1])
        else:
            cache_position = paddle.to_tensor([seq_length - 1])

        if past_key_values is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        past_key_values_length = 0
        print("past_key_values:", past_key_values)
        # if use_cache:
        # use_legacy_cache = not isinstance(past_key_values, transformers
        #     .cache_utils.Cache)
        # use_legacy_cache = not isinstance(past_key_values, list) and all(isinstance(item, paddle.Tensor) for item in past_key_values)
        #             if use_legacy_cache:
        # >>>>>>                past_key_values = (transformers.cache_utils.DynamicCache.
        #                     from_legacy_cache(past_key_values))
        #             past_key_values_length = past_key_values.get_usable_length(
        #                 seq_length)
        if position_ids is None:
            device = input_ids.place if input_ids is not None else inputs_embeds.place
            position_ids = paddle.arange(
                start=past_key_values_length, end=seq_length + past_key_values_length, dtype="int64"
            )
            # position_ids = position_ids.unsqueeze(axis=0).view(-1, seq_length)
            position_ids = paddle.unsqueeze(position_ids, axis=0)
            position_ids = paddle.reshape(position_ids, [-1, seq_length])

        else:
            device = input_ids.place
            # position_ids = position_ids.view(-1, seq_length).astype(dtype='int64')
            # position_ids = position_ids.reshape(-1, seq_length).astype(dtype='int64')
            position_ids = paddle.reshape(position_ids, [-1, seq_length]).astype(dtype="int64")
        if inputs_embeds is None:
            print("^" * 100)
            inputs_embeds = self.embed_tokens(input_ids)
        if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right' this may lead to unexpected behaviour for Flash Attention version of Qwen2. Make sure to  call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )
        # print("^()"*100)
        # print('attention_mask',attention_mask)
        print(self._attn_implementation)
        if self._attn_implementation == "flash_attention_2":
            attention_mask = attention_mask if attention_mask is not None and 0 in attention_mask else None
        # elif self._attn_implementation == 'sdpa' and not output_attentions:
        # >>>>>>            attention_mask = (transformers.modeling_attn_mask_utils.
        #                 _prepare_4d_causal_attention_mask_for_sdpa(attention_mask,
        #                 (batch_size, seq_length), inputs_embeds,
        #                 past_key_values_length, sliding_window=self.config.
        #                 sliding_window))

        else:
            # >>>>>>            attention_mask = (transformers.modeling_attn_mask_utils.
            #                 _prepare_4d_causal_attention_mask(attention_mask, (
            #                 batch_size, seq_length), inputs_embeds,
            #                 past_key_values_length, sliding_window=self.config.
            #                 sliding_window))
            print("5" * 200)
            attention_mask = None
            min_dtype = paddle.finfo(paddle.float16).min
            # print("past_key_values_length:",past_key_values_length)

            attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=seq_length,
                target_length=seq_length,
                dtype=inputs_embeds.dtype,
                # device=device,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=batch_size,
            )
            # attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            #     attention_mask,
            #     sequence_length=seq_length,
            #     target_length=past_key_values.get_max_length(),
            #     dtype=dtype,
            #     # device=device,
            #     min_dtype=min_dtype,
            #     cache_position=cache_position,
            #     batch_size=batch_size,
            # )
            print("attention_mask", attention_mask)
        # print("^**"*100)
        # print('attention_mask',attention_mask)
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    image_embeds,
                    media_offset,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                print("hidden_states:", hidden_states)
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    image_embeds=image_embeds,
                    media_offset=media_offset,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return paddlenlp.transformers.model_outputs.BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class HyperQwen2ForCausalLM(Qwen2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = HyperQwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = paddle.nn.Linear(
            in_features=config.hidden_size, out_features=config.vocab_size, bias_attr=False
        )
        # self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    # >>>>>>    @transformers.utils.add_start_docstrings_to_model_forward(
    #         QWEN2_INPUTS_DOCSTRING)
    # >>>>>>    @transformers.utils.replace_return_docstrings(output_type=
    #         CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        image_embeds=None,
        media_offset=None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, paddlenlp.transformers.model_outputs.CausalLMOutputWithPast]:
        """
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen2ForCausalLM

        >>> model = Qwen2ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # print(self.model)   HyperQwen2Model
        # print('::'*100)
        # print('attention_mask',attention_mask)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,  # (1,1,1,60)
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            image_embeds=image_embeds,
            media_offset=media_offset,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.astype(dtype="float32")
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = paddle.nn.CrossEntropyLoss()
            # shift_logits = shift_logits.view(-1, self.config.vocab_size)
            # shift_labels = shift_labels.view(-1)
            shift_logits = paddle.reshape(shift_logits, [-1, self.config.vocab_size])
            shift_labels = paddle.reshape(shift_labels, [-1])

            shift_labels = shift_labels.to(shift_logits.place)
            loss = loss_fct(shift_logits, shift_labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return paddlenlp.transformers.model_outputs.CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            # >>>>>>            if isinstance(past_key_values, transformers.cache_utils.Cache):
            #                 cache_length = past_key_values.get_seq_length()
            #                 past_length = past_key_values.seen_tokens
            #                 max_cache_length = past_key_values.get_max_length()
            if past_key_values is not None and isinstance(past_key_values, list):
                # 确保所有元素都是 paddle.Tensor，并且获取每个 Tensor 的序列长度
                if all(isinstance(tensor, paddle.Tensor) for tensor in past_key_values):
                    # 计算 cache_length 和 max_cache_length
                    cache_length = len(past_key_values)  # 仍然是 Tensor 的数量
                    past_length = sum(tensor.numel() for tensor in past_key_values)  # 计算所有 Tensor 的元素总数
                    max_cache_length = max(tensor.shape[-2] for tensor in past_key_values)  # 获取最大序列长度（假设是 shape[-2]）
                else:
                    raise ValueError("past_key_values should be a list of paddle.Tensors")

            else:
                cache_length = past_length = tuple(past_key_values[0][0].shape)[2]
                max_cache_length = None
            if attention_mask is not None and tuple(attention_mask.shape)[1] > tuple(input_ids.shape)[1]:
                input_ids = input_ids[:, -(tuple(attention_mask.shape)[1] - past_length) :]
            elif past_length < tuple(input_ids.shape)[1]:
                input_ids = input_ids[:, past_length:]
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + tuple(input_ids.shape)[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.astype(dtype="int64").cumsum(axis=-1) - 1
            position_ids.masked_fill_(mask=attention_mask == 0, value=1)
            if past_key_values:
                position_ids = position_ids[:, -tuple(input_ids.shape)[1] :]
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "image_embeds": kwargs.get("image_embeds"),
                "media_offset": kwargs.get("media_offset"),
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(axis=0, index=beam_idx.to(past_state.place)) for past_state in layer_past
                ),
            )
        return reordered_past
