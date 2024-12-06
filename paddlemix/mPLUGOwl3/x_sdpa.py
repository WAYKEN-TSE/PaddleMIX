import paddle
from icecream import ic
from einops import rearrange


class ScaleDotProductAttention(paddle.nn.Layer):

    def __init__(self, layer_number, causal=False, softmax_scale=None,
        attention_dropout=0.0):
        super().__init__()
        self.layer_number = layer_number
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, q, k, v, attn_mask=None, order='sbhd'):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q, k, v: The tensor containing the query, key, and value. (B, S, H, D)
        """
        if order == 'sbhd':
            q, k, v = [rearrange(x, 's b h d -> b h s d').contiguous() for
                x in (q, k, v)]
        elif order == 'bhsd':
            pass
        if attn_mask is not None:
            attn_mask = (~attn_mask.clone().astype(dtype='bool')).contiguous()
        else:
            attn_mask = None
        if self.training:
            if self.causal:
                assert tuple(q.shape)[-2] == tuple(k.shape)[-2]
            is_causal = self.causal
            dropout_p = self.dropout_p
        else:
            if self.causal:
                is_causal = tuple(q.shape)[-2] == tuple(k.shape)[-2]
            else:
                is_causal = self.causal
            dropout_p = 0.0
        assert self.softmax_scale == None or self.softmax_scale == paddle.utils.try_import(
            'math').sqrt(q.shape[-1]
            ), 'Fault: The scale parameter defaults to the square root of the last dimension of query, not allowed manually set'
        o = paddle.nn.functional.scaled_dot_product_attention(query=q, key=
            k, value=v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal
            =is_causal)
        o = rearrange(o, 'B Head L D -> L B (Head D)').contiguous()
        return o
