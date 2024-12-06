import sys
sys.path.append('/home/aistudio/paddle_test/mPLUGOwl3/utils')
import paddle_aux
import paddle
import paddlenlp
from paddlenlp.transformers.processing_utils import ProcessorMixin
"""
Processor class for mPLUGOwl3.
"""
from typing import List, Optional, Union, Dict, Any
import warnings
import re
# from .image_processing_mplugowl3 import mPLUGOwl3BatchFeature, mPLUGOwl3ImageProcessor
from .image_processing_mplugowl3 import mPLUGOwl3BatchFeature, mPLUGOwl3ImageProcessor,TensorType
OWL_MEDIA_TOKEN = ['<|image|>']


class MediaIndicesHelper:

    def __init__(self, tokenizer) ->None:
        self.media_position = []
        self.tokenizer = tokenizer

    def has_media(self, text, media_tokens=None):
        if media_tokens is None:
            media_tokens = OWL_MEDIA_TOKEN
        has_media_flag = any([(media_token == text) for media_token in
            media_tokens])
        if any([(media_token in text) for media_token in media_tokens]):
            assert has_media_flag, text
        return has_media_flag

    def add_media(self, text_chunk, text=None, tokenize_fn=None):
        assert tokenize_fn is not None
        assert text is not None
        assert text in OWL_MEDIA_TOKEN
        media_token_ids = tokenize_fn(text)
        start = len(text_chunk)
        end = start + len(media_token_ids)
        self.media_position.append([start, end])
        text_chunk.extend(media_token_ids)
        return len(media_token_ids)

    def cal_media_offset(self, input_ids):
        if len(self.media_position) == 0:
            return paddle.ones_like(x=input_ids) * -1000000
        media_starts = paddle.to_tensor(data=[_[0] for _ in self.
            media_position]).reshape(1, -1)
        rng = paddle.arange(end=tuple(input_ids.shape)[0]).reshape(-1, 1)
        matrix = (rng > media_starts).sum(axis=1)
        return matrix

    def len_images(self):
        return len(self.media_position)


# >>>>>>class mPLUGOwl3Processor(transformers.processing_utils.ProcessorMixin):
class mPLUGOwl3Processor(ProcessorMixin):
    """
    Args:
        image_processor ([`mPLUGOwl3ImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerWrapper`], *optional*):
            The tokenizer is a required input.
    """
    attributes = ['image_processor', 'tokenizer']
    image_processor_class = 'mPLUGOwl3ImageProcessor'
    tokenizer_class = 'AutoTokenizer'

    def __init__(self, image_processor: mPLUGOwl3ImageProcessor=None,
        tokenizer=None, prompt_style='chatml', inference_mode=True,
        addition_eod='<|endoftext|>'):
        super().__init__(image_processor, tokenizer)
        self.image_processor: mPLUGOwl3ImageProcessor
        self.prompt_style = prompt_style
        self.inference_mode = inference_mode
        self.media_tokens = ['<|image|>']
        self.addition_eod = addition_eod

    def build_text_qwen(self, messages):
        im_start, im_end = '<|im_start|>', '<|im_end|>'
        text = []
        for num_turn, message in enumerate(messages):
            if num_turn == 0 and message['role'] != 'system':
                if self.prompt_style != 'plain':
                    text.append({'text': f'{im_start}system\n{im_end}',
                        'label': 0})
            if message['role'] == 'system':
                if self.prompt_style != 'plain':
                    text.append({'text':
                        f"{im_start}system\n{message['content']}{im_end}",
                        'label': 0})
            elif message['role'] == 'user':
                if self.prompt_style != 'plain':
                    content = f"\n{im_start}user\n{message['content']}{im_end}"
                else:
                    content = message['content']
                pattern = '|'.join(map(re.escape, self.media_tokens))
                chunk_strs = re.split(f'({pattern})', content)
                for chunk_str in chunk_strs:
                    text.append({'text': chunk_str, 'label': 0})
            elif message['role'] == 'assistant':
                if self.prompt_style != 'plain':
                    text.append({'text': f'\n{im_start}assistant\n',
                        'label': 0})
                    text.append({'text': f"{message['content']}{im_end}",
                        'label': 1})
                else:
                    text.append({'text': f"{message['content']}", 'label': 1})
                text.append({'text': self.addition_eod, 'label': 1})
            else:
                raise NotImplementedError
        if self.inference_mode:
            while text and text[-1]['label'] == 1:
                text.pop()
        return text

    def wrapped_tokenize(self, text):
        return self.tokenizer(text).input_ids

    def encode_text_sft(self, texts):
        enc_chunk = []
        label_chunk = []
        enc_length = 0
        num_images = 0
        media_helper = MediaIndicesHelper(tokenizer=self.tokenizer)
        for current_ti, text_chunk in enumerate(texts):
            text = text_chunk['text']
            label = text_chunk['label']
            if not media_helper.has_media(text):
                curr_chunk = self.wrapped_tokenize(text)
                if label == 1:
                    enc_length += len(curr_chunk)
                    enc_chunk += curr_chunk
                    label_chunk += [label] * len(curr_chunk)
                else:
                    enc_length += len(curr_chunk)
                    enc_chunk += curr_chunk
                    label_chunk += [label] * len(curr_chunk)
            else:
                add_length = media_helper.add_media(enc_chunk, text=text,
                    tokenize_fn=self.wrapped_tokenize)
                enc_length += add_length
                label_chunk += [label] * add_length
                num_images += 1
        enc_chunk = paddle.to_tensor(data=enc_chunk).astype(dtype='int64')
        media_offset = []
        media_before = 0
        for i, _ in enumerate([media_helper]):
            mo = _.cal_media_offset(enc_chunk)
            media_offset.append(paddle.concat(x=[(paddle.ones(shape=[tuple(
                mo.shape)[0], 1]) * media_before).astype(dtype='int64').to(
                mo.place), (mo + media_before).unsqueeze(axis=1)], axis=1))
            media_before += _.len_images()
        media_offset = paddle.stack(x=media_offset, axis=0)
        return {'input_ids': enc_chunk.unsqueeze(axis=0), 'media_offset':
            media_offset}

    def __call__(self, messages, images=None, videos=None, max_length:
        Optional[int]=None, cut_enable=True, 
        # return_tensors: Optional[Union[str, transformers.utils.TensorType]]=transformers.utils.TensorType.PYTORCH, **kwargs) ->mPLUGOwl3BatchFeature:
        return_tensors: Optional[Union[str, TensorType]]=TensorType.PADDLE, **kwargs) ->mPLUGOwl3BatchFeature:
        medias = []
        if videos is not None:
            medias.extend([{'type': 'video', 'content': video,
                'use_video_span': True} for video in videos])
        if images is not None:
            medias.extend([{'type': 'image', 'content': image} for image in
                images])
        if len(medias):
            image_tensor_list = []
            pattern = '(<\\|image\\|>|<\\|video\\|>)'
            image_token_ptr = 0
            media_layout = []
            for message in messages:
                text_list = re.split(pattern, message['content'])
                text = ''
                for text_content in text_list:
                    if text_content in ['<|image|>', '<|video|>']:
                        media_item = medias[image_token_ptr]
                        image_token_ptr += 1
                        if text_content == '<|image|>':
                            assert media_item['type'] == 'image'
                            image = media_item['content']
                            image_inputs = self.image_processor([image],
                                cut_enable=cut_enable, return_tensors=
                                return_tensors)
                            if image_inputs.get('cut_shape', None) is not None:
                                cut_shape = image_inputs['cut_shape']
                                cut_text = (self.image_processor.
                                    cut_prompt_template(img_token=
                                    '<|image|>', h=cut_shape[0][0], w=
                                    cut_shape[0][1]))
                                text += cut_text
                                image_tensor_list.append(image_inputs[
                                    'pixel_values'])
                            else:
                                text += text_content
                        elif text_content == '<|video|>':
                            assert media_item['type'] == 'video'
                            video = media_item['content']
                            use_video_span = media_item['use_video_span']
                            image_tensor = self.image_processor(video,
                                cut_enable=False)['pixel_values']
                            image_tensor_list.append(image_tensor)
                            num_video_frame = tuple(image_tensor.shape)[0]
                            if use_video_span:
                                text_content = ('<|start_video_frame|>' + 
                                    '<|image|>' * num_video_frame +
                                    '<|end_video_frame|>')
                            else:
                                text_content = '<|image|>' * num_video_frame
                            text += text_content
                    else:
                        text += text_content
                message['content'] = text
            assert image_token_ptr == len(medias), (image_token_ptr, len(
                medias))
            assert all(len(tuple(_.shape)) == 4 for _ in image_tensor_list), [
                tuple(_.shape) for _ in image_tensor_list]
            num_image_tokens = sum([_['content'].count('<|image|>') for _ in
                messages])
            num_image_shapes = sum([tuple(_.shape)[0] for _ in
                image_tensor_list])
            assert num_image_tokens == num_image_shapes, (messages, [tuple(
                _.shape) for _ in image_tensor_list])
        image_tensor_list = paddle.concat(x=image_tensor_list, axis=0)
        text = self.build_text_qwen(messages)
        model_inputs = self.encode_text_sft(text)
        if len(medias) is not None:
            model_inputs.update({'pixel_values': image_tensor_list})
        return mPLUGOwl3BatchFeature(model_inputs)

    def check_media(self, images, messages):
        media_num = 0 if images is None else len(images)
        media_count = sum([message['content'].count('<|image|>') for
            message in messages])
        assert media_num == media_count

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        output_ids = args[0]
        result_text = []
        for result in output_ids:
            result = result[result != 0]
            if result[0] == self.tokenizer.bos_id:
                result = result[1:]
            if result[-1] == self.tokenizer.eos_id:
                result = result[:-1]
            result_text.append(self.tokenizer.decode(result, *args[1:], **
                kwargs).strip())
        return result_text

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        result = args[0]
        result = result[result != 0]
        if result[0] == self.tokenizer.bos_id:
            result = result[1:]
        if result[-1] == self.tokenizer.eos_id or hasattr(self.tokenizer,
            'eot_id') and result[-1] == self.tokenizer.eot_id:
            result = result[:-1]
        return self.tokenizer.decode(result, *args[1:], **kwargs).strip()

    def _convert(self, input_str, max_inp_length: Optional[int]=None):
        if self.version > 2.5 or not getattr(self.tokenizer,
            'add_bos_token', False):
            input_ids = self.tokenizer.encode(input_str)
        else:
            input_ids = [self.tokenizer.bos_id] + self.tokenizer.encode(
                input_str)
        if max_inp_length is not None:
            input_ids = input_ids[:max_inp_length]
        input_ids = paddle.to_tensor(data=input_ids, dtype='int32')
        start_cond = (input_ids == self.tokenizer.im_start_id) | (input_ids ==
            self.tokenizer.slice_start_id)
        end_cond = (input_ids == self.tokenizer.im_end_id) | (input_ids ==
            self.tokenizer.slice_end_id)
# >>>>>>        image_start_tokens = torch.where(start_cond)[0]
        image_start_tokens = paddle.nonzero(start_cond)[:,0]
        image_start_tokens += 1
# >>>>>>        image_end_tokens = torch.where(end_cond)[0]
        image_end_tokens = paddle.nonzero(end_cond)[:,0]
        valid_image_nums = max(len(image_start_tokens), len(image_end_tokens))
        image_bounds = paddle.hstack(x=[image_start_tokens[:
            valid_image_nums].unsqueeze(axis=-1), image_end_tokens[:
            valid_image_nums].unsqueeze(axis=-1)])
        return input_ids, image_bounds

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names +
            image_processor_input_names))

    def pad(self, inputs, max_length=None, padding_value=0, padding_side='left'
        ):
        items = []
        if isinstance(inputs[0], list):
            assert isinstance(inputs[0][0], paddle.Tensor)
            for it in inputs:
                for tr in it:
                    items.append(tr)
        else:
            assert isinstance(inputs[0], paddle.Tensor)
            items = inputs
        batch_size = len(items)
        shape = tuple(items[0].shape)
        dim = len(shape)
        assert dim <= 2
        if max_length is None:
            max_length = 0
        max_length = max(max_length, max(tuple(item.shape)[-1] for item in
            items))
        min_length = min(tuple(item.shape)[-1] for item in items)
        dtype = items[0].dtype
        if dim == 0:
            return paddle.stack(x=[item for item in items], axis=0), [0]
        elif dim == 1:
            if max_length == min_length:
                return paddle.stack(x=[item for item in items], axis=0), [0
                    ] * batch_size
            tensor = paddle.zeros(shape=(batch_size, max_length), dtype=dtype
                ) + padding_value
        else:
            tensor = paddle.zeros(shape=(batch_size, max_length, shape[-1]),
                dtype=dtype) + padding_value
        padding_length = []
        for i, item in enumerate(items):
            if dim == 1:
                if padding_side == 'left':
                    tensor[i, -len(item):] = item.clone()
                else:
                    tensor[i, :len(item)] = item.clone()
            elif dim == 2:
                if padding_side == 'left':
                    tensor[i, -len(item):, :] = item.clone()
                else:
                    tensor[i, :len(item), :] = item.clone()
            padding_length.append(tuple(tensor.shape)[-1] - len(item))
        return tensor, padding_length
