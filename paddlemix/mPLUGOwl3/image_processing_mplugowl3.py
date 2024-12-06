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
import math
import random
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np

# import paddle_aux
import paddle
import paddle.nn.functional as F
import paddlenlp
import PIL
import PIL.Image
import PIL.ImageSequence
from einops import rearrange, repeat
from paddlenlp.transformers.image_processing_utils import (
    BaseImageProcessor,
    BatchFeature,
)
from PIL import Image


def recursive_converter(converter, value):
    if isinstance(value, list):
        new_value = []
        for v in value:
            new_value += [recursive_converter(converter, v)]
        return new_value
    else:
        return converter(value)


def box_area(boxes):
    # 获取边界框的宽度和高度
    width = boxes[:, 2] - boxes[:, 0]
    height = boxes[:, 3] - boxes[:, 1]
    # 计算面积
    area = width * height
    return area


def custom_max(a, b):
    return paddle.where(a > b, a, b)


def custom_min(a, b):
    return paddle.where(a < b, a, b)


def box_iou(boxes1, area1, boxes2, eps=1e-05):
    # >>>>>>    area2 = torchvision.ops.boxes.box_area(boxes2)
    area1 = area1.astype("float32")
    boxes1 = boxes1.astype("float32")
    boxes2 = boxes2.astype("float32")

    area2 = box_area(boxes2).astype("float32")
    lt = custom_max(boxes1[:, None, :2], boxes2[:, :2])
    rb = custom_min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clip(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    iou = inter / (union + eps)
    return iou, union


available_anchor_strategy = ["docowl", "random", "highest", "last", "llava"]
grid_dict = {
    "grid_33": [
        (1, 1),
        (1, 2),
        (2, 1),
        (1, 3),
        (3, 1),
        (2, 2),
        (1, 4),
        (4, 1),
        (1, 5),
        (5, 1),
        (1, 6),
        (6, 1),
        (2, 3),
        (3, 2),
        (1, 7),
        (7, 1),
        (4, 2),
        (2, 4),
        (1, 8),
        (8, 1),
        (3, 3),
        (1, 9),
        (9, 1),
    ],
    "grid_squ_3x3": [(1, 1), (2, 2), (3, 3)],
    "grid_squ_4": [(2, 2), (1, 3), (1, 4), (3, 1), (4, 1)],
    "grid_squ_6": [(2, 2), (1, 3), (1, 4), (3, 1), (4, 1), (2, 3), (3, 2)],
    "grid_squ_2": [(2, 1)],
    "grid_squ_9": [
        (1, 1),
        (1, 2),
        (2, 1),
        (1, 3),
        (3, 1),
        (2, 2),
        (1, 4),
        (4, 1),
        (1, 5),
        (5, 1),
        (1, 6),
        (6, 1),
        (2, 3),
        (3, 2),
        (1, 7),
        (7, 1),
        (4, 2),
        (2, 4),
        (1, 8),
        (8, 1),
        (3, 3),
        (1, 9),
        (9, 1),
    ],
}
cut_prompt_template_dict = {
    "v0": lambda img_token, h, w: f"".join([f"{img_token}" for i in range(h) for j in range(w)]),
    "v1": lambda img_token, h, w: f"Cut to {h} rows {w} columns, "
    + " ".join([f"subimg({i},{j}){img_token}" for i in range(h) for j in range(w)]),
    "v1_global": lambda img_token, h, w: f"Cut to {h} rows {w} columns with a global view, "
    + " ".join([f"subimg({i},{j}){img_token}" for i in range(h) for j in range(w)] + [f"global_view{img_token}"]),
    "v2_global": lambda img_token, h, w: f"""Cut to {h} rows {w} columns with a global view
"""
    + "\n".join([" ".join([f"subimg({i},{j}){img_token}" for j in range(w)]) for i in range(h)])
    + f"""
global_view{img_token}""",
    "v3": lambda img_token, h, w: f"<|start_cut|>{h}*{w}"
    + " ".join([f"{img_token}" for i in range(h) for j in range(w)])
    + "<|end_cut|>",
    "v3_global": lambda img_token, h, w: f"""<|start_cut|>{h}*{w}
"""
    + "\n".join([" ".join([f"{img_token}" for j in range(w)]) for i in range(h)])
    + f"""
{img_token}<|end_cut|>""",
}


def anchor_rank(anchors, anchors_areas, input_image_size, eps=1e-05):
    input_image_bbox = paddle.to_tensor(data=[0, 0, input_image_size[1], input_image_size[0]]).unsqueeze(axis=0)
    boxes1 = anchors
    boxes2 = input_image_bbox
    boxes3 = anchors.clone()
    boxes3[:, 3] = input_image_size[0] / input_image_size[1] * anchors[:, 2]
    area1 = anchors_areas
    iou, _ = box_iou(boxes1, area1, boxes2)
    iou = iou.squeeze(axis=1)
    shape_iou, _ = box_iou(boxes1, area1, boxes3)
    shape_iou = shape_iou.diag()
    index = paddle.argmax(x=shape_iou * 100 + iou, axis=0)
    return index


def select_best_resolution(anchors, anchors_areas, input_image_size):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_size = input_image_size[1], input_image_size[0]
    possible_resolutions = [(_[2], _[3]) for _ in anchors]
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")
    index = 0
    for i, (width, height) in enumerate(possible_resolutions):
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = width * height - effective_resolution
        if (
            effective_resolution > max_effective_resolution
            or effective_resolution == max_effective_resolution
            and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = width, height
            index = i
    return index


def build_cut_shape_indices(cut_shape):
    cut_shape_indices = []
    for shape in cut_shape:
        n = shape[0] * shape[1]
        indices = paddle.concat(
            x=[repeat(paddle.to_tensor(data=shape), "l -> n l", n=n), paddle.arange(end=n).unsqueeze(axis=1)], axis=1
        )
        assert tuple(indices.shape)[0] == n
        assert tuple(indices.shape)[1] == 3
        cut_shape_indices.append(indices)
    cut_shape_indices = paddle.concat(x=cut_shape_indices, axis=0).astype(dtype="int64")
    return cut_shape_indices


class AnchorResize(paddle.nn.Layer):

    # >>>>>>    def __init__(self, image_size, anchors, interpolation=torchvision.
    #         transforms.transforms.InterpolationMode.BILINEAR, antialias=None,
    #         anchor_strategy='docowl'):
    def __init__(self, image_size, anchors, interpolation="bilinear", antialias=None, anchor_strategy="docowl"):
        super().__init__()
        self.image_size = image_size
        self.anchors = paddle.to_tensor(
            data=[[0, 0, _[1] * image_size[1], _[0] * image_size[0]] for _ in anchors], stop_gradient=not False
        )
        # >>>>>>        self.anchor_areas = torchvision.ops.boxes.box_area(self.anchors)
        self.anchor_areas = box_area(self.anchors)
        self.interpolation = interpolation
        self.antialias = antialias
        self.anchor_strategy = anchor_strategy
        assert self.anchor_strategy in available_anchor_strategy

    def resize_global(self, img):
        # >>>>>>        return torchvision.transforms.functional.resize(img, self.
        #             image_size, self.interpolation, max_size=None, antialias=self.
        #             antialias)
        image_np = np.array(img)
        image_tensor = paddle.to_tensor(image_np, dtype="float32")
        image_tensor = image_tensor.transpose([2, 0, 1])  # 变成 (3, 500, 500)
        if self.interpolation == "bilinear" or "bicubic":
            image_tensor = image_tensor.unsqueeze(0)  # 变成 (1, 3, 500, 500)
        return F.interpolate(image_tensor, size=self.image_size, mode=self.interpolation, align_corners=False)

    def forward(self, img, skip_resize=False):
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
        """
        if self.anchor_strategy == "docowl":
            selected_anchor = anchor_rank(self.anchors, self.anchor_areas, (img.size[1], img.size[0]))
        elif self.anchor_strategy == "random":
            selected_anchor = random.randint(0, len(self.anchors) - 1)
        elif self.anchor_strategy == "highest":
            selected_anchor = paddle.argmax(
                x=self.anchors[:, 2] * self.anchors[:, 3] * 100 - paddle.abs(x=self.anchors[:, 2] - self.anchors[:, 3])
            )
        elif self.anchor_strategy == "last":
            selected_anchor = len(self.anchors) - 1
        elif self.anchor_strategy == "llava":
            selected_anchor = select_best_resolution(self.anchors, self.anchor_areas, (img.size[1], img.size[0]))
        else:
            selected_anchor = None
        assert selected_anchor is not None
        target_size = self.anchors[selected_anchor][2:].tolist()
        if skip_resize:
            return selected_anchor
        # >>>>>>        return torchvision.transforms.functional.resize(img, [target_size[1
        #             ], target_size[0]], self.interpolation, max_size=None,
        #             antialias=self.antialias), selected_anchor
        image_np = np.array(img)
        image_tensor = paddle.to_tensor(image_np, dtype="float32")
        image_tensor = image_tensor.transpose([2, 0, 1])  # 变成 (3, 500, 500)
        if self.interpolation == "bilinear" or "bicubic":
            image_tensor = image_tensor.unsqueeze(0)  # 变成 (1, 3, 500, 500)
        return (
            F.interpolate(
                image_tensor, size=[target_size[1], target_size[0]], mode=self.interpolation, align_corners=False
            ),
            selected_anchor,
        )

    def __repr__(self) -> str:
        detail = f"(size={self.image_size}, anchor={self.anchors}, interpolation={self.interpolation.value}, antialias={self.antialias})"
        return f"{self.__class__.__name__}{detail}"


class CutMixin:
    def __init__(
        self,
        cut_cfg={
            "anchors": "grid_squ_6",
            "anchor_strategy": "docowl",
            "cut_prompt": "v3",
            "add_global": True,
            "cut_prob": 1.0,
        },
    ) -> None:
        if cut_cfg is None:
            self.cut_enable = False
            return
        else:
            self.cut_enable = True
        image_size = self.image_size
        anchors = cut_cfg.get("anchors", "grid_33")
        anchor_strategy = cut_cfg.get("anchor_strategy", "docowl")
        cut_prompt = cut_cfg.get("cut_prompt", "v0")
        self.cut_prob = cut_cfg.get("cut_prob", 1.0)
        self.force_shape_cut = cut_cfg.get("force_shape_cut", False)
        force_shape_cut_anchors = cut_cfg.get("force_shape_cut_anchors", "force_shape_cut_anchors")
        self.add_global = cut_cfg.get("add_global", False)
        if isinstance(image_size, int):
            image_size = image_size, image_size
        self.image_size = image_size
        if anchors in grid_dict:
            anchors = grid_dict[anchors]
        else:
            anchors = eval(anchors)
        self.anchors = [tuple(_) for _ in anchors]
        self.anchor_max = max([max(_) for _ in self.anchors])
        self.resizer = AnchorResize(
            image_size=image_size, anchors=anchors, interpolation="bicubic", anchor_strategy=anchor_strategy
        )
        if force_shape_cut_anchors in grid_dict:
            force_shape_cut_anchors = grid_dict[force_shape_cut_anchors]
        else:
            force_shape_cut_anchors = eval(force_shape_cut_anchors)
        self.force_shape_cut_anchors = [tuple(_) for _ in force_shape_cut_anchors]
        self.force_shape_cut_anchors_max = max([max(_) for _ in self.force_shape_cut_anchors])
        # >>>>>>        self.old_resizer = torchvision.transforms.Resize(image_size,
        #             interpolation=torchvision.transforms.transforms.
        #             InterpolationMode.BICUBIC)
        self.old_resizer = paddle.vision.transforms.Resize(size=image_size, interpolation="bicubic")
        # >>>>>>        self.image_transform = torchvision.transforms.Compose(self.
        #             image_transform.transforms[1:])
        self.image_transform = paddle.vision.transforms.Compose(self.image_transform.transforms[1:])
        if self.add_global:
            self.cut_prompt_template = cut_prompt_template_dict[cut_prompt + "_global"]
        else:
            self.cut_prompt_template = cut_prompt_template_dict[cut_prompt]
        self.media_tokens = ["<|image|>", "<|video|>"]

    def _process_image(self, images):
        new_images = []
        cut_shape = []
        for image in images:
            print(len(images))
            raw_image = image
            print(raw_image)
            print("-" * 100)
            image, selected_anchor = self.resizer(image)
            print(image.shape)
            print("-" * 100)
            image_input = self.image_transform(image)
            image_input = image_input[0]
            print(image_input.shape)
            cut_shape.append(
                (tuple(image_input.shape)[1] // self.image_size[0], tuple(image_input.shape)[2] // self.image_size[1])
            )
            image_input = rearrange(
                image_input, "C (num_h h) (num_w w) -> (num_h num_w) C h w", h=self.image_size[0], w=self.image_size[1]
            )
            new_images.append(image_input)
            print("1:", image_input.shape)
            if self.add_global:
                new_images.append(self.image_transform(self.resizer.resize_global(raw_image)))
                print("2:", new_images[1].shape)
                cut_shape.append((1, 1))
        print("cutshape:", cut_shape)
        new_images = paddle.concat(x=new_images, axis=0)
        cut_shape_indices = build_cut_shape_indices(cut_shape)
        return new_images, cut_shape, cut_shape_indices


class TensorType(Enum):
    PADDLE = "paddle"
    TORCH = "torch"


# >>>>>>class mPLUGOwl3BatchFeature(transformers.image_processing_utils.BatchFeature):
class mPLUGOwl3BatchFeature(BatchFeature):
    """
    Extend from BatchFeature for supporting various image size
    """

    def __init__(
        self,
        data: Optional[Dict[str, Any]] = None,
        # tensor_type:Union[None, str, transformers.utils.TensorType]=None):
        tensor_type: Union[None, str, TensorType] = None,
    ):
        super().__init__(data)
        self.convert_to_tensors(tensor_type=tensor_type)

    # def convert_to_tensors(self, tensor_type: Optional[Union[str,transformers.utils.TensorType]]=None):
    def convert_to_tensors(self, tensor_type: Optional[Union[str, TensorType]] = None):
        if tensor_type is None:
            return self

        is_tensor = lambda x: isinstance(x, paddle.Tensor)
        as_tensor = paddle.to_tensor

        def converter(value):
            try:
                if not is_tensor(value):
                    tensor = as_tensor(value)
                    return tensor
            except:
                if key == "overflowing_values":
                    raise ValueError("Unable to create tensor returning overflowing values of different lengths. ")
                raise ValueError(
                    "Unable to create tensor, you should probably activate padding with 'padding=True' to have batched tensors with the same length."
                )

        for key, value in self.items():
            self[key] = recursive_converter(converter, value)
        return self

    def to(self, *args, **kwargs) -> "mPLUGOwl3BatchFeature":
        # >>>>>>        transformers.utils.requires_backends(self, ['torch'])

        def cast_tensor(v):
            #             if paddle.is_floating_point(x=v):
            #                 """Class Method: *.to, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
            # >>>>>>                return v.to(*args, **kwargs)
            #             elif device is not None:
            #                 return v.to(device=device)
            #             else:
            #                 return v
            if isinstance(v, paddle.Tensor):
                # For floating point tensors
                if v.dtype in [paddle.float32, paddle.float64]:
                    if "dtype" in kwargs:
                        v = v.cast(kwargs["dtype"])
                    if "place" in kwargs:
                        v = v.place(kwargs["place"])
                    return v
                # For non-floating point tensors, only handle device
                elif "place" in kwargs:
                    return v.place(kwargs["place"])
            return v

        new_data = {}
        # Handle place (device in paddle)
        place = kwargs.get("place")
        if place is None and len(args) > 0:
            arg = args[0]
            if isinstance(arg, str) or isinstance(arg, paddle.CPUPlace) or isinstance(arg, paddle.CUDAPlace):
                place = arg
            else:
                raise ValueError(f"Attempting to cast a BatchFeature to type {str(arg)}. This is not supported.")

        #         device = kwargs.get('device')
        #         if device is None and len(args) > 0:
        #             arg = args[0]
        # # >>>>>>            if transformers.utils.is_torch_dtype(arg):
        #             if isinstance(arg, paddle.Tensor):
        #                 pass
        # # >>>>>>            elif isinstance(arg, str) or transformers.utils.is_torch_device(arg
        # #                 ) or isinstance(arg, int):
        # #                 device = arg
        #             elif isinstance(arg, str):
        #                 # 如果是字符串，可以直接使用该字符串作为设备标识
        #                 device = arg
        #             elif isinstance(arg, (int, paddle.device.Device)):
        #                 if isinstance(arg, int):
        #                     device = f'gpu:{arg}' if arg >= 0 else 'cpu'
        #                 else:
        #                     device = str(arg)
        #             else:
        #                 raise ValueError(
        #                     f'Attempting to cast a BatchFeature to type {str(arg)}. This is not supported.'
        #                     )
        for k, v in self.items():
            new_data[k] = recursive_converter(cast_tensor, v)
        self.data = new_data
        return self


# >>>>>>class mPLUGOwl3ImageProcessor(transformers.image_processing_utils.
#     BaseImageProcessor, CutMixin):
class mPLUGOwl3ImageProcessor(BaseImageProcessor, CutMixin):
    model_input_names = ["pixel_values"]

    def __init__(self, image_size, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], **kwargs):
        # super().__init__(**kwargs)
        self.image_size = image_size
        self.mean = mean
        self.std = std
        # >>>>>>        self.image_transform = torchvision.transforms.Compose([torchvision.
        #             transforms.Resize((image_size, image_size), interpolation=Image
        #             .BICUBIC), torchvision.transforms.ToTensor(), torchvision.
        #             transforms.Normalize(mean, std)])
        self.image_transform = paddle.vision.transforms.Compose(
            [
                paddle.vision.transforms.Resize(size=(image_size, image_size), interpolation="bicubic"),
                paddle.vision.transforms.ToTensor(),
                paddle.vision.transforms.Normalize(mean=mean, std=std),
            ]
        )

        CutMixin.__init__(self)

    def preprocess(
        self, images: Union[Image.Image, List[Image.Image]], cut_enable=True, **kwargs
    ) -> mPLUGOwl3BatchFeature:
        if isinstance(images, Image.Image):
            images_list = [images]
        else:
            images_list = images
        if self.cut_enable and cut_enable:
            image_data, cut_shape, cut_shape_indices = self._process_image(images_list)
        else:
            image_data = [self.image_transform(self.resizer.resize_global(image)) for image in images_list]
            image_data = paddle.stack(x=image_data, axis=0)
            cut_shape = cut_shape_indices = None
        return mPLUGOwl3BatchFeature(
            data={"pixel_values": image_data, "cut_shape": cut_shape, "cut_shape_indices": cut_shape_indices}
        )

    def to_dict(self):
        # encoder_dict = super().to_dict()
        encoder_dict = {}
        pop_keys = ["image_transform", "resizer", "old_resizer", "cut_prompt_template"]
        for pk in pop_keys:
            encoder_dict.pop(pk, None)
        return encoder_dict


# >>>>>>transformers.AutoImageProcessor.register('mPLUGOwl3ImageProcessor',
#     mPLUGOwl3ImageProcessor)
