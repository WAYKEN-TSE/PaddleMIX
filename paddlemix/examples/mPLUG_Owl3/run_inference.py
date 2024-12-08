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

from PIL import Image
import paddle
from paddlenlp.transformers import Qwen2Tokenizer
from paddlemix.models.mPLUGOwl3.configuration_mplugowl3 import mPLUGOwl3Config
from paddlemix.models.mPLUGOwl3.modeling_mplugowl3 import mPLUGOwl3Model
# from paddlemix.models.mPLUGOwl3.processing_mplugowl3 import mPLUGOwl3Processor
# from paddlemix.models.mPLUGOwl3.image_processing_mplugowl3 import mPLUGOwl3ImageProcessor

#model_path = 'mPLUG/mPLUG-Owl3-7B-241101'
model_path = 'mPLUG-Owl3-7B-241101'

config = mPLUGOwl3Config.from_pretrained(model_path)
# print(config)
model = mPLUGOwl3Model.from_pretrained(model_path, dtype=paddle.bfloat16).eval()
tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
processor = model.init_processor(tokenizer)

#image = Image.new('RGB', (500, 500), color='red')
image = Image.open("paddlemix/demo_images/examples_image1.jpg").convert("RGB")

messages = [
    {"role": "user", "content": """<|image|>Describe this image."""},
    {"role": "assistant", "content": ""}
]

inputs = processor(messages, images=[image], videos=None)
inputs['pixel_values'] = inputs['pixel_values'].cast(paddle.bfloat16)
# inputs['input_ids'] [1, 72] # torch [1, 74]
# inputs['input_ids'] = paddle.to_tensor([[151644,   8948,    198, 151645,    198, 151644,    872,    198,     27,
#              91,   2468,  41317,     91,     29,     17,      9,     18,    198,
#              27,     91,   1805,     91,     29,    220,     27,     91,   1805,
#              91,     29,    220,     27,     91,   1805,     91,     29,    198,
#              27,     91,   1805,     91,     29,    220,     27,     91,   1805,
#              91,     29,    220,     27,     91,   1805,     91,     29,    198,
#              27,     91,   1805,     91,     29,     27,     91,    408,  41317,
#              91,     29,  74785,    419,   2168,     13, 151645,    198, 151644,
#           77091,    198]]).astype(paddle.int64)
# inputs['media_offset'] [17, 23, 29, 35, 41, 47, 53]
# inputs['pixel_values'] [7, 3, 378, 378] sum 629145600

import numpy as np
inputs['pixel_values'] = paddle.to_tensor(np.load('pixel_values.npy')).cast(paddle.bfloat16)
inputs['media_offset'] = [paddle.to_tensor([18, 24, 30, 36, 42, 48, 54])]

inputs.update({
    'tokenizer': tokenizer,
    'max_new_tokens':100,
    'decode_text':True,
})

g = model.generate(**inputs)
print(g)
