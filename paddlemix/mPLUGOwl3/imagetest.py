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

import paddle
import paddlenlp
from decord import VideoReader, cpu
from mPLUGOwl3.configuration_mplugowl3 import mPLUGOwl3Config
from mPLUGOwl3.modeling_mplugowl3 import mPLUGOwl3Model
from paddlenlp.transformers import AutoTokenizer, PretrainedModel
from PIL import Image

model_path = "/home/aistudio/paddle_test/mPLUGOwl3"
config = mPLUGOwl3Config.from_pretrained(model_path)
# print(config)
model = mPLUGOwl3Model.from_pretrained(model_path, config=config, dtype="float16")
model = model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_path)
processor = model.init_processor(tokenizer)
image = Image.new("RGB", (500, 500), color="red")
messages = [
    {
        "role": "user",
        "content": """<|image|>
Describe this image.""",
    },
    {"role": "assistant", "content": ""},
]
inputs = processor(messages, images=[image], videos=None)
# inputs.to('cuda')
inputs.update({"tokenizer": tokenizer, "max_new_tokens": 100, "decode_text": True})
g = model.generate(**inputs)
print(g)
