import paddlenlp
import paddle
from paddlenlp.transformers import PretrainedModel, AutoTokenizer
from PIL import Image
from decord import VideoReader, cpu
from mPLUGOwl3.modeling_mplugowl3 import mPLUGOwl3Model
from mPLUGOwl3.configuration_mplugowl3 import mPLUGOwl3Config

model_path = '/home/aistudio/paddle_test/mPLUGOwl3'
config = mPLUGOwl3Config.from_pretrained(model_path)
# print(config)
model = mPLUGOwl3Model.from_pretrained(model_path, config=config, dtype="float16")
model=model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_path)
processor = model.init_processor(tokenizer)
image = Image.new('RGB', (500, 500), color='red')
messages = [{'role': 'user', 'content':
    """<|image|>
Describe this image."""}, {'role': 'assistant', 'content': ''}
    ]
inputs = processor(messages, images=[image], videos=None)
# inputs.to('cuda')
inputs.update({'tokenizer': tokenizer, 'max_new_tokens': 100, 'decode_text':
    True})
g = model.generate(**inputs)
print(g)
