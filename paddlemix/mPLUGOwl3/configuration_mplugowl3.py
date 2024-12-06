import os
import paddlenlp
""" mPLUGOwl3 model configuration"""
# from paddlenlp.transformers import PretrainedConfig, Qwen2Config
from typing import Union
from .configuration_hyper_qwen2 import HyperQwen2Config
# logger = paddle.utils.try_import('logging').getLogger(name=__name__)
from paddlemix.utils.log import logger
from .modeling_navit_siglip import SigLipVisionConfig


class mPLUGOwl3Config(HyperQwen2Config):
    model_type = 'mplugowl3'
    keys_to_ignore_at_inference = ['past_key_values']
    default_vision_config = {'hidden_size': 1152, 'image_size': 384,
        'intermediate_size': 4304, 'model_type': 'siglip_vision_model',
        'num_attention_heads': 16, 'num_hidden_layers': 27, 'patch_size': 14}

    def __init__(self, use_cache=True, vision_config=None, **kwargs):
        self.use_cache = use_cache
        if vision_config is None:
# >>>>>>            self.vision_config = (transformers.models.siglip.
#                 configuration_siglip.SiglipVisionConfig(**self.
#                 default_vision_config))
            self.vision_config = SigLipVisionConfig(**self.default_vision_config)
            logger.info('vision_config is None, using default vision config')
        elif isinstance(vision_config, dict):
# >>>>>>            self.vision_config = (transformers.models.siglip.
                # configuration_siglip.SiglipVisionConfig(**vision_config))
            self.vision_config = SigLipVisionConfig(**vision_config)
# >>>>>>        elif isinstance(vision_config, transformers.models.siglip.
#             configuration_siglip.SiglipVisionConfig):
        elif isinstance(vision_config, SigLipVisionConfig):
            self.vision_config = vision_config
        self.image_size = self.vision_config.image_size
        self.patch_size = self.vision_config.patch_size
        super().__init__(**kwargs)
