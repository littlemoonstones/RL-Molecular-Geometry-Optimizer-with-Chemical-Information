import torch.nn as nn
from typing import Dict
from lib.ModelType import ModelMetaData
from models.model_template import ModelActor, ModelCritic, ModelSACTwinQ
from models.SelfAttention.model0 import CustomModel as selfAttnModel0


MODEL_DICT: Dict[str, nn.Module] = {
    "0": selfAttnModel0,
    
}


class ModelFactory:
    def __init__(self, key:str, meta_data: ModelMetaData) -> None:
        self.key = key
        self.meta_data = meta_data
        assert key in MODEL_DICT
        self.customModel = MODEL_DICT.get(self.key)
    
    def getActor(self):
        print(self.meta_data)
        return ModelActor(self.customModel, self.meta_data)
    def getCritic(self):
        return ModelCritic(self.customModel, self.meta_data)
    def getTwinQ(self):
        return ModelSACTwinQ(self.customModel, self.meta_data)