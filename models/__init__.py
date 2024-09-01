from .distilbert_model import DistilBERTModel
from .mpt_model import MPTModel

def get_model(model_name):
    if model_name == 'distilbert':
        return DistilBERTModel()
    elif model_name == 'mpt':
        return MPTModel()
    else:
        raise ValueError(f"Unknown model: {model_name}")