from flask import current_app
from transformers import GPT2Tokenizer
from DoTLMViz import CkptedTransformer
from DoTLMViz.decomposition import PCA

import torch


def load_model():
    """
    An utility function for loading the models. The model to be loaded by the model_name
    in the current_app instance.
    """
    if current_app.model_name == "gpt2-small":
        current_app.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        current_app.model = CkptedTransformer.from_pretrained("gpt2-small")
        current_app.device = (
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        print(f"{current_app.model_name} model loaded successfully!!!")

    current_app.is_model_loaded = True


def perform_pca(data: torch.Tensor):
    """
    A utility function for performing PCA on the data.
    """
    pca = PCA(n_components=2)
    pca.fit(data.squeeze())
    return pca(data).tolist()
