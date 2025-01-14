from flask import Blueprint, current_app, request, Response, jsonify
from transformers import GPT2TokenizerFast

from DoTLMViz import CkptedTransformer

import torch

bp = Blueprint("model", __name__, url_prefix="/model")


@bp.route("/load", methods=["POST"])
def load_model():
    """
    Loads a model with the name that is in the POST request.
    """
    try:
        if request.method == "POST":
            model_name = request.json["model_name"]
            if model_name == "gpt2-small":
                current_app.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
                current_app.model = CkptedTransformer.from_pretrained("gpt2-small")
                print(f"{model_name} model loaded successfully!!!")
            return Response(None, status=201)
    except Exception as e:
        print("Error: ", str(e))
        return jsonify({"Error": str(e)}), 500


@bp.route("/run", methods=["GET", "POST"])
def run_model():
    try:
        if request.method == "POST":
            # print("run model clicked.")
            text = request.json["text"]
            # print(text)
            tokens = current_app.tokenizer(text)["input_ids"]
            tokens = torch.tensor(tokens).reshape(1, -1)
            logits, ckpts = current_app.model.run_with_ckpts(tokens)
            current_app.logits = logits
            current_app.ckpts = ckpts
            print("Successfully ran the model on the input text.")
            return Response(None, status=201)
    except Exception as e:
        print("Error: ", str(e))
        return jsonify({"Error": str(e)}), 500
