from flask import Blueprint, current_app, request, Response, jsonify
from transformers import GPT2TokenizerFast

from DoTLMViz import CkptedTransformer
from DoTLMViz.utils import get_output_dist

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
                current_app.device = (
                    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
                )
                print(f"{model_name} model loaded successfully!!!")
            return Response(None, status=201)
    except Exception as e:
        print("Error: ", str(e))
        return jsonify({"Error": str(e)}), 500


@bp.route("/run", methods=["GET", "POST"])
def run_model():
    """
    Runs a model on the input text that is in the POST request.
    """
    try:
        if request.method == "POST":
            text = request.json["text"]
            text = " " if text == "" else text
            tokens = current_app.tokenizer(text, return_tensors="pt")["input_ids"]
            logits, ckpts = current_app.model.run_with_ckpts(tokens.to(current_app.device))
            current_app.logits = logits
            current_app.ckpts = ckpts
            print("Successfully ran the model on the input text.")
            return Response(None, status=201)
    except Exception as e:
        print("Error: ", str(e))
        return jsonify({"Error": str(e)}), 500


@bp.route("/dist", methods=["GET"])
def get_dist():
    """
    Returns the first 20 token-probability pair from the output probability distribution.
    """
    try:
        dist = get_output_dist(current_app.logits[:, -1])
        sorted_dist, sorted_idxs = torch.sort(dist.squeeze().detach().cpu(), descending=True)
        sorted_tokens = current_app.tokenizer.convert_ids_to_tokens(sorted_idxs)
        return [sorted_dist[:20].tolist(), sorted_tokens[:20]]
    except Exception as e:
        print("Error: ", str(e))
        return jsonify({"Error": str(e)}), 500
