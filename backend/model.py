from flask import Blueprint, current_app, request, jsonify

from DoTLMViz.utils import get_output_dist, get_token_prob_mappings
from DoTLMViz import TransformerSampler
from . import utils

import torch

bp = Blueprint("model", __name__, url_prefix="/model")


@bp.route("/load", methods=["POST"])
def load_model():
    """
    Loads a model with the name that is in the POST request.
    """
    try:
        utils.if_not_load(request.json["model_name"])

        return jsonify({"message": f"{current_app.model_name} model loaded successfully"}), 200
    except Exception as e:
        print("Error: ", str(e))
        return jsonify({"Error": str(e)}), 500


@bp.route("/tokenize", methods=["POST"])
def tokenize():
    """
    Tokenize the text that is in the POST request.
    """
    try:
        text = request.json["text"]
        utils.tokenize_and_saveable(text)

        # above utils.tokenize_and_save garuntes that the tokens for the text exists
        return current_app.last_input["raw_tokens"]
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
            tokens = current_app.last_input["token_ids"]

            logits, ckpts = current_app.model.run_with_ckpts(tokens.to(current_app.device))
            current_app.logits = logits
            current_app.ckpts = ckpts

            print(f"Successfully ran the model on the input: {text}")
            return jsonify(
                {
                    "message": f"Successful inference over | input: {text} | tokens[{len(current_app.last_input['token_ids'][0])}] : {current_app.last_input['raw_tokens']}"
                }
            ), 201
        if request.method == "GET":
            return jsonify({"message": "Not Implemented"}), 500
    except Exception as e:
        print("Error: ", str(e))
        return jsonify({"Error": str(e)}), 500


@bp.route("/dist", methods=["GET", "POST"])
def get_dist():
    """
    Returns the first 20 token-probability pair from the output probability distribution.
    """
    try:
        if request.method == "POST":
            temperature = request.json["temperature"]

            if temperature == 0.0:
                logits = current_app.logits[:, -1]
            else:
                logits = TransformerSampler.apply_temperature(current_app.logits[:, -1], temperature)
            dist = get_output_dist(logits)
            sorted_dist, sorted_idxs = torch.sort(dist.squeeze().detach().cpu(), descending=True)
            sorted_tokens = current_app.tokenizer.convert_ids_to_tokens(sorted_idxs)
            mappings = get_token_prob_mappings(sorted_tokens[:20], sorted_dist[:20])
            return mappings
    except Exception as e:
        print("Error: ", str(e))
        return jsonify({"Error": str(e)}), 500


@bp.route("/sample", methods=["POST"])
def get_next_token():
    """
    Returns the next token by sampling using the parameters obtained in the POST request
    """
    try:
        if request.method == "POST":
            temperature = request.json["temperature"]
            k = request.json["k"]
            p = request.json["p"]

            next_token_id = TransformerSampler.sample_next_token(
                input_ids=current_app.last_input["token_ids"].squeeze(),
                logits=current_app.logits[:, -1].squeeze(),
                temperature=temperature,
                top_k=k,
                top_p=p,
                frequency_penalty=0.0,
                seed=None,
            )

            return {"next_token": next_token_id}
    except Exception as e:
        print("Error: ", str(e))
        return jsonify({"Error": str(e)}), 500
