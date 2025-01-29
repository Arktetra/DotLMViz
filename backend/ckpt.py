from flask import Blueprint, current_app, request, jsonify

from . import utils

bp = Blueprint("ckpt", __name__, url_prefix="/ckpt")


@bp.route("/act", methods=["POST"])
def get_act():
    """
    Get the checkpointed activation using the activation name, layer name
    and block number in the POST request.
    """
    try:
        if request.method == "POST":
            act_name = request.json["act_name"]
            layer_name = request.json["layer_name"]
            block = request.json["block"]

            act = current_app.ckpts.get(act_name, layer_name, block)

            if act_name == "embed" or act_name == "pos_embed":
                return utils.perform_pca(act)

            return act.squeeze().tolist()
    except Exception as e:
        print("Error: ", str(e))
        return jsonify({"Error": str(e)}), 500


@bp.route("/mlp_outs", methods=["POST"])
def get_mlp_outs():
    """
    Returns a neuron's output for each input using the activation name,
    layer name, block number, and neuron number in the POST request.
    """
    try:
        if request.method == "POST":
            act_name = request.json["act_name"]
            layer_name = request.json["layer_name"]
            block = request.json["block"]
            neuron = request.json["neuron"]

            act = current_app.ckpts.get(act_name, layer_name, block)

            return act.squeeze().T[neuron].tolist()
    except Exception as e:
        print("Error: ", str(e))
        return jsonify({"Error": str(e)}), 500
