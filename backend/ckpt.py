from flask import Blueprint, current_app, request, jsonify

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

            print(act_name, layer_name, block)
            print("inside here")

            act = current_app.ckpts.get(act_name, layer_name, block)
            return act.tolist()
    except Exception as e:
        print("Error: ", str(e))
        return jsonify({"Error": str(e)}), 500
