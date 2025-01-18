from flask import Blueprint, request, jsonify

from DoTLMViz.decomposition import PCA

import torch

bp = Blueprint("dimred", __name__, url_prefix="/dimred")


@bp.route("/pca", methods=["POST"])
def perform_pca():
    """
    Perform PCA on the data-points in the POST request.
    """
    try:
        if request.method == "POST":
            data = request.json["data"]
            X = torch.tensor(data)
            pca = PCA(n_components=2)
            pca.fit(X)
            return pca(X).tolist()
    except Exception as e:
        print("Error: ", str(e))
        return jsonify({"Error": str(e)}), 500
