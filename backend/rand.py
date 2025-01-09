from flask import Blueprint

import random

bp = Blueprint("rand", __name__, url_prefix="/api")


@bp.route("/rand", methods=["GET"])
def get_rand():
    return str(random.randint(0, 100))
