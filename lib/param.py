import os

import json

root = os.path.dirname(os.path.dirname(__file__))


def load_params():
    params_file = os.path.join(get_root(), 'param', 'params.json')
    with open(params_file) as json_file:
        jf = json.load(json_file)
    return jf


def get_root():
    return root
