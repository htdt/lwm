import re
import yaml


def replace_e_float(d):
    p = re.compile(r"^-?\d+(\.\d+)?e-?\d+$")
    for name, val in d.items():
        if type(val) == dict:
            replace_e_float(val)
        elif type(val) == str and p.match(val):
            d[name] = float(val)


def load_cfg(name, prefix="."):
    with open(f"{prefix}/{name}.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
        replace_e_float(cfg)
        return cfg
