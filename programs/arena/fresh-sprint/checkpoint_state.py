"""Primitive-only NumPy RNG state for tensor-safe owned checkpoint reloads."""
import math
import numpy as np


def encode_numpy_rng(state):
    name, keys, position, has_gauss, cached = state
    result = {"generator": name, "keys": [int(x) for x in keys],
              "position": int(position), "has_gauss": int(has_gauss), "cached": float(cached)}
    decode_numpy_rng(result)
    return result


def decode_numpy_rng(value):
    if set(value) != {"generator", "keys", "position", "has_gauss", "cached"}:
        raise ValueError("Unexpected RNG fields")
    if value["generator"] != "MT19937" or len(value["keys"]) != 624:
        raise ValueError("Expected exact MT19937 state")
    if any(type(x) is not int or not 0 <= x < 2**32 for x in value["keys"]):
        raise ValueError("Invalid RNG keys")
    if type(value["position"]) is not int or not 0 <= value["position"] <= 624:
        raise ValueError("Invalid RNG position")
    if type(value["has_gauss"]) is not int or value["has_gauss"] not in (0, 1):
        raise ValueError("Invalid cached-normal flag")
    if type(value["cached"]) not in (int, float) or not math.isfinite(value["cached"]):
        raise ValueError("Invalid cached normal")
    return (value["generator"], np.asarray(value["keys"], dtype=np.uint32),
            value["position"], value["has_gauss"], float(value["cached"]))
