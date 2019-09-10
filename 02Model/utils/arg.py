import argparse
import sys
from collections import defaultdict
parser = argparse.ArgumentParser(description='parse key pairs into a dictionary')


def isfloat(str):
    try:
        float(str)
    except ValueError:
        return False
    return True


def isint(str): # Assume positive
    return str.isdigit()


class StoreDictKeyPair(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values.split(","):
            temp = my_dict
            k, v = kv.split("=")
            if isint(v):
                v = int(v)
            elif isfloat(v):
                v = float(v)
            keys = k.split(".")
            for i, key in enumerate(keys):
                if i is len(keys) - 1:
                    temp[key] = v
                else:
                    temp[key] ={}
                    temp = temp[key]

        setattr(namespace, self.dest, my_dict)


def update(load, overwrite):
    if not overwrite:
        return
    for key, value in overwrite.items():
        assert key in load, "Invalid key in overwrite parameters"
        if isinstance(value, dict):
            assert isinstance(load[key], dict), "Invalid key in overwrite parameters"
            update(load[key], value)
        else:
            assert not isinstance(load[key], dict), "Invalid key in overwrite parameters"
            load[key] = value

