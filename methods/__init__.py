from .mid import MID
from .midm import MIDM

METHOD_LIST = ["mid","midm"]


def get_method(name):
    assert name in METHOD_LIST
    if name == "mid":
        return MID
    if name == "midm":
        return MIDM
 