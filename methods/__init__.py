from .intl import INTL
from .intl_m import INTL_M

METHOD_LIST = ["intl","intl_m"]

def get_method(name):
    assert name in METHOD_LIST
    if name == "ins":
        return INTL
    if name == "ins_m":
        return INTL_M
 