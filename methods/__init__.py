from .ins import INS
from .ins_m import INS_M

METHOD_LIST = ["ins","ins_m"]


def get_method(name):
    assert name in METHOD_LIST
    if name == "ins":
        return INS
    if name == "ins_m":
        return INS_M
 