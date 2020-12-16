# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pyarrow
import base64
from lz4 import frame


def pack(data):
    data = pyarrow.serialize(data).to_buffer()#.to_pybytes()
    # data = frame.compress(data, block_size=frame.BLOCKSIZE_MAX256KB)
    # data = base64.b64encode(data).decode("ascii")
    return data


def unpack(data):
    # data = base64.b64decode(data)
    # data = frame.decompress(data)
    data = pyarrow.deserialize(data)
    return data
