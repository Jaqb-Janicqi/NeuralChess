import base64
import numpy as np
import torch


def encode_a85(byte_rep):
    return base64.a85encode(byte_rep).decode()


def decode_a85(byte_rep, dtype=np.uint32):
    return np.frombuffer(base64.a85decode(byte_rep), dtype=dtype)


def decode_position(byte_rep):
    return decode_a85(byte_rep, np.int8).astype(np.float32).reshape(9, 8, 8)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def decode_policy(byte_rep):
    arr = decode_a85(byte_rep, np.uint32).astype(np.float32)
    arr = arr / arr.sum()
    return softmax(arr)
