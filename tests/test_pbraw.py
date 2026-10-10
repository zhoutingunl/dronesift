"""Schema-less protobuf decoder: varint / wiretype basics."""
import struct
from dronesift.pbraw import read_varint, decode

def test_read_varint_boundaries():
    assert read_varint(b'\x00', 0) == (0, 1)
    assert read_varint(b'\x7f', 0) == (127, 1)
    assert read_varint(b'\x80\x01', 0) == (128, 2)
    assert read_varint(b'\xff\xff\xff\xff\x07', 0) == (2**31 - 1, 5)

def test_decode_nested_message():
    tree = decode(b'\x1a\x04\x08\x96\x01')          # field 3: msg { 1: varint 150 }
    assert tree[0][0] == 3 and tree[0][1] == 'len'
    assert tree[0][3][0] == (1, 'v', 150)

def test_decode_scalar_wiretypes():
    tree = decode(b'\x08\x96\x01'                                   # 1: varint 150
                  b'\x15\x00\x00\x80\x3f'                           # 2: f32 1.0
                  b'\x11\x00\x00\x00\x00\x00\x00\xf0\x3f')          # 3: f64 1.0
    assert tree[0] == (1, 'v', 150)
    assert struct.unpack('<f', tree[1][2])[0] == 1.0
    assert struct.unpack('<d', tree[2][2])[0] == 1.0

def test_decode_bad_wiretype_returns_none():
    assert decode(b'\x0f') is None                  # field 1, wiretype 7
    assert decode(b'\x00\xff') is None              # truncated varint
