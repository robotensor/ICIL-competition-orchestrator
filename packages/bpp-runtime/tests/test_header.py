"""The hand-written safetensors header parser: what it accepts and everything it refuses."""

from __future__ import annotations

import io
import json
import struct

import pytest

from bpp_runtime.header import HeaderError, read_header
from st_testing import TENSORS, safetensors_bytes


def parse(data: bytes, **kwargs):
    return read_header(io.BytesIO(data), len(data), **kwargs)


def test_a_well_formed_file_parses():
    data = safetensors_bytes(metadata={"format": "pt"})
    header = parse(data)
    assert set(header.tensors) == set(TENSORS)
    assert header.tensors["model.weight"].shape == (4, 3)
    assert header.tensors["model.weight"].dtype == "F32"
    assert header.tensors["_dummy_variable"].numel == 0
    assert header.metadata == {"format": "pt"}
    assert header.param_count == 12 + 4 + 20
    assert header.total_bytes == len(data)


def test_zero_size_tensors_anywhere_in_the_layout():
    tensors = {"a": ("F32", [0]), "b": ("F32", [3]), "c": ("F32", [0]), "d": ("BF16", [2, 0])}
    header = parse(safetensors_bytes(tensors))
    assert [header.tensors[k].numel for k in "abcd"] == [0, 3, 0, 0]


def test_space_padding_after_the_json_is_accepted():
    # safetensors pads the header with spaces to a multiple of 8; this one needs two
    data = safetensors_bytes({"x": ("F32", [1])})
    (length,) = struct.unpack("<Q", data[:8])
    assert data[8 + length - 2 : 8 + length] == b"  "
    parse(data)


@pytest.mark.parametrize(
    ("header", "message"),
    [
        (
            b'{"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}, "a": '
            b'{"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}',
            "twice",
        ),
        (b'{"a": {"dtype": "F32", "shape": [NaN], "data_offsets": [0, 4]}}', "NaN"),
        (b'{"a": {"dtype": "F33", "shape": [1], "data_offsets": [0, 4]}}', "unknown dtype"),
        (b'{"a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}}', "hold 4 bytes"),
        (b'{"a": {"dtype": "F32", "shape": [-1], "data_offsets": [0, 4]}}', "non-negative"),
        (b'{"a": {"dtype": "F32", "shape": [true], "data_offsets": [0, 4]}}', "non-negative"),
        (b'{"a": {"dtype": "F32", "shape": [1.0], "data_offsets": [0, 4]}}', "non-negative"),
        (b'{"a": {"dtype": "F32", "shape": [1], "data_offsets": [4, 0]}}', "hold"),
        (b'{"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4], "x": 1}}', "exactly"),
        (b'{"a": {"dtype": "F32", "shape": [1]}}', "exactly"),
        (
            b'{"__metadata__": {"format": 1}, "a": {"dtype": "F32", "shape": [1], '
            b'"data_offsets": [0, 4]}}',
            "strings",
        ),
        (b'{"": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}', "empty name"),
        (b'["a"]', "JSON object"),
        (b'{"a": ', "not valid JSON"),
        (b"\xff\xfe{}", "UTF-8"),
    ],
)
def test_malformed_headers_are_refused(header, message):
    with pytest.raises(HeaderError, match=message):
        parse(safetensors_bytes({}, header=header, extra=bytes(4)))


def test_a_gap_between_tensors_is_refused():
    document = {
        "a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        "b": {"dtype": "F32", "shape": [1], "data_offsets": [8, 12]},
    }
    with pytest.raises(HeaderError, match="gap"):
        parse(safetensors_bytes({}, header=json.dumps(document).encode(), extra=bytes(12)))


def test_overlapping_tensors_are_refused():
    document = {
        "a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
        "b": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]},
    }
    with pytest.raises(HeaderError, match="overlaps"):
        parse(safetensors_bytes({}, header=json.dumps(document).encode(), extra=bytes(8)))


def test_bytes_after_the_last_tensor_are_refused():
    with pytest.raises(HeaderError, match="after the header"):
        parse(safetensors_bytes(extra=b"hidden"))


def test_a_truncated_data_section_is_refused():
    data = safetensors_bytes()
    with pytest.raises(HeaderError, match="after the header"):
        parse(data[:-4])


def test_a_huge_header_length_is_refused_before_reading_it():
    data = safetensors_bytes(length=1 << 40)
    with pytest.raises(HeaderError, match="at most"):
        parse(data)


def test_a_header_longer_than_the_file_is_refused():
    data = safetensors_bytes()
    with pytest.raises(HeaderError, match="claims"):
        read_header(io.BytesIO(data), 24)


def test_a_file_shorter_than_the_length_field_is_refused():
    with pytest.raises(HeaderError, match="too short"):
        parse(b"\x01\x02")


def test_a_header_that_is_not_an_object_start_is_refused():
    with pytest.raises(HeaderError, match="JSON object"):
        parse(safetensors_bytes({}, header=b" {}"))
