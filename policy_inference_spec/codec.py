from __future__ import annotations

from typing import Any

import cv2
import msgspec
import numpy as np
import numpy.typing as npt
import simplejpeg  # type: ignore[import-untyped]

from policy_inference_spec.protocol import ImageArray, ProtocolPayload

NDARRAY_MSGPACK_TAG = "__ndarray__"
_SUPPORTED_NDARRAY_DTYPES = frozenset({np.dtype(np.uint8), np.dtype(np.float32)})


class NdarrayField(msgspec.Struct):
    data: bytes
    shape: tuple[int, ...]
    dtype: str = "float32"
    codec: str = "raw"


def _resize_hwc_uint8(image: ImageArray, height: int, width: int) -> ImageArray:
    h, w = image.shape[:2]
    if (h, w) == (height, width):
        return image
    resize = getattr(cv2, "resize", None)
    inter_area = getattr(cv2, "INTER_AREA", 1)
    if resize is not None:
        return np.ascontiguousarray(resize(image, (width, height), interpolation=inter_area)).astype(
            np.uint8, copy=False
        )
    y_idx = np.linspace(0, h - 1, height, dtype=np.intp)
    x_idx = np.linspace(0, w - 1, width, dtype=np.intp)
    return np.ascontiguousarray(image[y_idx[:, None], x_idx[None, :]]).astype(np.uint8, copy=False)


def _as_hwc_uint8(image: npt.NDArray[np.uint8]) -> ImageArray:
    if image.ndim == 4:
        assert image.shape[0] == 1, f"JPEG transport only supports batch size 1, got shape {image.shape}"
        image = image[0]
    assert image.ndim == 3, f"JPEG transport expects HWC or BHWC arrays, got shape {image.shape}"
    assert image.shape[-1] == 3, f"JPEG transport expects channel-last RGB arrays, got shape {image.shape}"
    assert image.dtype == np.uint8, f"JPEG transport expects uint8 arrays, got {image.dtype}"
    return np.ascontiguousarray(image)


def _ndarray_to_msgpack_tag(array: npt.NDArray[Any]) -> dict[str, Any]:
    dtype = np.dtype(array.dtype)
    assert dtype in _SUPPORTED_NDARRAY_DTYPES, f"Unsupported ndarray dtype for msgpack transport: {dtype}"
    return {
        NDARRAY_MSGPACK_TAG: True,
        "data": array.tobytes(),
        "dtype": str(dtype),
        "shape": list(array.shape),
    }


def _ndarray_from_msgpack_tag(obj: Any) -> npt.NDArray[Any]:
    assert isinstance(obj, dict), f"expected ndarray tag dict, got {type(obj)}"
    tag = obj.get(NDARRAY_MSGPACK_TAG)
    if tag is None:
        tag = obj.get(NDARRAY_MSGPACK_TAG.encode())
    assert tag is True, f"invalid ndarray tag keys={sorted(repr(key) for key in obj)}"
    data = obj["data"] if "data" in obj else obj[b"data"]
    dtype = np.dtype(obj["dtype"] if "dtype" in obj else obj[b"dtype"])
    assert dtype in _SUPPORTED_NDARRAY_DTYPES, f"Unsupported ndarray dtype for msgpack transport: {dtype}"
    shape = tuple(obj["shape"] if "shape" in obj else obj[b"shape"])
    return np.frombuffer(data, dtype=dtype).reshape(shape)


def _msgpack_ndarray_encode_hook(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return _ndarray_to_msgpack_tag(obj)
    raise TypeError(f"Cannot encode type: {type(obj)!r}")


_MSGPACK_ENCODER = msgspec.msgpack.Encoder(enc_hook=_msgpack_ndarray_encode_hook)


def _walk_decode(obj: Any) -> Any:
    if isinstance(obj, dict):
        if NDARRAY_MSGPACK_TAG in obj or NDARRAY_MSGPACK_TAG.encode() in obj:
            return _ndarray_from_msgpack_tag(obj)
        normalized: dict[Any, Any] = {}
        for key, value in obj.items():
            if isinstance(key, bytes):
                try:
                    key = key.decode()
                except UnicodeDecodeError:
                    pass
            normalized[key] = _walk_decode(value)
        return normalized
    if isinstance(obj, list):
        return [_walk_decode(item) for item in obj]
    return obj


def encode_image(
    image: npt.NDArray[np.uint8],
    height: int,
    width: int,
    jpeg_quality: int = 75,
) -> NdarrayField:
    assert height > 0, f"height must be positive, got {height}"
    assert width > 0, f"width must be positive, got {width}"
    assert 0 < jpeg_quality <= 100, f"jpeg_quality must be in [1, 100], got {jpeg_quality}"
    normalized = _as_hwc_uint8(image)
    resized = _resize_hwc_uint8(normalized, height, width)
    return NdarrayField(
        data=simplejpeg.encode_jpeg(resized, quality=jpeg_quality),
        shape=image.shape,
        dtype=str(image.dtype),
        codec="jpeg",
    )


def serialize_to_msgpack(data: ProtocolPayload) -> bytes:
    return _MSGPACK_ENCODER.encode(data)


def deserialize_from_msgpack(data: bytes) -> ProtocolPayload:
    decoded = _walk_decode(msgspec.msgpack.decode(data))
    assert isinstance(decoded, dict), f"expected msgpack root dict, got {type(decoded)}"
    return decoded


__all__ = [
    "NDARRAY_MSGPACK_TAG",
    "NdarrayField",
    "deserialize_from_msgpack",
    "encode_image",
    "serialize_to_msgpack",
]
