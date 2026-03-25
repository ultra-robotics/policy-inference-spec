from __future__ import annotations

from typing import Any, TypeAlias

import cv2
import msgspec
import numpy as np
import numpy.typing as npt
import simplejpeg

InferenceMetadataValue: TypeAlias = str | int | float | bool | None

NDARRAY_MSGPACK_TAG = "__ndarray__"


class NdarrayField(msgspec.Struct):
    data: bytes
    shape: tuple[int, ...]
    dtype: str = "float32"
    codec: str = "raw"


def _chw_to_hwc(arr: npt.NDArray[Any]) -> npt.NDArray[np.uint8]:
    if arr.ndim == 4:
        assert arr.shape[0] == 1, f"JPEG transport only supports batch size 1, got shape {arr.shape}"
        arr = arr[0]
    assert arr.ndim == 3, f"JPEG transport expects CHW or BCHW arrays, got shape {arr.shape}"
    img = np.ascontiguousarray(arr.transpose(1, 2, 0))
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _hwc_to_chw(img: npt.NDArray[np.uint8], shape: tuple[int, ...]) -> npt.NDArray[Any]:
    target_shape = shape[1:] if len(shape) == 4 else shape
    assert len(target_shape) == 3, f"JPEG transport expects CHW target shape, got {shape}"
    target_h, target_w = target_shape[1], target_shape[2]
    if img.shape[:2] != (target_h, target_w):
        img = np.ascontiguousarray(cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)).astype(
            np.uint8, copy=False
        )
    chw = np.ascontiguousarray(img.transpose(2, 0, 1))
    if len(shape) == 4:
        return chw[None]
    return chw


def encode_ndarray(arr: npt.NDArray[Any], jpeg_quality: int | None = None, image_scale: float = 1.0) -> NdarrayField:
    if jpeg_quality is None:
        return NdarrayField(
            data=arr.tobytes(),
            shape=arr.shape,
            dtype=str(arr.dtype),
        )

    img = _chw_to_hwc(arr)
    if image_scale != 1.0:
        assert image_scale > 0, f"image_scale must be positive, got {image_scale}"
        scaled_w = max(1, int(round(img.shape[1] * image_scale)))
        scaled_h = max(1, int(round(img.shape[0] * image_scale)))
        img = np.ascontiguousarray(cv2.resize(img, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)).astype(
            np.uint8, copy=False
        )

    return NdarrayField(
        data=simplejpeg.encode_jpeg(img, quality=jpeg_quality),
        shape=arr.shape,
        dtype=str(arr.dtype),
        codec="jpeg",
    )


def decode_ndarray(field: NdarrayField) -> npt.NDArray[Any]:
    if field.codec == "jpeg":
        img = simplejpeg.decode_jpeg(field.data)
        arr = _hwc_to_chw(img, field.shape)
        return arr.astype(np.dtype(field.dtype), copy=False)
    assert field.codec == "raw", f"Unsupported ndarray codec: {field.codec}"
    return np.frombuffer(field.data, dtype=np.dtype(field.dtype)).reshape(field.shape)


def chw_from_wire_image(val: Any, target_shape: tuple[int, ...]) -> npt.NDArray[Any]:
    if isinstance(val, bytes):
        img = simplejpeg.decode_jpeg(val)
        return _hwc_to_chw(img, target_shape).astype(np.float32, copy=False)
    assert isinstance(val, np.ndarray), f"image value must be bytes or ndarray, got {type(val)}"
    if val.dtype == np.uint8 and val.ndim == 3:
        return _hwc_to_chw(val, target_shape).astype(np.float32, copy=False)
    raise AssertionError(f"unexpected image payload dtype/shape {val.dtype} {val.shape}")


def ndarray_to_msgpack_tag(arr: npt.NDArray[Any]) -> dict[str, Any]:
    return {
        NDARRAY_MSGPACK_TAG: {
            "data": arr.tobytes(),
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
        }
    }


def ndarray_from_msgpack_tag(obj: Any) -> npt.NDArray[Any]:
    assert isinstance(obj, dict), f"expected ndarray tag dict, got {type(obj)}"
    inner = obj.get(NDARRAY_MSGPACK_TAG)
    assert isinstance(inner, dict), f"invalid ndarray tag: {obj!r}"
    data = inner["data"]
    dtype = np.dtype(inner["dtype"])
    shape = tuple(inner["shape"])
    return np.frombuffer(data, dtype=dtype).reshape(shape)


def _msgpack_ndarray_encode_hook(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return ndarray_to_msgpack_tag(obj)
    raise TypeError(f"Cannot encode type: {type(obj)!r}")


def msgpack_encode(obj: Any) -> bytes:
    enc = msgspec.msgpack.Encoder(enc_hook=_msgpack_ndarray_encode_hook)
    return enc.encode(obj)


def _walk_decode(obj: Any) -> Any:
    if isinstance(obj, dict):
        if NDARRAY_MSGPACK_TAG in obj and isinstance(obj[NDARRAY_MSGPACK_TAG], dict):
            return ndarray_from_msgpack_tag(obj)
        return {k: _walk_decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_decode(x) for x in obj]
    return obj


def msgpack_decode(data: bytes) -> Any:
    raw = msgspec.msgpack.decode(data)
    return _walk_decode(raw)


__all__ = [
    "InferenceMetadataValue",
    "NDARRAY_MSGPACK_TAG",
    "NdarrayField",
    "chw_from_wire_image",
    "decode_ndarray",
    "encode_ndarray",
    "msgpack_decode",
    "msgpack_encode",
    "ndarray_from_msgpack_tag",
    "ndarray_to_msgpack_tag",
]
