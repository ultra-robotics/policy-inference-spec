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


def _resize_hwc_uint8(img: npt.NDArray[np.uint8], height: int, width: int) -> npt.NDArray[np.uint8]:
    h, w = img.shape[:2]
    if (h, w) == (height, width):
        return img
    resize = getattr(cv2, "resize", None)
    inter_area = getattr(cv2, "INTER_AREA", 1)
    if resize is not None:
        return np.ascontiguousarray(resize(img, (width, height), interpolation=inter_area)).astype(np.uint8, copy=False)
    y_idx = np.linspace(0, h - 1, height, dtype=np.intp)
    x_idx = np.linspace(0, w - 1, width, dtype=np.intp)
    return np.ascontiguousarray(img[y_idx[:, None], x_idx[None, :]]).astype(np.uint8, copy=False)


def _as_hwc_uint8(arr: npt.NDArray[Any]) -> npt.NDArray[np.uint8]:
    if arr.ndim == 4:
        assert arr.shape[0] == 1, f"JPEG transport only supports batch size 1, got shape {arr.shape}"
        arr = arr[0]
    assert arr.ndim == 3, f"JPEG transport expects HWC or BHWC arrays, got shape {arr.shape}"
    assert arr.shape[-1] == 3, f"JPEG transport expects channel-last RGB arrays, got shape {arr.shape}"
    img = np.ascontiguousarray(arr)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _hwc_to_array(img: npt.NDArray[np.uint8], shape: tuple[int, ...]) -> npt.NDArray[Any]:
    target_shape = shape[1:] if len(shape) == 4 else shape
    assert len(target_shape) == 3, f"JPEG transport expects HWC target shape, got {shape}"
    target_h, target_w, target_c = target_shape
    assert target_c == 3, f"JPEG transport expects RGB target shape, got {shape}"
    if img.shape[:2] != (target_h, target_w):
        img = _resize_hwc_uint8(img, target_h, target_w)
    assert img.shape[-1] == target_c, f"JPEG transport expects {target_c} channels, got {img.shape}"
    arr = np.ascontiguousarray(img)
    if len(shape) == 4:
        return arr[None]
    return arr


def encode_ndarray(arr: npt.NDArray[Any], jpeg_quality: int | None = None, image_scale: float = 1.0) -> NdarrayField:
    if jpeg_quality is None:
        return NdarrayField(
            data=arr.tobytes(),
            shape=arr.shape,
            dtype=str(arr.dtype),
        )

    img = _as_hwc_uint8(arr)
    if image_scale != 1.0:
        assert image_scale > 0, f"image_scale must be positive, got {image_scale}"
        scaled_w = max(1, int(round(img.shape[1] * image_scale)))
        scaled_h = max(1, int(round(img.shape[0] * image_scale)))
        img = _resize_hwc_uint8(img, scaled_h, scaled_w)

    return NdarrayField(
        data=simplejpeg.encode_jpeg(img, quality=jpeg_quality),
        shape=arr.shape,
        dtype=str(arr.dtype),
        codec="jpeg",
    )


def decode_ndarray(field: NdarrayField) -> npt.NDArray[Any]:
    if field.codec == "jpeg":
        img = simplejpeg.decode_jpeg(field.data)
        arr = _hwc_to_array(img, field.shape)
        return arr.astype(np.dtype(field.dtype), copy=False)
    assert field.codec == "raw", f"Unsupported ndarray codec: {field.codec}"
    return np.frombuffer(field.data, dtype=np.dtype(field.dtype)).reshape(field.shape)


def hwc_from_wire_image(val: Any, target_shape: tuple[int, ...]) -> npt.NDArray[Any]:
    if isinstance(val, bytes):
        img = simplejpeg.decode_jpeg(val)
        return _hwc_to_array(img, target_shape).astype(np.float32, copy=False)
    assert isinstance(val, np.ndarray), f"image value must be bytes or ndarray, got {type(val)}"
    if val.dtype == np.uint8 and val.ndim in (3, 4):
        return _hwc_to_array(_as_hwc_uint8(val), target_shape).astype(np.float32, copy=False)
    raise AssertionError(f"unexpected image payload dtype/shape {val.dtype} {val.shape}")


def ndarray_to_msgpack_tag(arr: npt.NDArray[Any]) -> dict[str, Any]:
    return {
        NDARRAY_MSGPACK_TAG: True,
        "data": arr.tobytes(),
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
    }


def ndarray_from_msgpack_tag(obj: Any) -> npt.NDArray[Any]:
    assert isinstance(obj, dict), f"expected ndarray tag dict, got {type(obj)}"
    tag = obj.get(NDARRAY_MSGPACK_TAG)
    if tag is None:
        tag = obj.get(NDARRAY_MSGPACK_TAG.encode())
    assert tag is True, f"invalid ndarray tag keys={sorted(repr(key) for key in obj)}"
    data = obj["data"] if "data" in obj else obj[b"data"]
    dtype = np.dtype(obj["dtype"] if "dtype" in obj else obj[b"dtype"])
    shape = tuple(obj["shape"] if "shape" in obj else obj[b"shape"])
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
        if NDARRAY_MSGPACK_TAG in obj or NDARRAY_MSGPACK_TAG.encode() in obj:
            return ndarray_from_msgpack_tag(obj)
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
        return [_walk_decode(x) for x in obj]
    return obj


def msgpack_decode(data: bytes) -> Any:
    raw = msgspec.msgpack.decode(data)
    return _walk_decode(raw)


__all__ = [
    "InferenceMetadataValue",
    "NDARRAY_MSGPACK_TAG",
    "NdarrayField",
    "decode_ndarray",
    "encode_ndarray",
    "hwc_from_wire_image",
    "msgpack_decode",
    "msgpack_encode",
    "ndarray_from_msgpack_tag",
    "ndarray_to_msgpack_tag",
]
