"""Independent byte-level MDL codec for targeted mapping checkpoints."""

from __future__ import annotations

import json
import struct
import zlib

import torch


def _tensor_record(name: str, tensor: torch.Tensor) -> tuple[dict, bytes]:
    """Quantize one tensor and return its fixed header plus payload."""

    value = tensor.detach().float().cpu()
    max_abs = float(value.abs().max().item())
    scale = max(max_abs / 32767.0, 1e-12)
    quantized = torch.round(value / scale).clamp(-32767, 32767).to(torch.int16).numpy()
    header = {"name": name, "shape": list(value.shape), "scale": scale, "dtype": "int16"}
    return header, quantized.tobytes(order="C")


def mapping_description_length_bits(mapping_path: str) -> dict:
    """Return the compressed bit length under the fixed MDL-UAP-v1 syntax."""

    checkpoint = torch.load(mapping_path, map_location="cpu", weights_only=False)
    records = []
    payload = bytearray()
    for name in sorted(checkpoint["mapping"]):
        header, tensor_bytes = _tensor_record(name, checkpoint["mapping"][name])
        records.append(header)
        payload.extend(struct.pack("<I", len(tensor_bytes)))
        payload.extend(tensor_bytes)

    header_bytes = json.dumps(
        {
            "protocol": "MDL-UAP-v1",
            "mode": checkpoint["mode"],
            "target": int(checkpoint["target"]),
            "epsilon": float(checkpoint["epsilon"]),
            "records": records,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    compressed = zlib.compress(bytes(payload), level=9)
    total_bytes = len(header_bytes) + len(compressed)
    return {
        "bits": int(total_bytes * 8),
        "bytes": total_bytes,
        "header_bytes": len(header_bytes),
        "payload_bytes": len(compressed),
        "tensor_count": len(records),
        "compression": "zlib-level-9",
        "protocol": "MDL-UAP-v1",
    }
