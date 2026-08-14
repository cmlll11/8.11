"""Independent byte-level MDL codec for targeted mapping checkpoints."""

from __future__ import annotations

import json
import struct
import zlib

import numpy as np
import torch


SUPPORTED_QUANTIZATION_BITS = (16, 8, 4)


def _tensor_record(
    name: str, tensor: torch.Tensor, quantization_bits: int
) -> tuple[dict, bytes, torch.Tensor]:
    """Quantize one tensor and return its header, payload, and decoded tensor."""

    if quantization_bits not in SUPPORTED_QUANTIZATION_BITS:
        raise ValueError(
            f"unsupported quantization_bits={quantization_bits}; "
            f"choose one of {SUPPORTED_QUANTIZATION_BITS}"
        )
    value = tensor.detach().float().cpu()
    max_abs = float(value.abs().max().item())
    max_code = (1 << (int(quantization_bits) - 1)) - 1
    scale = max(max_abs / max_code, 1e-12)
    quantized = (
        torch.round(value / scale)
        .clamp(-max_code, max_code)
        .to(torch.int64)
        .numpy()
        .reshape(-1)
    )
    decoded = torch.from_numpy((quantized.astype(np.float32) * scale).reshape(value.shape))

    if quantization_bits == 16:
        payload = quantized.astype(np.int16).tobytes(order="C")
        dtype = "int16"
    elif quantization_bits == 8:
        payload = quantized.astype(np.int8).tobytes(order="C")
        dtype = "int8"
    elif quantization_bits == 4:
        # Pack two signed 4-bit values into one byte, including the odd tail.
        values = (quantized + 8).astype(np.uint8)
        if values.size % 2:
            values = np.concatenate([values, np.array([8], dtype=np.uint8)])
        payload = (values[0::2] | (values[1::2] << 4)).tobytes(order="C")
        dtype = "int4-packed"
    else:  # pragma: no cover - guarded by the validation above.
        raise AssertionError("unreachable quantization branch")

    header = {
        "name": name,
        "shape": list(value.shape),
        "scale": scale,
        "dtype": dtype,
        "quantization_bits": int(quantization_bits),
    }
    return header, payload, decoded


def _mapping_tensors(checkpoint: dict) -> dict[str, torch.Tensor]:
    """Return the tensors that define a saved mapping."""

    if checkpoint["mode"] == "universal":
        return {"universal_delta": checkpoint["universal_delta"]}
    return checkpoint["mapping"]


def _encode_tensors(
    tensors: dict[str, torch.Tensor],
    *,
    mode: str,
    target: int,
    epsilon: float,
    attack_goal: str | None,
    quantization_bits: int,
) -> tuple[dict, dict[str, torch.Tensor]]:
    """Encode tensors and return syntax statistics plus decoded tensors."""

    records = []
    payload = bytearray()
    decoded_tensors: dict[str, torch.Tensor] = {}
    for name in sorted(tensors):
        header, tensor_bytes, decoded = _tensor_record(
            name, tensors[name], quantization_bits
        )
        records.append(header)
        payload.extend(struct.pack("<I", len(tensor_bytes)))
        payload.extend(tensor_bytes)
        decoded_tensors[name] = decoded

    syntax = {
        "protocol": "MDL-UAP-v1",
        "mode": mode,
        "target": int(target),
        "epsilon": float(epsilon),
        "quantization_bits": int(quantization_bits),
        "records": records,
    }
    if attack_goal is not None:
        syntax["attack_goal"] = attack_goal
    header_bytes = json.dumps(
        syntax,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    compressed = zlib.compress(bytes(payload), level=9)
    total_bytes = len(header_bytes) + len(compressed)
    stats = {
        "bits": int(total_bytes * 8),
        "bytes": total_bytes,
        "header_bytes": len(header_bytes),
        "payload_bytes": len(compressed),
        "tensor_count": len(records),
        "compression": "zlib-level-9",
        "quantization_bits": int(quantization_bits),
        "protocol": "MDL-UAP-v1",
    }
    return stats, decoded_tensors


def mapping_description_length_bits(mapping_path: str, quantization_bits: int = 16) -> dict:
    """Return compressed bits for one fixed quantization scheme."""

    checkpoint = torch.load(mapping_path, map_location="cpu", weights_only=False)
    stats, _ = _encode_tensors(
        _mapping_tensors(checkpoint),
        mode=checkpoint["mode"],
        target=checkpoint["target"],
        epsilon=checkpoint["epsilon"],
        attack_goal=checkpoint.get("attack_goal"),
        quantization_bits=quantization_bits,
    )
    return stats


def quantized_mapping_state_dict(
    mapping_path: str, quantization_bits: int
) -> tuple[dict[str, torch.Tensor], dict]:
    """Return a decoded mapping state and its exact candidate code length."""

    checkpoint = torch.load(mapping_path, map_location="cpu", weights_only=False)
    stats, decoded = _encode_tensors(
        _mapping_tensors(checkpoint),
        mode=checkpoint["mode"],
        target=checkpoint["target"],
        epsilon=checkpoint["epsilon"],
        attack_goal=checkpoint.get("attack_goal"),
        quantization_bits=quantization_bits,
    )
    # Preserve integer buffers such as BatchNorm counters exactly.
    original = checkpoint.get("mapping", {})
    for name, tensor in original.items():
        if not tensor.is_floating_point() and name in decoded:
            decoded[name] = tensor.detach().cpu().clone()
    return decoded, stats
