from __future__ import annotations

from typing import Any


REQUIRED_FIELDS = (
    "fps",
    "beat_times",
    "beat_numbers",
    "beat_confidences",
    "downbeat_times",
    "downbeat_confidences",
)


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    raise TypeError(f"expected list, got {type(value).__name__}")


def compare_payloads(
    golden: dict[str, Any],
    candidate: dict[str, Any],
    *,
    float_tolerance: float = 0.0,
) -> list[str]:
    errors: list[str] = []

    for field in REQUIRED_FIELDS:
        if field not in golden:
            errors.append(f"golden missing required field: {field}")
        if field not in candidate:
            errors.append(f"candidate missing required field: {field}")
    if errors:
        return errors

    if int(golden["fps"]) != int(candidate["fps"]):
        errors.append(f"fps mismatch: golden={golden['fps']} candidate={candidate['fps']}")

    number_fields = ("beat_numbers",)
    for field in number_fields:
        g = _as_list(golden[field])
        c = _as_list(candidate[field])
        if len(g) != len(c):
            errors.append(f"{field} length mismatch: golden={len(g)} candidate={len(c)}")
            continue
        for i, (gv, cv) in enumerate(zip(g, c)):
            if int(gv) != int(cv):
                errors.append(f"{field}[{i}] mismatch: golden={gv} candidate={cv}")
                break

    float_fields = (
        "beat_times",
        "beat_confidences",
        "downbeat_times",
        "downbeat_confidences",
    )
    for field in float_fields:
        g = _as_list(golden[field])
        c = _as_list(candidate[field])
        if len(g) != len(c):
            errors.append(f"{field} length mismatch: golden={len(g)} candidate={len(c)}")
            continue
        for i, (gv, cv) in enumerate(zip(g, c)):
            delta = abs(float(gv) - float(cv))
            if delta > float_tolerance:
                errors.append(
                    f"{field}[{i}] mismatch: golden={gv} candidate={cv} abs_diff={delta} tol={float_tolerance}"
                )
                break

    return errors
