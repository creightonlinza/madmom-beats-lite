from madmom_beats_lite.api import extract_beats
from madmom_beats_lite.types import BeatResult


def test_result_contract_shape() -> None:
    payload = BeatResult(
        fps=100,
        beat_times=[0.51, 1.03, 1.54],
        beat_numbers=[1, 2, 3],
        beat_confidences=[0.93, 0.88, 0.91],
        downbeat_times=[0.51],
        downbeat_confidences=[0.93],
    ).to_dict()

    assert list(payload.keys()) == [
        "fps",
        "beat_times",
        "beat_numbers",
        "beat_confidences",
        "downbeat_times",
        "downbeat_confidences",
    ]
    assert payload["fps"] == 100
    assert payload["beat_numbers"] == [1, 2, 3]


def test_extract_beats_input_validation() -> None:
    try:
        extract_beats("not-an-array", 44100)
    except TypeError as exc:
        assert "numpy.ndarray" in str(exc)
    else:
        raise AssertionError("TypeError expected")
