from madmom_beats_lite.progress import ProgressReporter


def test_progress_is_monotonic_and_integer() -> None:
    events = []
    reporter = ProgressReporter(events.append)

    reporter.emit(0, "start", "x")
    reporter.emit(20, "a", "x")
    reporter.emit(10, "b", "x")
    reporter.emit(101, "done", "x")

    percents = [e.percent for e in events]
    assert percents == [0, 20, 20, 100]
    assert all(isinstance(v, int) for v in percents)
