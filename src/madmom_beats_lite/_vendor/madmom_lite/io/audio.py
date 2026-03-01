"""PCM-only build: decoding helpers intentionally disabled."""


class LoadAudioFileError(Exception):
    """Raised when file-based decoding is requested in PCM-only build."""


def _unsupported(*_args, **_kwargs):
    raise LoadAudioFileError(
        "File decoding is not supported in madmom-beats-lite. "
        "Use analyze_pcm(samples: np.ndarray, sample_rate=44100)."
    )


load_wave_file = _unsupported
write_wave_file = _unsupported
load_audio_file = _unsupported
load_ffmpeg_file = _unsupported
