# ╔══════════════════════════════════════════════════════════════════╗
# ║  smoother.py — Exponential Moving Average landmark smoother     ║
# ║  Reduces per-frame jitter in MediaPipe landmark coordinates     ║
# ╚══════════════════════════════════════════════════════════════════╝

from config import EMA_ALPHA

_smoothed: dict[str, list] = {}


def smooth_landmark(name: str, lm) -> tuple:
    """
    EMA-smooth a single MediaPipe landmark.
    Returns (x, y, z) smoothed normalised coordinates.
    On first call for a landmark, seeds with the raw value (no lag spike).
    """
    new = [lm.x, lm.y, lm.z]
    if name not in _smoothed:
        _smoothed[name] = new[:]
    else:
        s = _smoothed[name]
        _smoothed[name] = [
            EMA_ALPHA * n + (1.0 - EMA_ALPHA) * s[i]
            for i, n in enumerate(new)
        ]
    return tuple(_smoothed[name])


def reset_smoother():
    """Clear all smoothed state. Call on session reset."""
    _smoothed.clear()
