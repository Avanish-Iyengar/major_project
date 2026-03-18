# ╔══════════════════════════════════════════════════════════════════╗
# ║  voice.py — Pre-cached audio playback via pygame                ║
# ║  Place your .mp3 files in ./audio/ before running.             ║
# ╚══════════════════════════════════════════════════════════════════╝

import os
import time
import threading
from config import TTS_COOLDOWN

AUDIO_DIR = "audio"

# Maps alert strings (exactly as appended to alerts[]) → .mp3 file paths
AUDIO_MAP = {
    "Knees caving in — push knees out":       f"{AUDIO_DIR}/knees_caving.mp3",
    "Leaning too far forward — chest up":     f"{AUDIO_DIR}/lean_forward.mp3",
    "Knee past toes — sit back more":         f"{AUDIO_DIR}/knee_past_toes.mp3",
    "Not deep enough":                        f"{AUDIO_DIR}/not_deep.mp3",
    "Heels lifting — press heels down":       f"{AUDIO_DIR}/heels_lifting.mp3",
    "Lower back rounding — brace your core":  f"{AUDIO_DIR}/butt_wink.mp3",
    "Neck not neutral — look straight ahead": f"{AUDIO_DIR}/neck_neutral.mp3",
}

# Human-readable spoken phrases (kept for reference / future re-generation)
AUDIO_PHRASES = {
    "Knees caving in — push knees out":       "Push your knees out",
    "Leaning too far forward — chest up":     "Chest up, stop leaning forward",
    "Knee past toes — sit back more":         "Sit back more, your knee is past your toes",
    "Not deep enough":                        "Go deeper on your squat",
    "Heels lifting — press heels down":       "Press your heels into the ground",
    "Lower back rounding — brace your core":  "Brace your core, your lower back is rounding",
    "Neck not neutral — look straight ahead": "Look straight ahead, keep your neck neutral",
}

_tts_available: bool = False
_last_spoken: dict[str, float] = {}


def init_tts():
    """Initialise pygame mixer and confirm audio files are present."""
    global _tts_available
    try:
        import pygame
        pygame.mixer.init()
        if any(os.path.exists(p) for p in AUDIO_MAP.values()):
            _tts_available = True
            print("🔊 TTS ready.")
        else:
            print("⚠  No audio files found in ./audio/ — running without voice alerts.")
    except Exception as e:
        print(f"⚠  pygame unavailable ({e}) — no voice output.")


def speak(text: str):
    """
    Play the pre-cached .mp3 for the given alert text.
    Non-blocking (runs in a daemon thread).
    Respects TTS_COOLDOWN — same alert won't repeat within cooldown window.
    """
    if not _tts_available:
        return
    now = time.time()
    if now - _last_spoken.get(text, 0) < TTS_COOLDOWN:
        return
    _last_spoken[text] = now
    path = AUDIO_MAP.get(text)
    if not path or not os.path.exists(path):
        return

    def _play():
        try:
            import pygame
            pygame.mixer.Sound(path).play()
        except Exception:
            pass

    threading.Thread(target=_play, daemon=True).start()
