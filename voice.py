# ╔══════════════════════════════════════════════════════════════════╗
# ║  voice.py — Pre-cached alert audio playback                     ║
# ║  Plays .mp3 files from ./audio/ via subprocess (thread-safe).   ║
# ╚══════════════════════════════════════════════════════════════════╝

import os
import time
import threading
from config import TTS_COOLDOWN

AUDIO_DIR = "audio"

AUDIO_MAP = {
    "Knees caving in — push knees out":       f"{AUDIO_DIR}/knees_caving.mp3",
    "Leaning too far forward — chest up":     f"{AUDIO_DIR}/lean_forward.mp3",
    "Knee past toes — sit back more":         f"{AUDIO_DIR}/knee_past_toes.mp3",
    "Not deep enough":                        f"{AUDIO_DIR}/not_deep.mp3",
    "Heels lifting — press heels down":       f"{AUDIO_DIR}/heels_lifting.mp3",
    "Lower back rounding — brace your core":  f"{AUDIO_DIR}/butt_wink.mp3",
    "Neck not neutral — look straight ahead": f"{AUDIO_DIR}/neck_neutral.mp3",
}

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
_last_spoken:   dict = {}
_player_cmd:    str | None = None


def _find_player() -> str:
    import shutil
    for cmd in ("ffplay", "mpg123", "mpv", "aplay"):
        if shutil.which(cmd):
            return cmd
    return "none"


def init_tts():
    global _tts_available
    if any(os.path.exists(p) for p in AUDIO_MAP.values()):
        _tts_available = True
        print("🔊 TTS ready.")
    else:
        print("⚠  No audio files found in ./audio/ — running without voice alerts.")


def speak(text: str):
    global _player_cmd
    if not _tts_available:
        return
    now = time.time()
    if now - _last_spoken.get(text, 0) < TTS_COOLDOWN:
        return
    _last_spoken[text] = now
    path = AUDIO_MAP.get(text)
    if not path or not os.path.exists(path):
        return
    if _player_cmd is None:
        _player_cmd = _find_player()
    if _player_cmd == "none":
        return

    def _play():
        try:
            import subprocess
            if _player_cmd == "ffplay":
                subprocess.Popen(
                    ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            elif _player_cmd == "aplay":
                subprocess.Popen(
                    ["aplay", "-q", path],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            elif _player_cmd in ("mpg123", "mpv"):
                subprocess.Popen(
                    [_player_cmd, path],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
        except Exception:
            pass

    threading.Thread(target=_play, daemon=True).start()
