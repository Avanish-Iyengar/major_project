# ╔══════════════════════════════════════════════════════════════════╗
# ║  gemini.py — Post-session Gemini coaching report                ║
# ║  Builds a prompt from session data, calls the API, renders the  ║
# ║  response in a scrollable OpenCV window.                        ║
# ╚══════════════════════════════════════════════════════════════════╝

from config import GEMINI_API_KEY

# Error keys — must match exactly what posture_checks.py appends to alerts[]
ALL_ERRORS = [
    "Knees caving in — push knees out",
    "Leaning too far forward — chest up",
    "Knee past toes — sit back more",
    "Not deep enough",
    "Heels lifting — press heels down",
    "Lower back rounding — brace your core",
    "Neck not neutral — look straight ahead",
]


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_gemini_prompt(session_log: list[dict], total: int, clean: int) -> str:
    """
    Converts the per-rep session log into a structured coaching prompt.
    session_log entries: {rep, errors: [str], depth_angle: int, clean: bool}
    """
    if not session_log:
        return ""

    error_counts: dict[str, int] = {e: 0 for e in ALL_ERRORS}
    for rep in session_log:
        for err in rep["errors"]:
            if err in error_counts:
                error_counts[err] += 1

    rep_lines = [
        f"  Rep {rep['rep']:>2}: {'CLEAN' if rep['clean'] else 'FAIL'} | "
        f"depth {rep['depth_angle']}° | "
        f"errors: {', '.join(rep['errors']) if rep['errors'] else 'none'}"
        for rep in session_log
    ]

    error_summary = "\n".join(
        f"  {count:>2}x  {err}"
        for err, count in sorted(error_counts.items(), key=lambda x: -x[1])
        if count > 0
    ) or "  None detected."

    return f"""You are an expert physiotherapy coach and movement specialist.
A patient just completed a squat session analyzed by a pose-estimation system.
Write a clear, encouraging, and actionable coaching report based on the data below.

SESSION DATA
Total reps:  {total}
Clean reps:  {clean}
Failed reps: {total - clean}
Clean rate:  {int(clean / total * 100) if total else 0}%

Error frequency (reps each error appeared):
{error_summary}

Rep-by-rep breakdown:
{chr(10).join(rep_lines)}

THE 7 ERRORS THIS SYSTEM DETECTS
1. "Knees caving in"          Valgus collapse: knees track inward during descent.
2. "Leaning too far forward"  Torso tilts more than 55 degrees from vertical.
3. "Knee past toes"           Side view: knee travels forward of the ankle.
4. "Not deep enough"          Knee never reached 120 degrees or below (parallel depth).
5. "Heels lifting"            Heels rise off the ground during descent.
6. "Lower back rounding"      Posterior pelvic tilt (butt wink) at the bottom.
7. "Neck not neutral"         Neck craning up or down excessively.

REPORT STRUCTURE — use EXACTLY these four sections:

## Overall Performance
One short paragraph on how the session went. Be honest but encouraging.

## Main Issues (prioritised)
Top 1-3 most frequent or impactful errors. For each:
- Name it clearly
- Explain what is physically happening and why it matters biomechanically
- Give 2-3 specific, actionable cues or drills to fix it

## What You Did Well
Acknowledge positives: consistency, errors that didn't appear, improvement across the set.

## Focus for Next Session
One clear, simple priority for next time.

Tone: professional but warm, like a physiotherapist talking to their patient.
Do NOT use asterisks. Plain text only. Keep the total report under 400 words.
"""


# ── API call ──────────────────────────────────────────────────────────────────

def call_gemini(prompt: str) -> str:
    """
    Calls Gemini 2.5 Flash via REST API.
    Returns the coaching text, or a descriptive ERROR string on failure.
    """
    try:
        import requests as req
    except ImportError:
        return "ERROR: 'requests' not installed. Run: pip install requests"

    if "YOUR_" in GEMINI_API_KEY:
        return (
            "ERROR: GEMINI_API_KEY not set.\n\n"
            "Get a free key at aistudio.google.com → Get API Key\n"
            "Then paste it into GEMINI_API_KEY in config.py"
        )

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-2.5-flash:generateContent"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": 2048},
    }
    try:
        r = req.post(
            url,
            headers={"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        if r.status_code == 200:
            return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        return f"ERROR {r.status_code}: {r.text[:300]}"
    except Exception as e:
        return f"ERROR: {e}"



# ── Generic prompt builder (used by exercise_runner) ─────────────────────────

def build_gemini_prompt_generic(definition, session_log: list[dict], total: int, clean: int) -> str:
    """
    Builds a Gemini coaching prompt from any ExerciseDefinition + session log.
    """
    if not session_log:
        return ""

    all_errors = definition.gemini_errors or [e for e in ALL_ERRORS]
    error_counts = {e: 0 for e in all_errors}
    for rep in session_log:
        for err in rep.get("errors", []):
            if err in error_counts:
                error_counts[err] += 1
            else:
                error_counts[err] = error_counts.get(err, 0) + 1

    rep_lines = [
        f"  Rep {rep['rep']:>2}: {'CLEAN' if rep['clean'] else 'FAIL'} | "
        f"best angle {rep.get('best_angle', '?')}° | "
        f"errors: {', '.join(rep['errors']) if rep['errors'] else 'none'}"
        for rep in session_log
    ]

    error_summary = "\n".join(
        f"  {count:>2}x  {err}"
        for err, count in sorted(error_counts.items(), key=lambda x: -x[1])
        if count > 0
    ) or "  None detected."

    error_descriptions = "\n".join(
        f"{i+1}. {e}" for i, e in enumerate(definition.gemini_errors)
    ) or "  (No specific error descriptions provided.)"

    return f"""You are an expert physiotherapy coach and movement specialist.
A patient just completed a {definition.name} session analyzed by a pose-estimation system.
Exercise: {definition.description}

SESSION DATA
Total reps:  {total}
Clean reps:  {clean}
Failed reps: {total - clean}
Clean rate:  {int(clean / total * 100) if total else 0}%

Error frequency (reps each error appeared):
{error_summary}

Rep-by-rep breakdown:
{chr(10).join(rep_lines)}

ERRORS THIS SYSTEM DETECTS FOR THIS EXERCISE
{error_descriptions}

REPORT STRUCTURE — use EXACTLY these four sections:

## Overall Performance
One short paragraph on how the session went. Be honest but encouraging.

## Main Issues (prioritised)
Top 1-3 most frequent or impactful errors. For each:
- Name it clearly
- Explain what is physically happening and why it matters biomechanically
- Give 2-3 specific, actionable cues or drills to fix it

## What You Did Well
Acknowledge positives: consistency, errors that didn't appear, improvement across the set.

## Focus for Next Session
One clear, simple priority for next time.

Tone: professional but warm, like a physiotherapist talking to their patient.
Do NOT use asterisks. Plain text only. Keep the total report under 400 words.
"""


# ── cv2 report window ─────────────────────────────────────────────────────────

import cv2
import numpy as np
import threading

_report_state: dict = {
    "text":  "  Analyzing your session with Gemini...",
    "ready": False,
}


def _fetch_thread(prompt: str):
    result = call_gemini(prompt)
    _report_state["text"]  = result
    _report_state["ready"] = True


def _wrap_text(text: str, max_chars: int) -> list:
    out = []
    for paragraph in text.split("\n"):
        if paragraph.strip() == "":
            out.append("")
            continue
        words, line = paragraph.split(), ""
        for word in words:
            if len(line) + len(word) + 1 <= max_chars:
                line += ("" if not line else " ") + word
            else:
                if line:
                    out.append(line)
                line = word
        if line:
            out.append(line)
    return out


def show_report_window(prompt: str):
    """
    Scrollable cv2 window showing the Gemini coaching report.
    API call runs in background thread — window opens instantly.
    Controls: UP/DOWN or W/S to scroll. Q or ESC to close.
    """
    WIN   = "Gemini Coaching Report"
    WIN_W, WIN_H = 900, 700
    BG    = (18, 18, 28)
    HDR   = (0, 165, 255)
    TXT   = (220, 220, 230)
    DIM   = (90, 90, 110)
    SEC   = (100, 210, 140)
    ERR   = (60, 80, 220)
    FONT  = cv2.FONT_HERSHEY_SIMPLEX
    LINE_H    = 26
    MARGIN_X  = 32
    MAX_CHARS = (WIN_W - MARGIN_X * 2) // 8

    _report_state["text"]  = "  Analyzing your session with Gemini..."
    _report_state["ready"] = False
    threading.Thread(target=_fetch_thread, args=(prompt,), daemon=True).start()

    scroll    = 0
    dot_frame = 0

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, WIN_W, WIN_H)

    while True:
        canvas = np.full((WIN_H, WIN_W, 3), BG, dtype=np.uint8)

        cv2.rectangle(canvas, (0, 0), (WIN_W, 52), (10, 10, 20), -1)
        cv2.putText(canvas, "GEMINI COACHING REPORT", (MARGIN_X, 34),
                    FONT, 0.75, HDR, 1, cv2.LINE_AA)
        cv2.putText(canvas, "UP/DOWN = scroll    Q/ESC = close", (WIN_W - 320, 34),
                    FONT, 0.38, DIM, 1)
        cv2.line(canvas, (0, 52), (WIN_W, 52), HDR, 1)

        body_top    = 66
        body_bottom = WIN_H - 34
        visible_h   = body_bottom - body_top
        max_lines   = visible_h // LINE_H

        if not _report_state["ready"]:
            dots = "." * ((dot_frame // 8 % 3) + 1)
            dot_frame += 1
            cv2.putText(canvas, f"  Analyzing your session with Gemini{dots}",
                        (MARGIN_X, body_top + 44), FONT, 0.55, TXT, 1, cv2.LINE_AA)
            cv2.putText(canvas, "  (This usually takes 3-8 seconds)",
                        (MARGIN_X, body_top + 74), FONT, 0.42, DIM, 1)
        else:
            lines       = _wrap_text(_report_state["text"], MAX_CHARS)
            total_lines = len(lines)
            max_scroll  = max(0, total_lines - max_lines)
            scroll      = max(0, min(scroll, max_scroll))

            y = body_top + LINE_H
            for line in lines[scroll: scroll + max_lines]:
                if line.startswith("## "):
                    cv2.putText(canvas, line[3:], (MARGIN_X, y),
                                FONT, 0.54, SEC, 1, cv2.LINE_AA)
                elif line.upper().startswith("ERROR"):
                    cv2.putText(canvas, line, (MARGIN_X, y),
                                FONT, 0.44, ERR, 1, cv2.LINE_AA)
                else:
                    cv2.putText(canvas, line, (MARGIN_X, y),
                                FONT, 0.44, TXT, 1, cv2.LINE_AA)
                y += LINE_H

            if total_lines > max_lines:
                sb_h = max(20, int(visible_h * max_lines / total_lines))
                sb_y = body_top + int(visible_h * scroll / max(1, total_lines))
                sb_y = min(sb_y, body_bottom - sb_h)
                cv2.rectangle(canvas, (WIN_W - 10, body_top),
                              (WIN_W - 5, body_bottom), (35, 35, 50), -1)
                cv2.rectangle(canvas, (WIN_W - 10, sb_y),
                              (WIN_W - 5, sb_y + sb_h), HDR, -1)

        cv2.line(canvas, (0, body_bottom + 2), (WIN_W, body_bottom + 2), (40, 40, 55), 1)
        status = "Fetching..." if not _report_state["ready"] else "Report complete."
        cv2.putText(canvas, status, (MARGIN_X, WIN_H - 10), FONT, 0.35, DIM, 1)

        cv2.imshow(WIN, canvas)
        key = cv2.waitKey(60) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
        elif key == 82 or key == ord('w'):
            scroll = max(0, scroll - 3)
        elif key == 84 or key == ord('s'):
            scroll += 3

    cv2.destroyWindow(WIN)
