# ╔══════════════════════════════════════════════════════════════════╗
# ║  drawing.py — Skeleton overlay and HUD panel rendering          ║
# ╚══════════════════════════════════════════════════════════════════╝

import cv2
from landmarks import SKELETON_CONNECTIONS
from config import DEPTH_INSUFFICIENT_ANGLE

# Colour palette (BGR)
C = {
    "green":  (0, 220, 100),
    "red":    (30, 60, 240),
    "yellow": (0, 200, 255),
    "white":  (240, 240, 240),
    "accent": (0, 165, 255),
    "dim":    (100, 100, 130),
    "panel":  (12, 12, 22),
    "shadow": (15, 15, 15),
}


def draw_skeleton(frame, pts: dict, color: tuple):
    """
    Draw the skeleton overlay on frame.
    Two-pass render (shadow + colour) for visual stability.
    Only draws connections where both endpoints passed visibility filtering.
    """
    for a, b in SKELETON_CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], C["shadow"], 5, cv2.LINE_AA)
            cv2.line(frame, pts[a], pts[b], color,       3, cv2.LINE_AA)
    for pt in pts.values():
        cv2.circle(frame, pt, 7, C["shadow"], -1)
        cv2.circle(frame, pt, 6, color,       -1)
        cv2.circle(frame, pt, 8, C["white"],   1)



def draw_hud(
    frame,
    reps: int,
    clean_reps: int,
    stage: str,
    alerts: list,
    fps: float,
    lean: float,
    kl: float,
    kr: float,
    view: str,
    rep_was_clean,
    rep_fail_reason: str,
    min_angle_this_rep: float = 180.0,
    neck_angle: float = 180.0,
    heel_rise: float = 0.0,
):
    """Render the semi-transparent left-side HUD panel onto frame."""
    h, w = frame.shape[:2]

    # Semi-transparent dark panel
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (300, h), C["panel"], -1)
    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)

    # ── Title ─────────────────────────────────────────────────────────────
    cv2.putText(frame, "SQUAT ANALYZER", (12, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, C["accent"], 1, cv2.LINE_AA)
    cv2.putText(frame, f"[{view} VIEW]", (12, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, C["dim"], 1)
    cv2.line(frame, (12, 58), (288, 58), C["accent"], 1)

    # ── Rep counters ──────────────────────────────────────────────────────
    cv2.putText(frame, "TOTAL", (12, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, C["white"], 1)
    cv2.putText(frame, str(reps), (12, 135),
                cv2.FONT_HERSHEY_DUPLEX, 3.0, C["white"], 2, cv2.LINE_AA)
    cv2.putText(frame, "CLEAN", (160, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, C["green"], 1)
    cv2.putText(frame, str(clean_reps), (160, 135),
                cv2.FONT_HERSHEY_DUPLEX, 3.0, C["green"], 2, cv2.LINE_AA)

    # ── Stage badge ───────────────────────────────────────────────────────
    s_col = C["green"] if stage == "UP" else C["yellow"]
    cv2.rectangle(frame, (12, 145), (145, 165), s_col, -1)
    cv2.putText(frame, f" STAGE: {stage}", (14, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, C["panel"], 1)

    # ── Last rep result ───────────────────────────────────────────────────
    if rep_was_clean is not None:
        if rep_was_clean:
            cv2.putText(frame, "CLEAN REP", (12, 182),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, C["green"], 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, f"BAD: {rep_fail_reason[:22]}", (12, 182),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, C["red"], 1, cv2.LINE_AA)

    # ── Live metrics ──────────────────────────────────────────────────────
    cv2.line(frame, (12, 192), (288, 192), (40, 40, 60), 1)
    metrics = [
        f"Knee (L): {int(kl)}",
        f"Knee (R): {int(kr)}",
        f"Torso lean: {int(lean)}",
        f"Neck angle: {int(neck_angle)}",
        f"Heel rise:  {heel_rise:.3f}",
        f"FPS: {fps:.0f}",
    ]
    for i, m in enumerate(metrics):
        cv2.putText(frame, m, (12, 212 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (190, 190, 210), 1)

    # ── Depth bar ─────────────────────────────────────────────────────────
    bar_y = 336
    cv2.putText(frame, "DEPTH", (12, bar_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, C["dim"], 1)
    bar_w, bar_h        = 200, 10
    angle_high, angle_low = 155.0, 80.0
    current_angle       = (kl + kr) / 2
    depth_ratio  = max(0.0, min(1.0, (angle_high - current_angle) / (angle_high - angle_low)))
    target_ratio = (angle_high - DEPTH_INSUFFICIENT_ANGLE) / (angle_high - angle_low)
    cv2.rectangle(frame, (12, bar_y), (12 + bar_w, bar_y + bar_h), (40, 40, 60), -1)
    bar_color = C["green"] if depth_ratio >= target_ratio else C["yellow"]
    cv2.rectangle(frame, (12, bar_y), (12 + int(bar_w * depth_ratio), bar_y + bar_h), bar_color, -1)
    target_x = 12 + int(bar_w * target_ratio)
    cv2.line(frame, (target_x, bar_y - 2), (target_x, bar_y + bar_h + 2), C["white"], 1)

    # ── Alerts ────────────────────────────────────────────────────────────
    cv2.line(frame, (12, 360), (288, 360), (40, 40, 60), 1)
    if alerts:
        cv2.putText(frame, "! ALERTS", (12, 378),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, C["red"], 1)
        y = 398
        for alert in alerts[:4]:
            words, line = alert.split(), ""
            for word in words:
                if len(line + word) < 28:
                    line += word + " "
                else:
                    cv2.putText(frame, line.strip(), (14, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.36, C["red"], 1)
                    y += 16
                    line = word + " "
            cv2.putText(frame, line.strip(), (14, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, C["red"], 1)
            y += 20
    else:
        cv2.putText(frame, "POSTURE OK", (12, 378),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, C["green"], 1)

    cv2.putText(frame, "Q = quit   R = reset", (12, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, C["dim"], 1)


def draw_hud_generic(
    frame,
    exercise,           # ExerciseDefinition
    reps: int,
    clean_reps: int,
    stage: str,
    alerts: list,
    fps: float,
    view: str,
    joint_angles: dict,
    driver_angle: float,
    best_angle: float,
    rep_was_clean,
    rep_fail_reason: str,
):
    """
    Generic HUD panel for any exercise.
    Displays the exercise name, rep counters, stage, live joint angles,
    a depth/range bar for the driver joint, and current alerts.
    """
    h, w = frame.shape[:2]

    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (300, h), C["panel"], -1)
    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)

    # ── Title ──────────────────────────────────────────────────────────
    title = exercise.name.upper()[:20]
    cv2.putText(frame, title, (12, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, C["accent"], 1, cv2.LINE_AA)
    cv2.putText(frame, f"[{view} VIEW]", (12, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, C["dim"], 1)
    cv2.line(frame, (12, 58), (288, 58), C["accent"], 1)

    # ── Rep counters ───────────────────────────────────────────────────
    cv2.putText(frame, "TOTAL", (12, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, C["white"], 1)
    cv2.putText(frame, str(reps), (12, 135),
                cv2.FONT_HERSHEY_DUPLEX, 3.0, C["white"], 2, cv2.LINE_AA)
    cv2.putText(frame, "CLEAN", (160, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, C["green"], 1)
    cv2.putText(frame, str(clean_reps), (160, 135),
                cv2.FONT_HERSHEY_DUPLEX, 3.0, C["green"], 2, cv2.LINE_AA)

    # ── Stage badge ────────────────────────────────────────────────────
    s_col = C["green"] if stage == "UP" else C["yellow"]
    cv2.rectangle(frame, (12, 145), (145, 165), s_col, -1)
    cv2.putText(frame, f" STAGE: {stage}", (14, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, C["panel"], 1)

    # ── Last rep result ────────────────────────────────────────────────
    if rep_was_clean is not None:
        if rep_was_clean:
            cv2.putText(frame, "CLEAN REP", (12, 182),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, C["green"], 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, f"BAD: {rep_fail_reason[:22]}", (12, 182),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, C["red"], 1, cv2.LINE_AA)

    # ── Live joint angles ──────────────────────────────────────────────
    cv2.line(frame, (12, 192), (288, 192), (40, 40, 60), 1)
    y = 210
    for name, angle in list(joint_angles.items())[:6]:   # max 6 metrics
        cv2.putText(frame, f"{name}: {int(angle)}", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (190, 190, 210), 1)
        y += 19
    cv2.putText(frame, f"FPS: {fps:.0f}", (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (190, 190, 210), 1)

    # ── Driver joint range bar ─────────────────────────────────────────
    rt       = exercise.rep_trigger
    bar_y    = y + 22
    bar_w, bar_h = 200, 10

    # Find driver JointCheck for optimal angle
    driver_jc = next(
        (jc for jc in exercise.joint_checks if jc.is_rep_driver), None
    )
    optimal   = driver_jc.optimal_angle if driver_jc else 90.0
    angle_high = max(rt.enter_angle, rt.exit_angle) + 20
    angle_low  = max(0.0, min(rt.enter_angle, rt.exit_angle) - 20)
    if rt.direction == "decrease":
        ratio = max(0.0, min(1.0, (angle_high - driver_angle) / max(1, angle_high - angle_low)))
        target_ratio = (angle_high - optimal) / max(1, angle_high - angle_low)
    else:
        ratio = max(0.0, min(1.0, (driver_angle - angle_low) / max(1, angle_high - angle_low)))
        target_ratio = (optimal - angle_low) / max(1, angle_high - angle_low)

    cv2.putText(frame, "RANGE", (12, bar_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C["dim"], 1)
    cv2.rectangle(frame, (12, bar_y), (12 + bar_w, bar_y + bar_h), (40, 40, 60), -1)
    bar_color = C["green"] if ratio >= target_ratio * 0.85 else C["yellow"]
    cv2.rectangle(frame, (12, bar_y), (12 + int(bar_w * ratio), bar_y + bar_h), bar_color, -1)
    tx = 12 + int(bar_w * max(0.0, min(1.0, target_ratio)))
    cv2.line(frame, (tx, bar_y - 2), (tx, bar_y + bar_h + 2), C["white"], 1)

    # ── Alerts ─────────────────────────────────────────────────────────
    alert_y = bar_y + bar_h + 20
    cv2.line(frame, (12, alert_y - 6), (288, alert_y - 6), (40, 40, 60), 1)
    if alerts:
        cv2.putText(frame, "! ALERTS", (12, alert_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, C["red"], 1)
        y2 = alert_y + 32
        for alert in alerts[:4]:
            words, line = alert.split(), ""
            for word in words:
                if len(line + word) < 28:
                    line += ("" if not line else " ") + word
                else:
                    cv2.putText(frame, line.strip(), (14, y2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.36, C["red"], 1)
                    y2 += 16; line = word
            cv2.putText(frame, line.strip(), (14, y2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, C["red"], 1)
            y2 += 20
    else:
        cv2.putText(frame, "POSTURE OK", (12, alert_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, C["green"], 1)

    cv2.putText(frame, "Q = quit   R = reset", (12, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, C["dim"], 1)


