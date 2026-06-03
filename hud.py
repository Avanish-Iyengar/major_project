import cv2
from drawing import C
from config import DEPTH_INSUFFICIENT_ANGLE


def draw_hud(frame, definition, state: dict, fps: float):
    """Draw semi-transparent left HUD panel onto frame in-place."""
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (300, h), C["panel"], -1)
    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)

    is_iso  = definition.isometric_trigger is not None
    title   = definition.name.upper()[:20]

    # Title
    cv2.putText(frame, title, (12, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, C["accent"], 1, cv2.LINE_AA)
    view_str = state.get("view", "FRONT")
    cv2.putText(frame, f"[{view_str} VIEW]", (12, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, C["dim"], 1)
    cv2.line(frame, (12, 58), (288, 58), C["accent"], 1)

    # Counters
    if is_iso:
        hold_secs  = state.get("hold_secs", 0.0)
        sets_done  = state.get("rep_count", 0)
        in_pos     = state.get("in_position", False)
        timer_col  = C["green"] if in_pos else C["yellow"]
        cv2.putText(frame, "HOLD TIME", (12, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, timer_col, 1)
        cv2.putText(frame, f"{hold_secs:.1f}s", (12, 135),
                    cv2.FONT_HERSHEY_DUPLEX, 2.4, timer_col, 2, cv2.LINE_AA)
        cv2.putText(frame, "SETS", (170, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, C["green"], 1)
        cv2.putText(frame, str(sets_done), (170, 135),
                    cv2.FONT_HERSHEY_DUPLEX, 3.0, C["green"], 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "TOTAL", (12, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, C["white"], 1)
        cv2.putText(frame, str(state.get("rep_count", 0)), (12, 135),
                    cv2.FONT_HERSHEY_DUPLEX, 3.0, C["white"], 2, cv2.LINE_AA)
        cv2.putText(frame, "CLEAN", (160, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, C["green"], 1)
        cv2.putText(frame, str(state.get("clean_reps", 0)), (160, 135),
                    cv2.FONT_HERSHEY_DUPLEX, 3.0, C["green"], 2, cv2.LINE_AA)

    # Stage/badge
    if is_iso:
        in_pos = state.get("in_position", False)
        b_col = C["green"] if in_pos else C["yellow"]
        b_text = "  IN POSITION  " if in_pos else "  GET IN POSITION  "
    else:
        stage = state.get("stage", "UP")
        b_col = C["green"] if stage == "UP" else C["yellow"]
        b_text = f"  STAGE: {stage}  "
    cv2.rectangle(frame, (12, 145), (288, 165), b_col, -1)
    cv2.putText(frame, b_text, (14, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, C["panel"], 1)

    # Last rep result (dynamic)
    if not is_iso:
        rep_was_clean = state.get("rep_was_clean")
        rep_fail_reason = state.get("rep_fail_reason", "")
        if rep_was_clean is True:
            cv2.putText(frame, "CLEAN REP", (12, 182),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, C["green"], 1, cv2.LINE_AA)
        elif rep_was_clean is False:
            cv2.putText(frame, f"BAD: {rep_fail_reason[:22]}", (12, 182),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, C["red"], 1, cv2.LINE_AA)

    # Joint metrics
    cv2.line(frame, (12, 192), (288, 192), (40, 40, 60), 1)
    joint_angles = state.get("joint_angles", {})
    y = 210
    for name, angle in list(joint_angles.items())[:6]:
        cv2.putText(frame, f"{name}: {int(angle)}", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (190, 190, 210), 1)
        y += 19
    cv2.putText(frame, f"FPS: {fps:.0f}", (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (190, 190, 210), 1)

    # Progress bar
    bar_y = y + 22
    cv2.putText(frame, "DEPTH" if not is_iso else "HOLD PROGRESS",
                (12, bar_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C["dim"], 1)
    bar_w, bar_h = 200, 10
    if is_iso:
        iso = definition.isometric_trigger
        hold_secs = state.get("hold_secs", 0.0)
        ratio = min(1.0, hold_secs / max(iso.hold_duration_secs, 1))
        target_ratio = 1.0
    else:
        rt = definition.rep_trigger
        driver_angle = state.get("driver_angle", 155.0)
        angle_high = max(rt.enter_angle, rt.exit_angle) + 20
        angle_low = max(0.0, min(rt.enter_angle, rt.exit_angle) - 20)
        span = max(1.0, angle_high - angle_low)
        if rt.direction == "decrease":
            ratio = max(0.0, min(1.0, (angle_high - driver_angle) / span))
        else:
            ratio = max(0.0, min(1.0, (driver_angle - angle_low) / span))
        target_ratio = (angle_high - DEPTH_INSUFFICIENT_ANGLE) / span
    cv2.rectangle(frame, (12, bar_y), (12 + bar_w, bar_y + bar_h), (40, 40, 60), -1)
    bar_color = C["green"] if ratio >= target_ratio * 0.85 else C["yellow"]
    cv2.rectangle(frame, (12, bar_y), (12 + int(bar_w * ratio), bar_y + bar_h), bar_color, -1)
    tx = 12 + int(bar_w * min(1.0, target_ratio))
    cv2.line(frame, (tx, bar_y - 2), (tx, bar_y + bar_h + 2), C["white"], 1)

    # Alerts
    alert_y = bar_y + bar_h + 20
    cv2.line(frame, (12, alert_y - 6), (288, alert_y - 6), (40, 40, 60), 1)
    alerts = state.get("alerts", [])
    if alerts:
        cv2.putText(frame, "! ALERTS", (12, alert_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, C["red"], 1)
        ay = alert_y + 32
        for alert in alerts[:4]:
            words, line = alert.split(), ""
            for word in words:
                if len(line + word) < 28:
                    line += ("" if not line else " ") + word
                else:
                    cv2.putText(frame, line.strip(), (14, ay),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.36, C["red"], 1)
                    ay += 16; line = word
            cv2.putText(frame, line.strip(), (14, ay),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, C["red"], 1)
            ay += 20
    else:
        cv2.putText(frame, "POSTURE OK", (12, alert_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, C["green"], 1)

    hint = "Q = quit   R = reset" if not is_iso else "Q = quit   R = reset hold"
    cv2.putText(frame, hint, (12, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, C["dim"], 1)
