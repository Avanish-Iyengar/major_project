from math_utils import angle_2d, angle_3d
from landmarks import LM

def compute_joint_angle(jc, lms, pts):
    """Compute angle for a joint check, handling 2D/3D cases."""
    if jc.use_3d:
        try:
            return angle_3d(lms[LM[jc.landmark_a]], lms[LM[jc.landmark_b]], lms[LM[jc.landmark_c]])
        except Exception:
            return None
    if jc.landmark_a not in pts or jc.landmark_b not in pts or jc.landmark_c not in pts:
        return None
    return angle_2d(pts[jc.landmark_a], pts[jc.landmark_b], pts[jc.landmark_c])

def run_checks(definition, lms, pts, view):
    """Run joint checks for current view. Return alerts, joint_angles, driver_angle."""
    alerts, joint_angles, driver_angle = [], {}, 0.0
    for jc in definition.joint_checks:
        if view not in jc.check_in_views:
            continue
        angle = compute_joint_angle(jc, lms, pts)
        if angle is None:
            continue
        joint_angles[jc.display_name] = angle
        if jc.is_rep_driver and view in jc.driver_for_views:
            driver_angle = angle
        if jc.alert_too_low and angle < jc.min_angle:
            alerts.append(jc.alert_too_low)
        elif jc.alert_too_high and angle > jc.max_angle:
            alerts.append(jc.alert_too_high)
    return alerts, joint_angles, driver_angle
