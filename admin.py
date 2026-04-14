#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║  admin.py — Interactive exercise file generator           ║
║                                                                  ║
║  Run:   python admin.py                                   ║
║  Creates a new file in exercises/ that you can run directly.    ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import re
import json
import requests
from config import OLLAMA_MODEL, OLLAMA_URL

# ── Available landmarks ───────────────────────────────────────────────────────
AVAILABLE_LANDMARKS = [
    "nose", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot", "right_foot",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    val = input(f"  {prompt}{suffix}: ").strip()
    return val if val else default

def ask_float(prompt: str, default: float) -> float:
    while True:
        raw = ask(prompt, str(default))
        try:
            return float(raw)
        except ValueError:
            print("      Please enter a number.")

def ask_int(prompt: str, default: int) -> int:
    while True:
        raw = ask(prompt, str(default))
        try:
            return int(raw)
        except ValueError:
            print("      Please enter a whole number.")

def ask_choice(prompt: str, choices: list, default: str) -> str:
    display = "/".join(choices)
    while True:
        val = ask(f"{prompt} ({display})", default).lower()
        if val in [c.lower() for c in choices]:
            return val
        print(f"      Choose one of: {display}")

def ask_landmark(prompt: str, default: str) -> str:
    print(f"\n    Available landmarks:")
    for i, lm in enumerate(AVAILABLE_LANDMARKS):
        print(f"      {i+1:>2}. {lm}")
    while True:
        raw = ask(prompt, default)
        if raw in AVAILABLE_LANDMARKS:
            return raw
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(AVAILABLE_LANDMARKS):
                return AVAILABLE_LANDMARKS[idx]
        except ValueError:
            pass
        print(f"    ⚠  Enter a landmark name or number (1–{len(AVAILABLE_LANDMARKS)})")

def slug(name: str) -> str:
    """Convert exercise name to a valid Python identifier."""
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')

# ── Ollama integration ────────────────────────────────────────────────────────

def call_ollama(prompt: str, model: str = None) -> dict:
    """Call Ollama API, return parsed JSON."""
    if model is None:
        model = OLLAMA_MODEL
    
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "format": "json",
                "stream": False
            },
            timeout=60
        )
        if response.status_code != 200:
            print(f"   Ollama error: {response.text}")
            return {}
        
        data = response.json()
        return json.loads(data.get("response", "{}"))
    
    except requests.exceptions.RequestException as e:
        print(f"   Ollama connection failed: {e}")
        return {}
    except json.JSONDecodeError:
        print("    Invalid JSON from Ollama")
        return {}

def build_extraction_prompt(user_text: str) -> str:
    """Build prompt for Ollama to extract exercise parameters."""
    return f"""You are an exercise definition extractor. Extract structured data from the user's description.

User description:
{user_text}

Extract and return ONLY valid JSON (no markdown, no explanation) with this example structure:

{{
  "name": "Exercise Name",
  "description": "One sentence description",
  "type": "dynamic" or "isometric",
  "valid_views": ["FRONT"] or ["SIDE"] or ["FRONT", "SIDE"],
  "joint_checks": [
    {{
      "display_name": "Knee (L)",
      "landmark_a": "left_hip",
      "landmark_b": "left_knee",
      "landmark_c": "left_ankle",
      "min_angle": 0.0,
      "optimal_angle": 90.0,
      "max_angle": 180.0,
      "alert_too_low": "",
      "alert_too_high": "",
      "check_in_views": ["FRONT", "SIDE"],
      "use_3d": false,
      "is_rep_driver": true
    }}
  ],
  "rep_trigger": {{
    "joint_display_name": "Knee (L)",
    "enter_angle": 105.0,
    "exit_angle": 155.0,
    "depth_target": 120.0,
    "depth_alert": "Not deep enough",
    "direction": "decrease"
  }},
  "isometric_trigger": null,
  "gemini_errors": ["error description 1"]
}}

For isometric exercises, set "rep_trigger": null and provide:
{{
  "isometric_trigger": {{
    "joint_display_name": "Knee (L)",
    "hold_min_angle": 55.0,
    "hold_max_angle": 95.0,
    "hold_duration_secs": 10.0,
    "alert_not_in_pos": "Bend to 55-95 degrees",
    "alert_too_low": "Too flexed",
    "alert_too_high": "Not bent enough"
  }}
}}

Available landmarks: {", ".join(AVAILABLE_LANDMARKS)}

Direction: "decrease" for flexion (squat, curl), "increase" for extension.
"""


def ai_create_exercise(model=None):
    """Use Ollama to generate exercise from natural language."""
    print("\n" + "═" * 60)
    print("  AI Exercise Generator (Ollama)")
    print("═" * 60)
    
    if model is None:
        model = OLLAMA_MODEL

    print("\nDescribe the exercise in natural language.")
    print("Example: 'Wall sit isometric hold. Monitor left knee at 90 degrees")
    print("         for 15 seconds. Alert if knee goes below 70 or above 110.'")
    print("\nYour description (Ctrl+D when done):")
    
    lines = []
    try:
        while True:
            lines.append(input())
    except EOFError:
        pass
    
    user_text = "\n".join(lines).strip()
    if not user_text:
        print("\n    No input. Cancelled.\n")
        return
    
    print("\n   Calling Ollama...")
    prompt = build_extraction_prompt(user_text)
    data = call_ollama(prompt, model)
    
    if not data or "name" not in data:
        print("   Failed to parse response. Try manual wizard.\n")
        return
    
    # Validate and clean
    ex_type = data.get("type", "dynamic")
    name = data.get("name", "Unnamed Exercise")
    
    print(f"\n  Extracted: {name} ({ex_type})")
    confirm = ask("Generate file? (yes/no)", "yes").lower()
    if confirm != "yes":
        print("  Cancelled.\n")
        return
    
    # Generate file
    filename = slug(name) + ".py"
    out_dir = os.path.join(os.path.dirname(__file__), "exercises")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    
    if ex_type == "isometric":
        # Prepare isometric data structure
        iso_data = {
            "name": name,
            "description": data.get("description", ""),
            "valid_views": data.get("valid_views", ["FRONT", "SIDE"]),
            "gemini_errors": data.get("gemini_errors", []),
            "isometric": data.get("isometric_trigger", {})
        }
        # Map fields if needed
        if "joint_display_name" in iso_data["isometric"]:
            iso_data["isometric"]["joint_name"] = iso_data["isometric"]["joint_display_name"]
        if "joint_checks" in data and len(data["joint_checks"]) > 0:
            jc = data["joint_checks"][0]
            iso_data["isometric"].setdefault("landmark_a", jc.get("landmark_a", "left_hip"))
            iso_data["isometric"].setdefault("landmark_b", jc.get("landmark_b", "left_knee"))
            iso_data["isometric"].setdefault("landmark_c", jc.get("landmark_c", "left_ankle"))
        
        code = generate_isometric_file(iso_data)
    else:
        code = generate_exercise_file(data)
    
    with open(out_path, "w") as f:
        f.write(code)
    
    print(f"\n✅ Created: exercises/{filename}")
    print(f"   Run it:  python exercises/{filename}\n")


# ── Code generation ───────────────────────────────────────────────────────────

def generate_exercise_file(data: dict) -> str:
    """Render the collected data into a Python exercise file string."""

    joint_check_blocks = []
    for i, jc in enumerate(data["joint_checks"]):
        is_driver = "True" if jc["is_rep_driver"] else "False"
        use_3d    = "True" if jc["use_3d"]        else "False"
        views_str = str(jc["check_in_views"])
        block = f"""\
            JointCheck(
                display_name   = {jc["display_name"]!r},
                landmark_a     = {jc["landmark_a"]!r},
                landmark_b     = {jc["landmark_b"]!r},
                landmark_c     = {jc["landmark_c"]!r},
                min_angle      = {jc["min_angle"]},
                optimal_angle  = {jc["optimal_angle"]},
                max_angle      = {jc["max_angle"]},
                alert_too_low  = {jc["alert_too_low"]!r},
                alert_too_high = {jc["alert_too_high"]!r},
                check_in_views = {views_str},
                use_3d         = {use_3d},
                is_rep_driver  = {is_driver},
            ),"""
        joint_check_blocks.append(block)

    joint_checks_str = "\n".join(joint_check_blocks)

    rt = data["rep_trigger"]
    gemini_errors_str = "\n".join(
        f"            {e!r}," for e in data["gemini_errors"]
    )
    valid_views_str = str(data["valid_views"])
    name_slug       = slug(data["name"])

    return f'''\
# ╔══════════════════════════════════════════════════════════════════╗
# ║  exercises/{name_slug}.py — {data["name"]}
# ║  Run:  python exercises/{name_slug}.py
# ╚══════════════════════════════════════════════════════════════════╝

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from exercise_base import ExerciseDefinition, JointCheck, RepTrigger
from exercise_runner import run_exercise


def get_definition() -> ExerciseDefinition:
    return ExerciseDefinition(
        name        = {data["name"]!r},
        description = {data["description"]!r},
        valid_views = {valid_views_str},

        joint_checks = [
{joint_checks_str}
        ],

        rep_trigger = RepTrigger(
            joint_display_name = {rt["joint_display_name"]!r},
            enter_angle        = {rt["enter_angle"]},
            exit_angle         = {rt["exit_angle"]},
            depth_target       = {rt["depth_target"]},
            depth_alert        = {rt["depth_alert"]!r},
            direction          = {rt["direction"]!r},
        ),

        extra_checks  = [],   # add custom check functions here if needed

        gemini_errors = [
{gemini_errors_str}
        ],
    )


if __name__ == "__main__":
    run_exercise(get_definition())
'''

# ── Isometric exercise generator ──────────────────────────────────────────────

def generate_isometric_file(data: dict) -> str:
    name_slug       = slug(data["name"])
    valid_views_str = str(data["valid_views"])
    gemini_str      = "\n".join(f"            {e!r}," for e in data["gemini_errors"])
    iso             = data["isometric"]

    return f'''\
# exercises/{name_slug}.py — {data["name"]}
# Run:  python exercises/{name_slug}.py

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exercise_base import ExerciseDefinition, JointCheck, IsometricTrigger
from exercise_runner import run_exercise

HOLD_MIN    = {iso["hold_min_angle"]}
HOLD_MAX    = {iso["hold_max_angle"]}
HOLD_TARGET = {iso["hold_duration_secs"]}

def get_definition() -> ExerciseDefinition:
    return ExerciseDefinition(
        name        = {data["name"]!r},
        description = {data["description"]!r},
        valid_views = {valid_views_str},

        joint_checks = [
            JointCheck(
                display_name    = {iso["joint_name"]!r},
                landmark_a      = {iso["landmark_a"]!r},
                landmark_b      = {iso["landmark_b"]!r},
                landmark_c      = {iso["landmark_c"]!r},
                min_angle       = HOLD_MIN,
                optimal_angle   = (HOLD_MIN + HOLD_MAX) / 2,
                max_angle       = HOLD_MAX,
                alert_too_low   = {iso["alert_too_low"]!r},
                alert_too_high  = {iso["alert_too_high"]!r},
                check_in_views  = {valid_views_str},
                use_3d          = False,
                is_rep_driver   = False,
            ),
        ],

        rep_trigger = None,

        isometric_trigger = IsometricTrigger(
            joint_display_name = {iso["joint_name"]!r},
            hold_min_angle     = HOLD_MIN,
            hold_max_angle     = HOLD_MAX,
            hold_duration_secs = HOLD_TARGET,
            alert_not_in_pos   = {iso["alert_not_in_pos"]!r},
            alert_break        = "Position lost — return to hold position",
        ),

        extra_checks  = [],   # add custom checks here if needed

        gemini_errors = [
{gemini_str}
        ],
    )


if __name__ == "__main__":
    run_exercise(get_definition())
'''


def _run_isometric_wizard(name, description, valid_views):
    print("\n── Isometric hold settings ───────────────────────────────")
    print("  Define the joint to monitor and the acceptable hold zone.\n")

    joint_name  = ask("  Joint display name", "Knee (L)")
    print("  Which landmarks form the angle? (A → B vertex → C)")
    lm_a        = ask_landmark("  Landmark A", "left_hip")
    lm_b        = ask_landmark("  Landmark B (vertex)", "left_knee")
    lm_c        = ask_landmark("  Landmark C", "left_ankle")
    hold_min    = ask_float("  Minimum hold angle (degrees)", 55.0)
    hold_max    = ask_float("  Maximum hold angle (degrees)", 95.0)
    hold_target = ask_float("  Hold duration goal (seconds)", 10.0)
    alt_low     = ask("  Alert when angle too low", "Knee too flexed — straighten slightly")
    alt_high    = ask("  Alert when angle too high", "Knee not bent enough — increase flexion")
    alt_not_in  = ask("  Alert when not in hold zone", f"Bend to {int(hold_min)}-{int(hold_max)} degrees to begin hold")

    print("\n── Gemini error descriptions ──────────────────────────────")
    gemini_errors = [
        ask("  Error desc (too low)",  f'"Angle too low" — {joint_name} below {hold_min} degrees'),
        ask("  Error desc (too high)", f'"Angle too high" — {joint_name} above {hold_max} degrees'),
        ask("  Error desc (position lost)", '"Position lost" — exited hold zone before completing set'),
    ]

    data = {
        "name":        name,
        "description": description,
        "valid_views": valid_views,
        "gemini_errors": [e for e in gemini_errors if e],
        "isometric": {
            "joint_name":    joint_name,
            "landmark_a":    lm_a,
            "landmark_b":    lm_b,
            "landmark_c":    lm_c,
            "hold_min":      hold_min,
            "hold_max":      hold_max,
            "hold_target":   hold_target,
            "alert_too_low": alt_low,
            "alert_too_high":alt_high,
            "alert_not_in_pos": alt_not_in,
        }
    }

    filename = slug(name) + ".py"
    out_dir  = os.path.join(os.path.dirname(__file__), "exercises")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)

    with open(out_path, "w") as f:
        f.write(generate_isometric_file(data))

    print(f"\n✅ Created: exercises/{filename}")
    print(f"   Run it:  python exercises/{filename}\n")


# ── List exercises ───────────────────────────────────────────────────────────

def list_exercises():
    ex_dir = os.path.join(os.path.dirname(__file__), "exercises")
    if not os.path.exists(ex_dir):
        print("\n  No exercises/ directory found.\n")
        return []
    
    files = [f for f in os.listdir(ex_dir) if f.endswith(".py") and f != "__init__.py"]
    if not files:
        print("\n  No exercise files found.\n")
        return []
    
    print("\n  Available exercises:")
    for i, f in enumerate(sorted(files), 1):
        print(f"    {i:>2}. {f[:-3]}")
    return sorted(files)


# ── Read existing exercise ────────────────────────────────────────────────────

def read_exercise(filename):
    ex_dir = os.path.join(os.path.dirname(__file__), "exercises")
    path = os.path.join(ex_dir, filename)
    
    if not os.path.exists(path):
        print(f"\n  ⚠  File not found: {filename}\n")
        return
    
    with open(path, "r") as f:
        content = f.read()
    
    print("\n" + "─" * 60)
    print(f"  {filename}")
    print("─" * 60)
    print(content)
    print("─" * 60 + "\n")


# ── Delete exercise ───────────────────────────────────────────────────────────

def delete_exercise(filename):
    ex_dir = os.path.join(os.path.dirname(__file__), "exercises")
    path = os.path.join(ex_dir, filename)
    
    if not os.path.exists(path):
        print(f"\n  ⚠  File not found: {filename}\n")
        return
    
    confirm = ask(f"Delete {filename}? (yes/no)", "no").lower()
    if confirm != "yes":
        print("  Cancelled.\n")
        return
    
    os.remove(path)
    print(f"\n  ✅ Deleted: {filename}\n")


# ── Create new exercise ───────────────────────────────────────────────────────

def create_exercise():
    print("\n" + "═" * 60)
    print("  Create New Exercise")
    print("═" * 60)
    print("  Press Enter to accept defaults shown in [brackets].\n")

    # ── Basic info ────────────────────────────────────────────────────
    print("── Exercise info ─────────────────────────────────────────")
    name        = ask("Exercise name (e.g. Lunge, Shoulder Press)")
    description = ask("One-line description",
                      f"Analysis of the {name} exercise.")
    ex_type     = ask_choice("Exercise type", ["dynamic", "isometric"], "dynamic")
    views_raw   = ask_choice("Valid camera views", ["front", "side", "both"], "both")
    valid_views = (
        ["FRONT", "SIDE"] if views_raw == "both" else
        ["FRONT"]         if views_raw == "front" else
        ["SIDE"]
    )

    if ex_type == "isometric":
        _run_isometric_wizard(name, description, valid_views)
        return

    # ── Joint checks ──────────────────────────────────────────────────
    print("\n── Joint checks ──────────────────────────────────────────")
    print("  Define each joint angle rule. At least one must be the")
    print("  rep driver (the joint whose angle counts reps).")
    n_joints = ask_int("How many joint checks?", 2)

    joint_checks = []
    rep_driver_name = None

    for i in range(n_joints):
        print(f"\n  Joint check {i+1} of {n_joints}:")
        display_name = ask("  Display name (e.g. 'Knee (L)')", f"Joint {i+1}")
        print(f"  Landmark A → B (vertex) → C defines the angle at B.")
        lm_a  = ask_landmark("  Landmark A", "left_hip")
        lm_b  = ask_landmark("  Landmark B (vertex)", "left_knee")
        lm_c  = ask_landmark("  Landmark C", "left_ankle")
        min_a = ask_float("  Min angle (below this = problem, 0 = no min check)", 0.0)
        opt_a = ask_float("  Optimal angle (target marker on bar)", 90.0)
        max_a = ask_float("  Max angle (above this = problem, 180 = no max check)", 180.0)
        alt_low  = ask("  Alert when too low  (leave blank to skip)", "")
        alt_high = ask("  Alert when too high (leave blank to skip)", "")
        view_c   = ask_choice("  Active in which views", ["front", "side", "both"], "both")
        use_3d   = ask_choice("  Use 3D angle (better for side view)", ["yes", "no"], "no") == "yes"
        is_driver = ask_choice("  Is this the rep-driving joint?", ["yes", "no"],
                               "yes" if i == 0 else "no") == "yes"
        if is_driver:
            rep_driver_name = display_name

        jc_views = (
            ["FRONT", "SIDE"] if view_c == "both" else
            ["FRONT"]         if view_c == "front" else
            ["SIDE"]
        )
        joint_checks.append({
            "display_name":   display_name,
            "landmark_a":     lm_a,
            "landmark_b":     lm_b,
            "landmark_c":     lm_c,
            "min_angle":      min_a,
            "optimal_angle":  opt_a,
            "max_angle":      max_a,
            "alert_too_low":  alt_low,
            "alert_too_high": alt_high,
            "check_in_views": jc_views,
            "use_3d":         use_3d,
            "is_rep_driver":  is_driver,
        })

    if rep_driver_name is None:
        rep_driver_name = joint_checks[0]["display_name"]
        joint_checks[0]["is_rep_driver"] = True
        print(f"  ⚠  No rep driver selected — defaulting to '{rep_driver_name}'")

    # ── Rep trigger ───────────────────────────────────────────────────
    print("\n── Rep trigger ───────────────────────────────────────────")
    print(f"  Driver joint: {rep_driver_name}")
    direction    = ask_choice("  Direction", ["decrease", "increase"], "decrease")
    enter_angle  = ask_float("  Enter DOWN when angle crosses", 145.0)
    exit_angle   = ask_float("  Return UP when angle crosses",  155.0)
    depth_target = ask_float("  Minimum angle required for a valid rep (0 = no check)", 0.0)
    depth_alert  = ask("  Alert when depth not reached", "Not deep enough") if depth_target > 0 else ""

    # ── Gemini error descriptions ─────────────────────────────────────
    print("\n── Gemini error descriptions ──────────────────────────────")
    print("  These are sent to Gemini so it understands each error.")
    print("  Press Enter with no input to finish.")
    gemini_errors = []
    for jc in joint_checks:
        if jc["alert_too_low"]:
            suggested = f'"{jc["alert_too_low"]}" — {jc["display_name"]} angle below {jc["min_angle"]}°'
            confirmed = ask(f"  Error desc", suggested)
            gemini_errors.append(confirmed)
        if jc["alert_too_high"]:
            suggested = f'"{jc["alert_too_high"]}" — {jc["display_name"]} angle above {jc["max_angle"]}°'
            confirmed = ask(f"  Error desc", suggested)
            gemini_errors.append(confirmed)
    if depth_target > 0 and depth_alert:
        suggested = f'"{depth_alert}" — joint never reached {depth_target}° during rep'
        confirmed = ask(f"  Error desc", suggested)
        gemini_errors.append(confirmed)

    # ── Assemble and write ────────────────────────────────────────────
    data = {
        "name":        name,
        "description": description,
        "valid_views": valid_views,
        "joint_checks": joint_checks,
        "rep_trigger": {
            "joint_display_name": rep_driver_name,
            "enter_angle":        enter_angle,
            "exit_angle":         exit_angle,
            "depth_target":       depth_target,
            "depth_alert":        depth_alert,
            "direction":          direction,
        },
        "gemini_errors": gemini_errors,
    }

    filename  = slug(name) + ".py"
    out_dir   = os.path.join(os.path.dirname(__file__), "exercises")
    os.makedirs(out_dir, exist_ok=True)
    out_path  = os.path.join(out_dir, filename)

    code = generate_exercise_file(data)
    with open(out_path, "w") as f:
        f.write(code)

    print(f"\n✅ Created: exercises/{filename}")
    print(f"   Run it:  python exercises/{filename}\n")


if __name__ == "__main__":
    while True:
        print("\n" + "═" * 60)
        print("  XAI Exercise Admin")
        print("═" * 60)
        print("  1. Create new exercise (wizard)")
        print("  2. Create with AI (Ollama)")
        print("  3. List exercises")
        print("  4. Read exercise file")
        print("  5. Delete exercise")
        print("  6. Exit")
        
        choice = ask("\n  Choose", "1")
        
        if choice == "1":
            create_exercise()
        elif choice == "2":
            ai_create_exercise()
        elif choice == "3":
            list_exercises()
        elif choice == "4":
            files = list_exercises()
            if files:
                idx = ask_int("  Select number", 1) - 1
                if 0 <= idx < len(files):
                    read_exercise(files[idx])
        elif choice == "5":
            files = list_exercises()
            if files:
                idx = ask_int("  Select number", 1) - 1
                if 0 <= idx < len(files):
                    delete_exercise(files[idx])
        elif choice == "6":
            print("\n  Bye.\n")
            break
        else:
            print("  Invalid choice.\n")
