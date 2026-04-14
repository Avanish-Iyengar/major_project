"""
╔══════════════════════════════════════════════════════════════════╗
║  XAI Physiotherapy — Main Menu                                  ║
║                                                                  ║
║  Run:  python main.py                                           ║
║  Add:  python admin.py                                          ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import importlib.util

# Project root is the folder that contains this file.
# Add it to sys.path so all modules (exercise_base, landmarks, etc.) resolve
# correctly regardless of where the user invokes python from.
PROJECT_ROOT  = os.path.dirname(os.path.abspath(__file__))
EXERCISES_DIR = os.path.join(PROJECT_ROOT, "exercises")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def discover_exercises() -> list:
    """
    Scan exercises/ and return [(display_name, filepath)] for every
    .py file that exposes a get_definition() function.
    Silently skips files that fail to import.
    """
    found = []
    if not os.path.isdir(EXERCISES_DIR):
        return found

    for fname in sorted(os.listdir(EXERCISES_DIR)):
        if not fname.endswith(".py") or fname.startswith("_"):
            continue
        fpath = os.path.join(EXERCISES_DIR, fname)
        try:
            spec   = importlib.util.spec_from_file_location("_probe_" + fname, fpath)
            module = importlib.util.module_from_spec(spec)
            # Prevent the if __name__ == "__main__" block from running
            module.__name__ = "_probe_" + fname
            spec.loader.exec_module(module)
            if hasattr(module, "get_definition"):
                found.append((module.get_definition().name, fpath))
        except Exception as e:
            import traceback
            print(f"  ⚠  Could not load {fname}: {e}")
            traceback.print_exc()

    return found


def main():
    exercises = discover_exercises()

    print("\n" + "═" * 50)
    print("  XAI Physiotherapy")
    print("═" * 50)

    if not exercises:
        print("  ⚠  No exercise files found in exercises/")
        print("  Run:  python admin.py  to add one.")
        print("═" * 50 + "\n")
        sys.exit(0)

    for i, (name, _) in enumerate(exercises):
        print(f"  {i + 1}. {name}")
    print(f"  {len(exercises) + 1}. Add new exercise (admin)")
    print("═" * 50)

    while True:
        raw = input("  Enter number: ").strip()
        try:
            choice = int(raw)
        except ValueError:
            print("  ⚠  Enter a number.")
            continue

        if choice == len(exercises) + 1:
            admin_path = os.path.join(PROJECT_ROOT, "admin.py")
            os.execv(sys.executable, [sys.executable, admin_path])

        elif 1 <= choice <= len(exercises):
            _, fpath = exercises[choice - 1]
            spec   = importlib.util.spec_from_file_location("selected_exercise", fpath)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            from exercise_runner import run_exercise
            run_exercise(module.get_definition())
            break

        else:
            print(f"  ⚠  Enter a number between 1 and {len(exercises) + 1}.")


if __name__ == "__main__":
    main()
