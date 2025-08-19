"""
Watches backend Python files and runs pytest automatically when they change.
"""
import sys
import subprocess
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

WATCH_DIRS = [
    Path(__file__).resolve().parent.parent / "python" / "core",
    Path(__file__).resolve().parent.parent / "python" / "api",
]

class TestRunnerHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix == ".py":
            print(f"\n🔄 Detected change in {path}, running pytest...\n")
            run_tests()

def run_tests():
    try:
        subprocess.run(
            ["pytest", "-q", "--disable-warnings", "--maxfail=1"],
            check=False
        )
    except Exception as e:
        print(f"❌ Error running pytest: {e}")

if __name__ == "__main__":
    observers = []
    for d in WATCH_DIRS:
        if not d.exists():
            continue
        observer = Observer()
        observer.schedule(TestRunnerHandler(), str(d), recursive=True)
        observer.start()
        observers.append(observer)
        print(f"👀 Watching {d} for changes...")

    try:
        while True:
            pass
    except KeyboardInterrupt:
        for obs in observers:
            obs.stop()
        for obs in observers:
            obs.join()