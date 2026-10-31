"""Keep system awake (process-level) and run the 8G training."""
import ctypes
import subprocess
import sys

# ES_CONTINUOUS | ES_SYSTEM_REQUIRED - holds while this thread is alive
ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)

args = [
    sys.executable, "-u", "-m", "supermimic.scripts.train",
    "--run-name", "lafan_8g_v1",
    "--env.num-envs", "512",
    "--env.anchor-obs",
    "--policy-trainer.batch-size", "512",
    "--policy-trainer.horizon", "16",
    "--schedule.max-hours", "36",
    "--schedule.checkpoint-interval", "2000",
]
sys.exit(subprocess.call(args))
