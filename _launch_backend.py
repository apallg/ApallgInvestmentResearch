import subprocess, os, sys
env = dict(os.environ)
env["OPENBLAS_NUM_THREADS"] = "1"
env["OMP_NUM_THREADS"] = "8"
env["MKL_NUM_THREADS"] = "1"
env["PYTHONUNBUFFERED"] = "1"
os.makedirs("logs", exist_ok=True)
log = open("logs/backend_restart.log", "ab")
p = subprocess.Popen(
    [sys.executable, "-u", "-m", "backend.main"],
    stdout=log, stderr=subprocess.STDOUT, cwd=os.getcwd(), env=env,
    creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
)
print("PID", p.pid)
