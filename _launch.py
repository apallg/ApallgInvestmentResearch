import subprocess, sys, os
script = sys.argv[1]
role = sys.argv[2]
folds = sys.argv[3]
extra = sys.argv[4:]
logpath = os.path.join("data", "experiments", f"exp_{role}_{folds}folds.log")
env = dict(os.environ)
env["OMP_NUM_THREADS"] = "8"
env["OPENBLAS_NUM_THREADS"] = "1"
env["MKL_NUM_THREADS"] = "1"
with open(logpath, "wb") as log:
    p = subprocess.Popen(
        [sys.executable, script, "--role", role, "--folds", folds, *extra],
        stdout=log, stderr=subprocess.STDOUT, cwd=os.getcwd(), env=env,
        creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
    )
print("PID", p.pid, "LOG", logpath)
