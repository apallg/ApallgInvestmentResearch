import subprocess, os
env = dict(os.environ)
log = open("logs/frontend_dev.log", "ab")
p = subprocess.Popen(
    ["cmd", "/c", "npm run dev"],
    stdout=log, stderr=subprocess.STDOUT, cwd=os.path.join(os.getcwd(), "frontend"), env=env,
    creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
)
print("PID", p.pid)
