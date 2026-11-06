@echo off
cd /d "%~dp0."
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=8
set MKL_NUM_THREADS=1
set PYTHONUNBUFFERED=1

:restart
echo [%date% %time%] Starting backend (port 8001)...
"C:\Users\zhj\AppData\Local\Programs\Python\Python311\python.exe" -u -m backend.main
echo.
echo [%date% %time%] Backend stopped (code %ERRORLEVEL%). Restarting in 3s...
timeout /t 3 /nobreak >nul
goto restart
