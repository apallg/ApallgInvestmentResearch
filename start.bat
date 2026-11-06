@echo off
setlocal enabledelayedexpansion
title Apallg TouYan V2 - Start
cd /d "%~dp0."

echo ================================
echo   Apallg TouYan V2
echo ================================

for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8001.*LISTENING" 2^>nul') do (
    taskkill /f /pid %%a >nul 2>&1
    echo [OK] Killed backend :8001 PID=%%a
)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":517[4-9].*LISTENING" 2^>nul') do (
    taskkill /f /pid %%a >nul 2>&1
    echo [OK] Killed frontend :5174 PID=%%a
)
powershell -Command "Get-NetTCPConnection -LocalPort 5174,5175,5176 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }" >nul 2>&1
timeout /t 2 /nobreak >nul

set PYTHON_EXE=C:\Users\zhj\AppData\Local\Programs\Python\Python311\python.exe
if exist ".env" (
    for /f "usebackq tokens=1,2 delims==" %%a in (".env") do (
        if "%%a"=="PYTHON_EXE" set "PYTHON_EXE=%%b"
    )
)

echo.
echo   Backend  : http://localhost:8001
echo   Frontend : http://localhost:5174
echo.

echo Starting Backend...
start "V2 Backend" _run_backend.bat

echo Starting Frontend...
start "V2 Frontend" cmd /c "cd frontend && npm run dev"

echo.
echo   Both windows opened. Backend takes ~20s to init.
echo   Close this window freely.
timeout /t 3 /nobreak >nul
endlocal
