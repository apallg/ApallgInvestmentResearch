@echo off
cd /d "%~dp0frontend"
echo [%date% %time%] V2 Frontend starting...
echo   Directory: %cd%
echo   Node:
where node 2>&1
echo   NPM:
where npm 2>&1
echo.
echo Starting Vite...
call npm run dev
echo.
echo [%date% %time%] Frontend stopped (exit code: %ERRORLEVEL%)
pause
