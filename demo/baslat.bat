@echo off
REM Isaret demosu - BASLAT (her seferinde cift tikla)
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo Kurulum yapilmamis. Once kurulum.bat dosyasina cift tikla.
    pause
    exit /b 1
)

echo Kamera aciliyor... Cikmak icin kamera penceresinde q tusuna bas.
"venv\Scripts\python.exe" app.py
pause
