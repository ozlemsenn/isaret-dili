@echo off
REM Isaret demosu - KURULUM (sadece bir kez cift tikla)
cd /d "%~dp0"
set "PY="

echo [1/3] Python araniyor...
call :dene py -3.11
call :dene py -3.12
call :dene py -3.13
call :dene py -3.10
call :dene python

if not defined PY (
    echo Uygun Python bulunamadi. Python 3.11 kuruluyor, birkac dakika surebilir...
    winget install -e --id Python.Python.3.11 --scope user --accept-package-agreements --accept-source-agreements
    if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set PY="%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
)
if not defined PY (
    echo.
    echo Python otomatik kurulamadi.
    echo python.org adresinden Python 3.11 indir, kurarken "Add python.exe to PATH" kutusunu isaretle,
    echo sonra bu dosyaya tekrar cift tikla.
    pause
    exit /b 1
)
echo Kullanilan Python: %PY%

echo [2/3] Proje icin ayri bir Python ortami (venv) hazirlaniyor...
if not exist "venv\Scripts\python.exe" %PY% -m venv venv
if not exist "venv\Scripts\python.exe" (
    echo HATA: venv olusturulamadi. Bu pencerenin ekran goruntusunu al.
    pause
    exit /b 1
)

echo [3/3] Kutuphaneler kuruluyor, 5-10 dakika surebilir, cok yazi akacak, normal...
"venv\Scripts\python.exe" -m pip install --upgrade pip
"venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo HATA olustu. Bu pencerenin ekran goruntusunu al.
) else (
    echo.
    echo KURULUM BITTI. Artik baslat.bat dosyasina cift tiklayabilirsin.
)
pause
exit /b 0

:dene
if defined PY exit /b 0
%* -c "import sys; assert (3, 10) <= sys.version_info[:2] <= (3, 13)" >nul 2>&1
if not errorlevel 1 set "PY=%*"
exit /b 0
