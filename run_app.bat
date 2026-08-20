@echo off
cd /d "%~dp0"

title Customer RFM-T and Customer Intelligence Engine

echo =======================================================================
echo   Customer RFM-T and Customer Intelligence AI Platform Launcher
echo =======================================================================
echo.

REM 1. Identify Python executable
set "PYTHON_EXE="

python --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set "PYTHON_EXE=python"
    goto :PYTHON_OK
)

py -3 --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set "PYTHON_EXE=py -3"
    goto :PYTHON_OK
)

if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    goto :PYTHON_OK
)

if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" (
    set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    goto :PYTHON_OK
)

if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    goto :PYTHON_OK
)

echo [ERROR] Python was not found in PATH or standard install directories.
echo Please ensure Python 3.10+ is installed from python.org and added to PATH.
echo.
pause
exit /b 1

:PYTHON_OK
echo [INFO] Python detected: %PYTHON_EXE%

REM 2. Activate virtual environment if present
if exist "venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment venv
    call venv\Scripts\activate.bat
)
if exist ".venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment .venv
    call .venv\Scripts\activate.bat
)

REM 3. Ensure dataset is present
if not exist "data\ecommerce_transactions.csv" (
    echo [INFO] Generating 24-month enterprise transaction dataset...
    %PYTHON_EXE% generate_data.py
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Dataset generation failed.
        pause
        exit /b 1
    )
)

REM 4. Launch Streamlit Dashboard
echo.
echo [INFO] Starting Streamlit Customer Intelligence Engine...
echo [INFO] Your default web browser will open automatically.
echo [INFO] To stop the application, close this window or press Ctrl+C.
echo.

%PYTHON_EXE% -m streamlit run app.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Streamlit exited with an error.
    pause
)

pause
