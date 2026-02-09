@echo off
echo ========================================
echo    StatMaple - Quick Start Script
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo [1/4] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        echo Please make sure Python 3.8+ is installed
        pause
        exit /b 1
    )
    echo ✓ Virtual environment created
    echo.
) else (
    echo ✓ Virtual environment already exists
    echo.
)

REM Activate virtual environment
echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo ✓ Virtual environment activated
echo.

REM Install requirements
echo [3/4] Installing dependencies...
echo This may take a few minutes on first run...
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo ✓ Dependencies installed
echo.

echo ========================================
echo    Starting StatMaple...
echo ========================================
echo.
echo The application will open in your browser at:
echo http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

REM Start Streamlit
streamlit run app.py

pause
