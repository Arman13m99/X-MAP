:: --- Windows batch file content (save as start.bat) ---
@echo off
echo Starting Tapsi Food Map Dashboard...

:: Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Install/update dependencies
echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

:: Run the production server
echo Starting production server...
python run_production.py
