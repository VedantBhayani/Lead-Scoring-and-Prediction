@echo off
echo ==== Lead Scoring and Prediction System ====
echo.
echo Installing backend requirements...
pip install -r MAIN/backend_requirements.txt
echo.
echo Starting the system...
python run_system.py
pause 