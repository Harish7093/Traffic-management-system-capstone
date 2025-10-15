@echo off
echo Installing Python dependencies...
pip install -r requirements.txt

if %ERRORLEVEL% NEQ 0 (
    echo Error installing dependencies. Please check your Python and pip installation.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo All dependencies installed successfully!
echo.
echo To run the application, use: streamlit run app.py
pause
