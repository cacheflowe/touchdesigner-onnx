@REM This script installs Python modules for TouchDesigner using 
@REM the Python executable bundled with TouchDesigner for compatibility.
@REM Modules are installed to a local directory to avoid conflicts with the system Python.

set PYTHON_PATH=C:\Program Files\Derivative\TouchDesigner\bin\python.exe
"%PYTHON_PATH%" -m pip install -r requirements.txt --target="../_local_modules" 
pause
