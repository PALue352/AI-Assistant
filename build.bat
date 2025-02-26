:: AI_Assistant/build.bat
@echo off
echo Building AI Assistant package...
set "PYTHON_DIR=D:\Python312"  :: Matches your install choice—adjust if changed
if not exist "%PYTHON_DIR%\python.exe" (
    echo Python not found at %PYTHON_DIR%. Please run the installer first.
    pause
    exit /b 1
)
"%PYTHON_DIR%\python.exe" setup.py sdist bdist_wheel
if %ERRORLEVEL% NEQ 0 (
    echo Failed to build package. Check errors.
    pause
    exit /b %ERRORLEVEL%
)
echo Package built successfully in dist/
pause
exit /b 0