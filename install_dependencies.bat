@echo off
echo Installing dependencies for AI Assistant...
set "PYTHON_DIR=D:\Python312"
if not exist "%PYTHON_DIR%\python.exe" (
    echo Python not found at %PYTHON_DIR%. Please run the installer to install Python first.
    pause
    exit /b 1
)

:: Check for Microsoft C++ Build Tools
echo Checking for Microsoft C++ Build Tools...
where cl >nul 2>&1
if %errorlevel% neq 0 (
    echo Microsoft C++ Build Tools not found. Installing...
    powershell -Command "Invoke-WebRequest -Uri 'https://aka.ms/vs/17/release/vs_buildtools.exe' -OutFile 'vs_buildtools.exe'"
    start /wait vs_buildtools.exe --add Microsoft.VisualStudio.Workload.VCTools --quiet
    del vs_buildtools.exe
    echo C++ Build Tools installed. Please restart your computer and rerun this script.
    pause
    exit /b 1
)

:: Check for GPU and install correct PyTorch version
echo Checking for GPU/CUDA support...
"%PYTHON_DIR%\python.exe" -c "import torch; print(torch.cuda.is_available())" > temp.txt
set /p CUDA_AVAILABLE=<temp.txt
del temp.txt

if "%CUDA_AVAILABLE%"=="True" (
    echo GPU detected. Installing PyTorch with CUDA support...
    "%PYTHON_DIR%\python.exe" -m pip install torch==2.6.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
) else (
    echo No GPU detected. Installing PyTorch CPU-only version...
    "%PYTHON_DIR%\python.exe" -m pip install torch==2.6.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
)

:: Install other dependencies (offline or online)
echo Installing Python dependencies...
"%PYTHON_DIR%\python.exe" -m pip install --upgrade pip
"%PYTHON_DIR%\python.exe" -m pip install -r "%~dp0requirements.txt"
if %errorlevel% neq 0 (
    echo Failed to install dependencies. Check logs and try again.
    pause
    exit /b %errorlevel%
)

:: Pre-download Hugging Face models (ensure internet connection for first run)
echo Pre-downloading Hugging Face models...
set "MODEL_CACHE=%~dp0model_cache"
mkdir "%MODEL_CACHE%"
"%PYTHON_DIR%\python.exe" -m huggingface_hub.cli.download "Qwen/Qwen-7B" --local-dir "%MODEL_CACHE%\Qwen-7B"
"%PYTHON_DIR%\python.exe" -m huggingface_hub.cli.download "nlpconnect/vit-gpt2-image-captioning" --local-dir "%MODEL_CACHE%\vit-gpt2"
"%PYTHON_DIR%\python.exe" -m huggingface_hub.cli.download "microsoft/phi-3-mini-4k-instruct" --local-dir "%MODEL_CACHE%\phi-3-mini"

:: Pre-download EasyOCR models
echo Pre-downloading EasyOCR models...
"%PYTHON_DIR%\python.exe" -c "import easyocr; reader = easyocr.Reader(['en']);"

:: Install and configure Ollama
echo Installing and configuring Ollama...
if not exist "%~dp0Ollama\ollama-windows.exe" (
    echo Downloading Ollama Windows installer...
    powershell -Command "Invoke-WebRequest -Uri 'https://ollama.com/download/windows' -OutFile '%~dp0Ollama\ollama-windows.exe'"
)
start /wait %~dp0Ollama\ollama-windows.exe --silent
"%PYTHON_DIR%\python.exe" -m ollama serve & start /b "" "%PYTHON_DIR%\python.exe" -m ollama pull qwen:latest

:: Install voice dependencies
echo Installing voice recognition dependencies...
"%PYTHON_DIR%\python.exe" -m pip install speechrecognition pyaudio gtts pyttsx3

echo Dependencies installed successfully.
pause
exit /b 0