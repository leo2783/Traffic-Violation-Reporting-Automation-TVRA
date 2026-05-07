@echo off
setlocal
cd /d "%~dp0"

python -m PyInstaller --version >nul 2>nul
if errorlevel 1 (
    echo PyInstaller is not installed in the current Python environment.
    echo Install it with: python -m pip install pyinstaller
    exit /b 1
)

python -m PyInstaller --clean --noconfirm file_compare_tool.spec
echo Built exe: %cd%\dist\TVRAFileCompareTool.exe
endlocal