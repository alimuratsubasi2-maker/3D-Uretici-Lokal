@echo off
cd /d "%~dp0"
echo 3D Model Uretici baslatiliyor...
start "3D Model Uretici - Sunucu (bu pencereyi kapatma)" /min cmd /c python app.py
echo Model yukleniyor, lutfen bekleyin...

:bekle
timeout /t 2 /nobreak >nul
curl -s -o nul -w "%%{http_code}" http://127.0.0.1:5050/ 2>nul | findstr "200" >nul
if errorlevel 1 goto bekle

start "" http://127.0.0.1:5050
echo Tarayici acildi. Sunucu penceresini kapatirsan uygulama durur.
pause
