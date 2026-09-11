@echo off
chcp 65001 >nul
title Voicebot System Launcher
echo =======================================================
echo   ĐANG KHỞI ĐỘNG HỆ THỐNG VIETNAMESE VOICEBOT FULL-DUPLEX
echo =======================================================

:: 1. Kiểm tra và khởi động n8n ở cửa sổ riêng (nếu cần dùng workflow AI)
echo [1/2] Đang kiểm tra và khởi động n8n Workflow...
start "n8n Workflow Server" cmd /c "npx n8n"

:: 2. Khởi động FastAPI Audio Gateway với môi trường Anaconda chatbot
echo [2/2] Đang khởi động FastAPI Audio Gateway (Port 8000)...
start "FastAPI Audio Server" cmd /k "cd /d C:\Speech&Audio && C:\Anaconda\envs\chatbot\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload"

:: 3. Đợi 3 giây để server khởi động rồi tự động mở trình duyệt
echo Đang mở trình duyệt giao diện Voicebot...
timeout /t 3 /nobreak >nul
start http://127.0.0.1:8000

echo =======================================================
echo   ĐÃ BẬT XONG TOÀN BỘ HỆ THỐNG!
echo   - Giao diện Voicebot: http://127.0.0.1:8000
echo   - Quản trị n8n:       http://localhost:5678
echo =======================================================
pause
