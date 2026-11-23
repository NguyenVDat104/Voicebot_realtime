@echo off
chcp 65001 >nul
title Voicebot Docker Launcher
echo ==========================================================
echo   KHỞI ĐỘNG VIETNAMESE VOICEBOT BẰNG DOCKER TRÊN DESKTOP
echo ==========================================================

:: 1. Kiểm tra Docker daemon có đang chạy không
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo [LỖI] Docker Desktop chưa được bật!
    echo Vui lòng mở ứng dụng Docker Desktop trên Windows rồi chạy lại file này.
    pause
    exit /b 1
)

:: 2. Build và khởi chạy container
echo [1/2] Đang build và khởi động Voicebot Container qua Docker Compose...
docker compose up -d --build

if %errorlevel% neq 0 (
    echo [LỖI] Quá trình khởi động Docker thất bại. Vui lòng kiểm tra log ở trên.
    pause
    exit /b 1
)

:: 3. Đợi vài giây để server sẵn sàng rồi tự mở trình duyệt
echo [2/2] Container đã chạy ngầm thành công!
echo Đang mở trình duyệt giao diện Voicebot tại http://localhost:8000...
timeout /t 3 /nobreak >nul
start http://localhost:8000

echo ==========================================================
echo   HỆ THỐNG VOICEBOT ĐÃ SẴN SÀNG TRÊN DOCKER!
echo   - Giao diện Voicebot: http://localhost:8000
echo   - Kiểm tra log container: docker logs -f voicebot-desktop
echo   - Dừng container:         docker compose down
echo ==========================================================
pause
