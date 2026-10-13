@echo off
title Synapta Worker - Instalador
echo.
echo  ================================================
echo   Synapta Worker - Instalacion para Windows
echo  ================================================
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
echo.
echo  Cierra esta ventana cuando termines de leer el mensaje.
pause
