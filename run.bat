@echo off
chcp 65001 >nul
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo 파이썬이 없습니다. python.org 에서 Python 3.11 이상을 설치하세요. 설치할 때 "Add python.exe to PATH"를 꼭 체크하세요.
  pause
  exit /b 1
)

rem .env 파일이 있으면 키를 불러옴 (# 으로 시작하는 줄과 빈 값은 건너뜀)
if exist ".env" (
  for /f "usebackq eol=# tokens=1,* delims==" %%a in (".env") do if not "%%b"=="" set "%%a=%%b"
)

echo [1/2] 필요한 프로그램 설치 확인 중... (처음엔 몇 분 걸릴 수 있어요)
python -m pip install -q -r requirements.txt
if errorlevel 1 (
  echo 설치에 실패했습니다. 인터넷 연결을 확인하세요.
  pause
  exit /b 1
)

echo [2/2] 화면 여는 중... 브라우저가 자동으로 열립니다. 끄려면 이 창을 닫으세요.
echo 첫 화면은 진단 계산 때문에 20초 정도 걸립니다.
python -m streamlit run app.py
pause
