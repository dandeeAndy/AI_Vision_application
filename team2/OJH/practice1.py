import os

# 현재 작업 디렉토리 출력
current_working_dir = os.getcwd()
print("현재 작업 디렉토리:", current_working_dir)

# 현재 스크립트 파일의 절대 경로
script_path = os.path.abspath(__file__)
print("스크립트 파일 경로:", script_path)

# 스크립트가 있는 디렉토리 경로
script_dir = os.path.dirname(script_path)
print("스크립트 디렉토리:", script_dir)