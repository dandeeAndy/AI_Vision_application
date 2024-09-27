import numpy as np

data_A = [[1,2],[3,4],[5,6]] # 데이터 생성
np.save('data', data_A) # 데이터를 파일로 저장(파일명 : data_A.npy)
del data_A # 저장된 파일이 잘 불러와 지는지 확인하기 위해 삭제

data_A2 = np.load('data.npy') # 저장된 파일 불러오기