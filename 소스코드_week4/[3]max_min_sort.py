import numpy as np

data = [20, 10, 50, 60, 70, 90, 110, 40, 45, 65, 80, 15]

max_val = np.max(data) # 최댓값
max_idx = np.argmax(data) # 최댓값에 해당하는 인덱스 찾기

min_val = np.min(data) # 최솟값
min_idx = np.argmin(data) # 최솟값에 해당하는 인덱스 찾기

sor = np.sort(data) # 오름차순 정렬
sor_index = np.argsort(data) # 오름차순 결과에 대한 인덱스 찾기
sor_reverse = np.sort(sor)[::-1] # 내림차순 정렬

