import cv2
import torch
import time
import numpy as np
import os

# YOLOv5 모델 로드 (torch hub 사용)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.conf = 0.4  # Confidence threshold (기본값보다 약간 높게 설정)
model.iou = 0.45  # IOU threshold (기본값: 0.45)

# 입력 비디오 파일 경로 설정
input_video_path = r'C:\vision_team_2\test.mp4'
output_video_path = os.path.join(os.path.dirname(input_video_path), 'output_video.mp4')

# 비디오 파일 로드
cap = cv2.VideoCapture(input_video_path)

# 배경을 저장할 공간 초기화
background_frame = None

# 프레임 크기 정보 얻기
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 비디오 저장 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (frame_width, frame_height))

# FPS 계산용 변수 초기화
prev_frame_time = 0
new_frame_time = 0

# 이전 프레임 초기화
previous_frame = None

# 바운딩 박스 확장 비율 설정 (예: 30% 확장)
expand_ratio = 0.3

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # YOLOv5를 사용하여 사람 탐지
    results = model(frame)
    # 결과에서 사람만 필터링
    person_detections = results.xyxy[0].cpu().numpy()
    person_detections = [detection for detection in person_detections if detection[5] == 0]  # 클래스 ID 0은 'person' 클래스
    
    # 마스크 초기화
    mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    
    # 사람의 위치에 해당하는 영역을 마스크로 처리
    for detection in person_detections:
        x1, y1, x2, y2 = map(int, detection[:4])  # Bounding box 좌표
        
        # 바운딩 박스를 확장
        width = x2 - x1
        height = y2 - y1
        x1 = max(0, x1 - int(width * expand_ratio))
        y1 = max(0, y1 - int(height * expand_ratio))
        x2 = min(frame_width, x2 + int(width * expand_ratio))
        y2 = min(frame_height, y2 + int(height * expand_ratio))
        
        # 확장된 사람 영역을 흰색(255)으로 마스킹
        mask[y1:y2, x1:x2] = 255  

    # 프레임 간 차이를 사용하여 새로운 움직임 감지
    if previous_frame is not None:
        # 프레임 간 차이 계산
        frame_diff = cv2.absdiff(cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        _, motion_mask = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
        
        # 움직임이 감지된 영역을 마스크에 추가
        mask = cv2.bitwise_or(mask, motion_mask)
    
    # 사람이 있는 영역을 제외한 나머지 부분만 배경으로 저장
    if background_frame is None:
        background_frame = frame.copy()
    else:
        # Numpy 배열 슬라이싱을 이용한 빠른 배경 업데이트
        update_mask = cv2.bitwise_not(mask)
        background_frame[update_mask == 255] = frame[update_mask == 255]

    # 사람이 있는 경우 배경으로 대체
    for detection in person_detections:
        x1, y1, x2, y2 = map(int, detection[:4])  # Bounding box 좌표
        
        # 바운딩 박스를 확장
        width = x2 - x1
        height = y2 - y1
        x1 = max(0, x1 - int(width * expand_ratio))
        y1 = max(0, y1 - int(height * expand_ratio))
        x2 = min(frame_width, x2 + int(width * expand_ratio))
        y2 = min(frame_height, y2 + int(height * expand_ratio))
        
        # 사람 부분을 배경으로 대체
        frame[y1:y2, x1:x2] = background_frame[y1:y2, x1:x2]  
    
    # 이전 프레임 업데이트
    previous_frame = frame.copy()

    # 프레임률(FPS) 계산
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    
    # FPS를 프레임에 표시
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 결과 프레임을 출력
    cv2.imshow('Output', frame)
    out.write(frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
out.release()
cv2.destroyAllWindows()
