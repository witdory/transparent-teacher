import cv2
import torch
import time
import numpy as np

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.conf = 0.4  # 기본 confidence threshold
model.iou = 0.45  # IOU threshold

# 카메라 또는 비디오 입력
video_source = 0  # 실시간 카메라: 0, 비디오 파일 경로: 'path_to_video.mp4'
cap = cv2.VideoCapture(video_source)

# 배경 저장 변수 및 프레임 크기
background_frame = None
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 바운딩 박스 확장 비율
expand_ratio = 0.3

# FPS 계산 변수
prev_frame_time = 0

# 프레임 차이를 저장할 변수
previous_frame = None

# IOU 계산 함수
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0

# 중복 바운딩 박스 제거
def non_max_suppression(detections, iou_threshold=0.5):
    detections = sorted(detections, key=lambda x: x[4], reverse=True)
    filtered_detections = []

    while detections:
        chosen_box = detections.pop(0)
        filtered_detections.append(chosen_box)
        detections = [box for box in detections if calculate_iou(chosen_box[:4], box[:4]) < iou_threshold]

    return filtered_detections

# 비디오 처리 루프
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 원본 프레임 저장
    original_frame = frame.copy()

    # YOLO를 사용한 사람 탐지
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()
    
    # 사람 탐지만 필터링
    person_detections = [d for d in detections if d[5] == 0]
    person_detections = non_max_suppression(person_detections, iou_threshold=0.5)

    # 사람 영역 마스크 생성
    mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    for detection in person_detections:
        x1, y1, x2, y2 = map(int, detection[:4])
        width = x2 - x1
        height = y2 - y1
        x1 = max(0, x1 - int(width * expand_ratio))
        y1 = max(0, y1 - int(height * expand_ratio))
        x2 = min(frame_width, x2 + int(width * expand_ratio))
        y2 = min(frame_height, y2 + int(height * expand_ratio))

        # 사람 영역 마스킹
        mask[y1:y2, x1:x2] = 255

    # 프레임 간 움직임 감지
    if previous_frame is not None:
        frame_diff = cv2.absdiff(cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        _, motion_mask = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_or(mask, motion_mask)

    # 배경 업데이트
    if background_frame is None:
        background_frame = frame.copy()
    else:
        update_mask = cv2.bitwise_not(mask)
        motion_intensity = np.sum(motion_mask) / (frame_width * frame_height)

        # 움직임 강도가 5% 이상인 경우에만 배경 업데이트
        if motion_intensity > 0.05:
            background_frame[update_mask == 255] = frame[update_mask == 255]

    # 탐지된 사람 영역을 배경으로 교체
    for detection in person_detections:
        x1, y1, x2, y2 = map(int, detection[:4])
        width = x2 - x1
        height = y2 - y1
        x1 = max(0, x1 - int(width * expand_ratio))
        y1 = max(0, y1 - int(height * expand_ratio))
        x2 = min(frame_width, x2 + int(width * expand_ratio))
        y2 = min(frame_height, y2 + int(height * expand_ratio))

        frame[y1:y2, x1:x2] = background_frame[y1:y2, x1:x2]

    # 이전 프레임 저장
    previous_frame = frame.copy()

    # FPS 계산 및 화면에 표시
    new_frame_time = time.time()
    fps = int(1 / (new_frame_time - prev_frame_time))
    prev_frame_time = new_frame_time

    cv2.putText(frame, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 결과 프레임 출력
    cv2.imshow("Live Lecture Background Removal", frame)
    cv2.imshow("Original Video", original_frame)  # 원본 화면 출력

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
