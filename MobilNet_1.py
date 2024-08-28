import cv2
import numpy as np
import os

# MobileNet SSD 모델과 설정 파일의 정확한 경로 지정
prototxt_path = 'deploy.prototxt'
weights_path = 'mobilenet_iter_73000.caffemodel'
video_path = 'test.mp4'

# 경로 확인
if not os.path.exists(prototxt_path):
    print(f"Prototxt file not found: {prototxt_path}")
if not os.path.exists(weights_path):
    print(f"Weights file not found: {weights_path}")
if not os.path.exists(video_path):
    print(f"Video file not found: {video_path}")

# MobileNet 모델 로드
net = cv2.dnn.readNetFromCaffe(prototxt_path, weights_path)

# 클래스 이름 설정 (MobileNet SSD는 COCO 데이터셋의 클래스들을 지원)
classes = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

# 사람 클래스의 인덱스 찾기
person_idx = classes.index('person')

cap = cv2.VideoCapture(video_path)

# 비디오 저장 객체 생성
output_path = 'output_mobilenet.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# 배경 추출기 초기화
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

# 초기 프레임 수집을 위한 카운터
initial_frame_count = 30
frame_count = 0

# 추적기 초기화
tracker_initialized = False
tracker = cv2.TrackerKCF_create()
bbox = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    if not tracker_initialized or frame_count % 5 == 0:  # 5 프레임마다 MobileNet으로 객체 재탐지
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
        net.setInput(blob)
        detections = net.forward()

        # 결과 처리
        boxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            class_id = int(detections[0, 0, i, 1])
            if confidence > 0.5 and class_id == person_idx:
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (x, y, w, h) = box.astype("int")
                boxes.append((x, y, w - x, h - y))

        if boxes:
            bbox = tuple(boxes[0])
            tracker = cv2.TrackerKCF_create()
            tracker.init(frame, bbox)
            tracker_initialized = True

    if tracker_initialized:
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in bbox]
        else:
            tracker_initialized = False

    # 사람 영역 마스크 생성
    mask = np.ones((height, width), dtype=np.uint8) * 255
    if tracker_initialized and bbox is not None:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(mask, (x, y), (x + w, y + h), 0, -1)

    if frame_count < initial_frame_count:
        # 초기 프레임 동안 배경 추출을 위해 사람 영역 제외
        fgmask = fgbg.apply(frame)
        fgmask[mask == 0] = 0
    else:
        # 초기 프레임 이후 배경 업데이트
        fgmask = fgbg.apply(frame, learningRate=0.01)
        fgmask[mask == 0] = 0

    background = fgbg.getBackgroundImage()

    if background is not None and tracker_initialized and bbox is not None:
        # 사람 영역을 배경으로 치환
        x, y, w, h = [int(v) for v in bbox]
        frame[y:y+h, x:x+w] = background[y:y+h, x:x+w]

    # 프레임 번호를 현재 프레임에 표시
    cv2.putText(frame, f'Frame: {frame_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # 결과 출력
    cv2.imshow('Processed Frame', frame)

    # 프레임 저장
    out_video.write(frame)

    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out_video.release()
cv2.destroyAllWindows()
