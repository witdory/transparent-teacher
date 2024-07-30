import cv2
import numpy as np
import os

# YOLOv3-tiny 가중치 파일과 설정 파일의 정확한 경로 지정
weights_path = 'C:/vision_team_2/yolov3-tiny.weights'
cfg_path = 'C:/vision_team_2/yolov3-tiny.cfg'
names_path = 'C:/vision_team_2/coco.names'
video_path = 'C:/vision_team_2/test.mp4'

# 경로 확인
if not os.path.exists(weights_path):
    print(f"Weights file not found: {weights_path}")
if not os.path.exists(cfg_path):
    print(f"Config file not found: {cfg_path}")
if not os.path.exists(names_path):
    print(f"Names file not found: {names_path}")
if not os.path.exists(video_path):
    print(f"Video file not found: {video_path}")

# YOLO 가중치 파일과 설정 파일 로드
net = cv2.dnn.readNet(weights_path, cfg_path)

# 클래스 이름 로드
with open(names_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# 사람 클래스의 인덱스 찾기
person_idx = classes.index('person')

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(video_path)

# 배경 추출기 초기화
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

# 초기 프레임 수집을 위한 카운터
initial_frame_count = 30
frame_count = 0

# 추적기 초기화
tracker_initialized = False
tracker = cv2.legacy.TrackerKCF_create()
bbox = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    if not tracker_initialized or frame_count % 5 == 0:  # 5 프레임마다 YOLO로 객체 재탐지
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
        net.setInput(blob)

        layer_names = net.getLayerNames()
        try:
            unconnected_out_layers = net.getUnconnectedOutLayers()
            output_layers = [layer_names[i - 1] for i in unconnected_out_layers.flatten()]
        except:
            print("Failed to get output layers")
            break

        outs = net.forward(output_layers)

        # 결과 처리
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and class_id == person_idx:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append((x, y, w, h))

        if boxes:
            bbox = tuple(boxes[0])
            tracker = cv2.legacy.TrackerKCF_create()
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

    # 결과 출력
    cv2.imshow('Processed Frame', frame)

    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
