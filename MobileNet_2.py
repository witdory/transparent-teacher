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

# 칠판 영역 선택용 변수
roi_selected = False
roi = None
roi_start = None

# 추적기 초기화
tracker_initialized = False
tracker = cv2.TrackerKCF_create()
bbox = None

# 마우스 콜백 함수 정의
def select_roi(event, x, y, flags, param):
    global roi, roi_selected, roi_start, temp_frame
    
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_start = (x, y)
        roi = None
    
    elif event == cv2.EVENT_MOUSEMOVE and roi_start:
        temp_frame = frame.copy()
        cv2.rectangle(temp_frame, roi_start, (x, y), (0, 0, 255), 2)
    
    elif event == cv2.EVENT_LBUTTONUP:
        roi = [roi_start, (x, y)]
        roi_selected = True
        x0, y0 = roi[0]
        x1, y1 = roi[1]

# 첫 프레임에서 칠판 영역 선택
ret, frame = cap.read()
if not ret:
    print("Failed to read the video.")
    cap.release()
    out_video.release()
    cv2.destroyAllWindows()
    exit()

# 사용자가 칠판 영역을 선택할 수 있도록 창 열기
cv2.namedWindow('Select ROI')
cv2.setMouseCallback('Select ROI', select_roi)
temp_frame = frame.copy()

while not roi_selected:
    cv2.imshow('Select ROI', temp_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        out_video.release()
        cv2.destroyAllWindows()
        exit()

# 칠판 영역이 선택된 후 빨간색 사각형을 표시
x0, y0 = roi[0]
x1, y1 = roi[1]
cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 2)
cv2.destroyWindow('Select ROI')

# 비디오 프레임 처리 시작
background = None  # background 변수 초기화

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    # 선택된 칠판 영역을 계속 표시
    if roi_selected:
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 2)

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
            # 칠판 영역과 강의자 영역이 겹치면 강의자 영역을 배경으로 덮음
            if (x < x1 and x + w > x0) and (y < y1 and y + h > y0):
                if background is not None:
                    frame[y:y+h, x:x+w] = background[y:y+h, x:x+w]  # 강의자 영역을 배경으로 덮음
        else:
            tracker_initialized = False

    # 강의자 영역 밖의 배경 업데이트
    if frame_count < initial_frame_count:
        fgmask = fgbg.apply(frame)
        fgmask[y:y+h, x:x+w] = 0  # 강의자 영역 업데이트 제외
    else:
        fgmask = fgbg.apply(frame, learningRate=0.01)
        fgmask[y:y+h, x:x+w] = 0  # 강의자 영역 업데이트 제외

    background = fgbg.getBackgroundImage()

    if background is not None:
        # 강의자 영역 외의 배경 업데이트
        mask_background = np.ones_like(frame, dtype=np.uint8) * 255
        mask_background[y:y+h, x:x+w] = frame[y:y+h, x:x+w]
        frame = np.where(mask_background == 255, background, frame)

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