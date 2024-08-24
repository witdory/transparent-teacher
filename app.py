import cv2 as cv
import numpy as np

# Background Subtractor 객체 생성
backSub = cv.createBackgroundSubtractorKNN()

# 비디오 캡처 객체 생성
capture = cv.VideoCapture('video3.mp4')

# 모폴로지 변형을 위한 커널 생성
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))

# 첫 번째 프레임을 배경으로 초기화
ret, frame = capture.read()
background = frame.copy()

# 배경 업데이트 비율
alpha = 0.05
update_frequency = 1  # 프레임마다 배경 업데이트

frame_count = 0

while True:
    ret, frame = capture.read()
    if frame is None:
        break

    frame_count += 1

    # 배경 모델 업데이트 및 포그라운드 마스크 생성
    fgMask = backSub.apply(frame)

    # 모폴로지 변형을 통해 노이즈 제거
    fgMask_morph = cv.morphologyEx(fgMask, cv.MORPH_OPEN, kernel)

    # 프레임 번호를 현재 프레임에 표시
    cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
    cv.putText(frame, str(int(capture.get(cv.CAP_PROP_POS_FRAMES))), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    # 배경에서 사람 위치를 덮어씌움
    mask = fgMask_morph.astype(bool)
    frame[mask] = background[mask]

    # 주기적으로 배경 이미지 업데이트
    if frame_count % update_frequency == 0:
        background = cv.addWeighted(background, 1 - alpha, frame, alpha, 0)

    # 결과 프레임을 화면에 표시
    cv.imshow('Background Removed Video', frame)

    keyboard = cv.waitKey(1) & 0xFF
    if keyboard == 27:
        break

capture.release()
cv.destroyAllWindows()
