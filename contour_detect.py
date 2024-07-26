import cv2 as cv
import numpy as np

# Background Subtractor 객체 생성 (history 파라미터 조정)
backSub = cv.createBackgroundSubtractorKNN(history=500, dist2Threshold=400.0, detectShadows=False)

# 비디오 캡처 객체 생성
capture = cv.VideoCapture('video3.mp4')

# 모폴로지 변형을 위한 커널 생성
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))

# 첫 번째 프레임을 배경으로 초기화
ret, frame = capture.read()
background = frame.copy()

# 이전 프레임 저장
prev_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

# 프레임 간 차이 계산을 위한 임계값
motion_threshold = 5  # 임계값을 낮춰 작은 움직임도 감지

while True:
    ret, frame = capture.read()
    if frame is None:
        break

    # 현재 프레임을 그레이스케일로 변환
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # 프레임 간 차이 계산
    frame_diff = cv.absdiff(prev_frame, gray_frame)
    motion_mask = cv.threshold(frame_diff, motion_threshold, 255, cv.THRESH_BINARY)[1]

    # 모션이 있을 때만 배경 모델 업데이트 및 포그라운드 마스크 생성
    fgMask = backSub.apply(frame)

    # 모폴로지 변형을 통해 노이즈 제거
    fgMask_morph = cv.morphologyEx(fgMask, cv.MORPH_OPEN, kernel)

    # 외곽선 검출
    contours, _ = cv.findContours(fgMask_morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # 모든 외곽선 내부를 배경으로 채움
    for contour in contours:
        if cv.contourArea(contour) > 3000:  # 작은 영역 무시
            # 외곽선 내부를 배경으로 채우기
            mask = np.zeros_like(frame)
            cv.drawContours(mask, [contour], -1, (255, 255, 255), -1)
            frame = cv.bitwise_and(frame, cv.bitwise_not(mask)) + cv.bitwise_and(background, mask)

    # 배경 이미지 업데이트
    if cv.countNonZero(motion_mask) > 50:  # 작은 움직임도 감지하도록 조건 조정
        background = cv.addWeighted(background, 0.999, frame, 0.001, 0)

    # 외곽선을 빨간색으로 표시
    for contour in contours:
        if cv.contourArea(contour) > 3000:  # 작은 영역 무시
            cv.drawContours(frame, [contour], -1, (0, 0, 255), 2)

    # 프레임 번호를 현재 프레임에 표시
    cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
    cv.putText(frame, str(int(capture.get(cv.CAP_PROP_POS_FRAMES))), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    # 이전 프레임 업데이트
    prev_frame = gray_frame.copy()

    # 결과 프레임을 화면에 표시
    cv.imshow('Background Removed Video', frame)

    keyboard = cv.waitKey(1) & 0xFF
    if keyboard == 27:
        break

capture.release()
cv.destroyAllWindows()
