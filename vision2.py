import cv2
import numpy as np

def process_video(video_path):
    # 비디오 캡처 초기화
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Background Subtractor 객체 생성 (KNN 사용)
    fgbg = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400.0, detectShadows=False)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 배경 차별화 적용
        fgmask = fgbg.apply(frame)
        
        # 마스크를 이용하여 교수님 영역 추출
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        
        # 마스크 후처리 (경계 선명하게 하기)
        fgmask = cv2.dilate(fgmask, kernel, iterations=2)
        fgmask = cv2.erode(fgmask, kernel, iterations=2)
        
        # 교수님 영역만 추출하여 배경으로 대체
        fgmask_inv = cv2.bitwise_not(fgmask)
        fg = cv2.bitwise_and(frame, frame, mask=fgmask_inv)
        
        # 실시간 배경 업데이트
        bg_frame = fgbg.getBackgroundImage()
        if bg_frame is not None:
            bg_part = cv2.bitwise_and(bg_frame, bg_frame, mask=fgmask)
            combined = cv2.add(fg, bg_part)
        else:
            combined = fg
        
        # 결과 프레임 출력
        cv2.imshow('Frame', combined)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 사용자에게 동영상 파일 경로 입력 받기
video_path = input("동영상 파일 경로를 입력하세요: ").strip()
process_video(video_path)
