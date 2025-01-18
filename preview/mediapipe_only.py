#mediapipe body + hand + body shape mask
import cv2
import time
import numpy as np
import mediapipe as mp

# MediaPipe Pose 및 Hands 초기화
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=4, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 카메라 입력
video_source = 0
cap = cv2.VideoCapture(video_source)

# 배경 저장 변수 및 프레임 크기
background_frame = None
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# FPS 계산 변수
prev_frame_time = 0

# 신체 모양대로 마스크 생성 함수
def create_detailed_pose_mask(pose_landmarks, hand_landmarks_list, frame_width, frame_height):
    mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

    if not pose_landmarks:
        return mask

    # 랜드마크 가져오기
    landmarks = pose_landmarks.landmark
    keypoints = {
        "nose": landmarks[mp_pose.PoseLandmark.NOSE],
        "left_shoulder": landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
        "right_shoulder": landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
        "left_hip": landmarks[mp_pose.PoseLandmark.LEFT_HIP],
        "right_hip": landmarks[mp_pose.PoseLandmark.RIGHT_HIP],
    }

    # 머리와 목 직사각형 처리
    nose_x = int(keypoints["nose"].x * frame_width)
    nose_y = int(keypoints["nose"].y * frame_height)
    left_shoulder = (int(keypoints["left_shoulder"].x * frame_width), int(keypoints["left_shoulder"].y * frame_height))
    right_shoulder = (int(keypoints["right_shoulder"].x * frame_width), int(keypoints["right_shoulder"].y * frame_height))

    head_top_y = max(nose_y - 70, 0)
    neck_bottom_y = min(left_shoulder[1], right_shoulder[1])
    cv2.rectangle(mask, (nose_x - 50, head_top_y), (nose_x + 50, neck_bottom_y), 255, -1)  # 머리와 목 연결 직사각형

    # 몸통 영역 다각형
    body_points = [
        left_shoulder,
        right_shoulder,
        (int(keypoints["right_hip"].x * frame_width), int(keypoints["right_hip"].y * frame_height)),
        (int(keypoints["left_hip"].x * frame_width), int(keypoints["left_hip"].y * frame_height))
    ]
    body_polygon = np.array(body_points, dtype=np.int32)
    cv2.fillConvexPoly(mask, body_polygon, 255)

    # 하체 부분 마스크 처리: 좁은 사각형으로
    left_hip = (int(keypoints["left_hip"].x * frame_width), int(keypoints["left_hip"].y * frame_height))
    right_hip = (int(keypoints["right_hip"].x * frame_width), int(keypoints["right_hip"].y * frame_height))
    center_hip_x = (left_hip[0] + right_hip[0]) // 2
    center_hip_y = (left_hip[1] + right_hip[1]) // 2
    h_rect_width = 150  # 하체 마스크의 폭 (줄임)
    h_rect_height = 300  # 하체 마스크의 높이
    cv2.rectangle(mask, (center_hip_x - h_rect_width // 2, center_hip_y),
                  (center_hip_x + h_rect_width // 2, frame_height), 255, -1)

    # POSE_CONNECTIONS를 사용하여 랜드마크 연결선을 그리기
    for connection in mp_pose.POSE_CONNECTIONS:
        start_idx, end_idx = connection
        start_point = landmarks[start_idx]
        end_point = landmarks[end_idx]

        # 가시성이 낮은 경우 무시
        if start_point.visibility < 0.5 or end_point.visibility < 0.5:
            continue

        # 좌표 변환
        x1, y1 = int(start_point.x * frame_width), int(start_point.y * frame_height)
        x2, y2 = int(end_point.x * frame_width), int(end_point.y * frame_height)

        # 선 그리기: 어깨 연결선은 두껍게 처리
        line_thickness = 30 if start_idx in [mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                                             mp_pose.PoseLandmark.RIGHT_SHOULDER.value] else 20
        cv2.line(mask, (x1, y1), (x2, y2), 255, thickness=line_thickness)

    return mask

# 비디오 처리 루프
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 원본 프레임 유지
    original_frame = frame.copy()

    # MediaPipe Pose 및 Hands로 랜드마크 탐지
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(rgb_frame)
    hand_results = hands.process(rgb_frame)

    # MediaPipe Pose 기반 마스크 생성
    pose_mask = create_detailed_pose_mask(
        pose_results.pose_landmarks,
        hand_results.multi_hand_landmarks,
        frame_width,
        frame_height
    )

    # 마스크 후처리: 경계 부드럽게 처리
    kernel = np.ones((15, 15), np.uint8)  # 팽창 커널 크기 증가
    pose_mask = cv2.dilate(pose_mask, kernel, iterations=3)

    # 배경 업데이트: 사람 영역을 제외한 부분만 업데이트
    if background_frame is None:
        background_frame = frame.copy()
        print("Background initialized.")
    else:
        inverse_mask = cv2.bitwise_not(pose_mask)  # 사람 영역 제외
        background_frame[inverse_mask == 255] = frame[inverse_mask == 255]

    # 최종 프레임 생성: 사람 영역을 배경으로 덮어쓰기
    processed_frame = background_frame.copy()
    processed_frame[pose_mask == 255] = background_frame[pose_mask == 255]

    # MediaPipe Pose 및 손 뼈대를 최종 프레임에 표시 (랜드마크 점 제거)
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            processed_frame,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=None,  # 점 비활성화
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=3)
        )
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                processed_frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=None,  # 점 비활성화
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

    # FPS 계산 및 표시
    new_frame_time = time.time()
    fps = int(1 / (new_frame_time - prev_frame_time)) if prev_frame_time > 0 else 30
    prev_frame_time = new_frame_time
    cv2.putText(processed_frame, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Original Frame", original_frame)
    cv2.imshow("Processed Frame with Background Substitution", processed_frame)
    cv2.imshow("Mask", pose_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
