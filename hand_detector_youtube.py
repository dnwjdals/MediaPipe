# hand_detector_youtube.py
# -*- coding: utf-8 -*-

import cv2
import mediapipe as mp
import yt_dlp

# YouTube 스트림 URL을 가져오는 함수
def get_youtube_stream(url):
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'quiet': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info['url']

# 분석할 유튜브 영상 URL
youtube_url = "https://youtu.be/6mEq4UmUSyg?si=9Fxucka7MA-H6Ulb"

# YouTube 실시간 스트림 URL 가져오기
video_stream_url = get_youtube_stream(youtube_url)
print(f"▶ 유튜브 스트림 URL 불러옴 : {video_stream_url}")

# Mediapipe 손 인식 모듈 설정
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# YouTube 영상 열기
cap = cv2.VideoCapture(video_stream_url)

if not cap.isOpened():
    print("❌ 유튜브 영상을 열 수 없습니다.")
    exit()

print("✅ YouTube 영상 재생 시작 - 손 인식 준비 중...")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("⚠️ 프레임을 불러올 수 없습니다. (영상 종료)")
        break

    # 색상 변환 (BGR → RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    # 결과 시각화
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

    # 결과 창 표시
    cv2.imshow("🖐️ Mediapipe Hand Detector (YouTube)", image)

    # ESC 키 종료
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
