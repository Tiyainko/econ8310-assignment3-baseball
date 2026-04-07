import cv2
import os

VIDEO_DIR = "data/videos"
FRAME_DIR = "data/frames"

os.makedirs(FRAME_DIR, exist_ok=True)

for video_file in os.listdir(VIDEO_DIR):
    if video_file.endswith(".mov"):
        video_path = os.path.join(VIDEO_DIR, video_file)
        cap = cv2.VideoCapture(video_path)

        video_name = os.path.splitext(video_file)[0]
        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_name = f"{video_name}_frame_{frame_id}.jpg"
            frame_path = os.path.join(FRAME_DIR, frame_name)

            cv2.imwrite(frame_path, frame)
            frame_id += 1

        cap.release()
        print(f"Finished {video_file} with {frame_id} frames")
