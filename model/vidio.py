from ultralytics import YOLO
import cv2
import os

# ===================== 换成你自己的火焰模型！ =====================
model = YOLO(r"D:\CVSTUDY\Final\runs\detect\fire_final\weights\best.pt")
# =================================================================

VIDEO_FOLDER = r"D:\CVSTUDY\Final\vedio"
OUTPUT_FOLDER = os.path.join(VIDEO_FOLDER, "fire_detected_videos")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

video_list = [
    "0c49baf8-38e711ad.mp4",
    "00e18d86-3389d8dc.mp4",
    "02ced98b-b85f206e.mp4",
    "11925064_1920_1080_25fps.mp4",
    "12263032_3840_2160_30fps.mp4",
    "12263621_3840_2160_30fps.mp4",
    "12266932_1080_1920_60fps.mp4",
    "12393269_1440_2560_30fps.mp4",
    "12487517_1920_1080_30fps.mp4",
    "12487565_1920_1080_30fps.mp4",
    "12629298_1920_1080_30fps.mp4",
    "12834101_2160_3840_30fps.mp4",
    "12906053_1080_1920_30fps.mp4",
    "13123780_1080_1920_60fps.mp4",
    "13201780_1920_1080_25fps.mp4",
    "13228967_3840_2160_30fps.mp4",
    "13236224_3840_2160_24fps.mp4",
    "1226103820.mp4"
]

for i, v_name in enumerate(video_list, 1):
    video_path = os.path.join(VIDEO_FOLDER, v_name)
    print(f"\n🎬 处理 {i}/{len(video_list)}：{v_name}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 打不开：{video_path}")
        continue

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    save_path = os.path.join(OUTPUT_FOLDER, f"detected_{v_name}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        res = model(frame, verbose=False)
        frame = res[0].plot()

        out.write(frame)

    cap.release()
    out.release()
    print(f"✅ 完成：{save_path}")

print("\n🎉 所有视频处理完毕！")
cv2.destroyAllWindows()