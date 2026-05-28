from ultralytics import YOLO
import torch

if __name__ == '__main__':
    print("GPU可用:", torch.cuda.is_available())
    model = YOLO("yolov8s.pt")

    model.train(
        data=r"D:\CVSTUDY\Final\real_images\real_images\fire_dataset\data.yaml",
        epochs=10,
        imgsz=640,
        batch=4,
        device=0,
        workers=2,
        name="fire_gpu_final"
    )