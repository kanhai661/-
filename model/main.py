from ultralytics import YOLO
import torch

if __name__ == '__main__':
    print("="*50)
    print("GPU 可用:", torch.cuda.is_available())
    print("CUDA 版本:", torch.version.cuda)
    print("当前 PyTorch 版本:", torch.__version__)
    print("="*50)

    model = YOLO("yolov8s.pt")

    model.train(
        data=r"D:\CVSTUDY\Final\real_images\real_images\fire_dataset\data.yaml",
        epochs=10,
        imgsz=640,
        batch=4,
        device=0,      # GPU 正常使用
        workers=0,      # 防止Windows报错
        name="fire_gpu_final"
    )