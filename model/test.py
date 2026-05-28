from ultralytics import YOLO
import os

# ===================== 你的模型 + 要检测的文件夹 =====================
model = YOLO(r"D:\CVSTUDY\Final\runs\detect\fire_gpu_final\weights\best.pt")

# 要检测的整个文件夹（你截图里的 test/images）
IMAGE_FOLDER = r"D:\CVSTUDY\Final\test\images"
# ====================================================================

if __name__ == '__main__':
    print("===== 开始批量检测整个文件夹 🔥 =====")

    # 批量预测
    results = model.predict(
        source=IMAGE_FOLDER,   # 直接传入文件夹路径
        conf=0.25,             # 灵敏度
        save=True,             # 自动保存所有结果
        show=False
    )

    # 统计结果
    total = len(results)
    fire_count = 0

    for res in results:
        if len(res.boxes) > 0:
            fire_count += 1

    print(f"\n✅ 检测完成！")
    print(f"总图片数：{total} 张")
    print(f"检测到火焰：{fire_count} 张")
    print(f"未检测到火焰：{total - fire_count} 张")
    print("\n结果已保存到： runs/detect/predict/")