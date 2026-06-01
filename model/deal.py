import os
import random
import shutil

# ===================== 【终极正确路径】 =====================
BASE_DIR = r"D:\CVSTUDY\Final\real_images\real_images"

# 你的真实结构！！！
FIRE_IMAGES_DIR = os.path.join(BASE_DIR, "real_fire", "images")
FIRE_LABELS_DIR = os.path.join(BASE_DIR, "real_fire", "labels")

NON_FIRE_DIR = os.path.join(BASE_DIR, "real_non_fire")

OUTPUT_DATASET_DIR = os.path.join(BASE_DIR, "fire_dataset")
TRAIN_RATIO = 0.8
# =============================================================

# 创建输出文件夹
os.makedirs(os.path.join(OUTPUT_DATASET_DIR, "images", "train"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DATASET_DIR, "images", "val"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DATASET_DIR, "labels", "train"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DATASET_DIR, "labels", "val"), exist_ok=True)

all_images = []

# 加载火焰图片（正确从 images 文件夹读）
for img_name in os.listdir(FIRE_IMAGES_DIR):
    if img_name.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        all_images.append((os.path.join(FIRE_IMAGES_DIR, img_name), "fire"))

# 加载非火焰图片
for img_name in os.listdir(NON_FIRE_DIR):
    if img_name.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        all_images.append((os.path.join(NON_FIRE_DIR, img_name), "non_fire"))

print(f"✅ 总共加载图片：{len(all_images)} 张")

# 划分训练/验证
random.shuffle(all_images)
split = int(len(all_images) * TRAIN_RATIO)
train_list = all_images[:split]
val_list = all_images[split:]

# 处理复制
def process_data(data_list, mode):
    img_dst = os.path.join(OUTPUT_DATASET_DIR, "images", mode)
    lab_dst = os.path.join(OUTPUT_DATASET_DIR, "labels", mode)

    for img_path, typ in data_list:
        name = os.path.basename(img_path)  # ✅ 这里修好了！
        shutil.copy(img_path, os.path.join(img_dst, name))
        txt_name = os.path.splitext(name)[0] + ".txt"

        if typ == "fire":
            # 从正确的 labels 文件夹复制！！！
            src_label = os.path.join(FIRE_LABELS_DIR, txt_name)
            if os.path.exists(src_label):
                shutil.copy(src_label, os.path.join(lab_dst, txt_name))
            else:
                open(os.path.join(lab_dst, txt_name), "w").close()
        else:
            open(os.path.join(lab_dst, txt_name), "w").close()

process_data(train_list, "train")
process_data(val_list, "val")

print("\n🎉 数据集处理完成！标签已正确复制！")
print(f"训练集：{len(train_list)} 张")
print(f"验证集：{len(val_list)} 张")