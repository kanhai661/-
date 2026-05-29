from ultralytics import YOLO
import requests
import json


# ======================== 批量测试图片（适配你当前路径） ========================
def test_images():
    print("\n===== 开始批量检测图片 🔥 =====")

    # 你的训练好的模型权重
    model = YOLO(r"D:\CVSTUDY\Final\runs\detect\fire_gpu_final\weights\best.pt")

    # 适配你截图里真实存在的测试图片路径！
    IMAGE_FOLDER = r"D:\CVSTUDY\Final\data\test\images"

    results = model.predict(
        source=IMAGE_FOLDER,
        conf=0.25,
        save=True,
        show=False
    )

    total = len(results)
    fire_count = 0
    total_fire_boxes = 0

    for res in results:
        cnt = len(res.boxes)
        total_fire_boxes += cnt
        if cnt > 0:
            fire_count += 1

    print(f"\n✅ 检测完成！")
    print(f"总图片数：{total} 张")
    print(f"检测到火焰：{fire_count} 张")
    print(f"总火焰目标数量：{total_fire_boxes} 个")
    print(f"未检测到火焰：{total - fire_count} 张")
    print("\n结果已保存到： runs/detect/predict/")

    return total, fire_count, total_fire_boxes


# ======================== Ollama 本地AI 分析 ========================
def ollama_ai_analysis(total_img, fire_img, total_fire):
    print("\n===== 🤖 正在调用 Ollama 本地大模型分析 =====")

    OLLAMA_API = "http://localhost:11434/api/generate"
    MODEL_NAME = "qwen2.5:0.5b"

    prompt = f"""
你是专业消防安全专家。请根据以下检测结果，生成专业火灾分析报告：

检测结果：
- 总检测图片：{total_img} 张
- 含火焰图片：{fire_img} 张
- 火焰目标总数：{total_fire} 个

请你分析：
1. 火情危险等级（低/中/高/紧急）
2. 火灾风险评估
3. 应急处理建议
4. 预防措施

请用专业、简洁、条理清晰的中文回答。
"""

    data = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }

    try:
        res = requests.post(OLLAMA_API, json=data)
        return res.json()["response"]
    except Exception as e:
        return f"❌ 请确保 Ollama 正在运行！错误：{str(e)}"


# ======================== 主程序 ========================
if __name__ == '__main__':
    print("=" * 70)
    print("        🔥 火灾检测系统（YOLOv8 + Ollama本地大模型）")
    print("=" * 70)

    # 1. 测试图片（适配你当前目录）
    total_img, fire_img, total_fire = test_images()

    # 2. Ollama AI 分析
    ai_report = ollama_ai_analysis(total_img, fire_img, total_fire)

    # 3. 输出最终报告
    print("\n" + "=" * 70)
    print("               📝 Ollama AI 火灾分析报告")
    print("=" * 70)
    print(ai_report)

    # 4. 保存报告
    with open("火灾智能检测报告.txt", "w", encoding="utf-8") as f:
        f.write(f"""
火灾视觉检测结果：
- 总图片数：{total_img}
- 含火焰图片：{fire_img}
- 火焰总数：{total_fire}

AI 专业分析：
{ai_report}
""")

    print("\n✅ 全部完成！报告已保存：火灾智能检测报告.txt")