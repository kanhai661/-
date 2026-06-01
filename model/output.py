import gradio as gr
from ultralytics import YOLO
import requests
import json

# 加载模型
model = YOLO(r"D:\CVSTUDY\Final\runs\detect\fire_gpu_final\weights\best.pt")

# Ollama
def get_ai_analysis(total_img, fire_img, total_fire):
    prompt = f"""
检测结果：
总图片：{total_img}
含火焰：{fire_img}
火焰总数：{total_fire}

请分析火情等级、风险、处理建议、预防措施。
"""
    res = requests.post("http://localhost:11434/api/generate", json={
        "model": "qwen2.5:0.5b",
        "prompt": prompt,
        "stream": False
    })
    return res.json()["response"]

# 检测函数
def detect_image(img):
    results = model(img, conf=0.25)
    fire_num = len(results[0].boxes)
    ai_report = get_ai_analysis(1, 1 if fire_num>0 else 0, fire_num)
    return results[0].plot(), ai_report

# 界面
with gr.Blocks(title="火灾检测+NLP智能分析") as demo:
    gr.Markdown("# 🔥 火灾检测系统 + Ollama大模型")
    with gr.Row():
        image_in = gr.Image(type="pil", label="上传图片")
        image_out = gr.Image(label="检测结果")
    report = gr.Textbox(label="AI智能分析报告", lines=8)
    btn = gr.Button("开始检测")
    btn.click(detect_image, inputs=image_in, outputs=[image_out, report])

demo.launch()