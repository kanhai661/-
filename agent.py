# -*- coding: utf-8 -*-
import requests
import json
from datetime import datetime

OLLAMA_BASE = "http://localhost:11434"
MODEL_NAME = "mannix/smallthinker:q2_k"

SYSTEM_PROMPT = """你是"火灾智能检测系统"的安全顾问智能体，名为「火卫」。你的职责是为用户提供专业的消防安全建议和知识解答。

## 你的核心身份
- 你是一个专业的消防安全顾问AI，专注于火灾预防、检测、应急响应领域
- 你服务于一个基于YOLOv8深度学习的火灾智能检测系统
- 你的回答必须基于科学事实和消防安全规范，不得编造数据

## 你的两大工作模式

### 模式一：检测结果分析（当接收到YOLO检测结果时）
当用户发送检测结果数据时，你需要：
1. **风险评估**：根据置信度、目标数量、检测类型等判断火情严重程度
   - 置信度 > 0.8 且目标数 > 0：高风险，需立即响应
   - 置信度 0.5~0.8 且目标数 > 0：中风险，需密切关注
   - 置信度 < 0.5 或无目标：低风险，可能为误检，建议复核
2. **应急建议**：根据风险等级给出对应级别的处置建议
3. **预防措施**：针对检测场景提出后续预防建议
4. **注意事项**：提醒用户关注的关键安全要点

### 模式二：安全知识问答（当用户直接提问时）
你可以回答以下领域的安全问题：
- 火灾预防知识（家庭、办公、工业、森林等场景）
- 火灾应急逃生方法
- 灭火器使用与选择
- 消防法规与标准
- 火灾报警与求救
- 电气火灾防范
- 化学品火灾处理
- 建筑消防设施
- 深度学习在火灾检测中的应用
- YOLO系列算法原理与特点
- 本系统的使用方法与功能说明

## 回答规范
1. 回答要结构化，使用编号列表或分段，条理清晰
2. 涉及安全建议时，按紧急程度排序，最紧急的放最前面
3. 如果问题超出消防安全领域，礼貌说明并尝试从安全角度给出关联建议
4. 不要编造不存在的法规、标准编号或统计数据
5. 回答语言简洁专业，避免过度冗长
6. 当检测到火灾时，必须强调"人身安全第一，先撤离再报警"

## 检测结果分析模板
当收到检测结果时，按以下格式回复：

🔥 **火情风险评估**
- 风险等级：[高/中/低]
- 检测置信度：[数值]
- 检测目标数：[数值]
- 检测模式：[图像/视频/摄像头]

🚨 **应急建议**
[根据风险等级给出1-3条最紧急的处置建议]

🛡️ **预防措施**
[2-3条后续预防建议]

⚠️ **特别提醒**
[关键安全注意事项]"""


def chat_with_agent(messages, stream=False):
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": stream,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_predict": 1024
        }
    }

    if stream:
        return _stream_chat(payload)
    else:
        return _normal_chat(payload)


def _normal_chat(payload):
    resp = requests.post(f"{OLLAMA_BASE}/api/chat", json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data.get("message", {}).get("content", "")


def _stream_chat(payload):
    payload["stream"] = True
    resp = requests.post(f"{OLLAMA_BASE}/api/chat", json=payload, stream=True, timeout=120)
    resp.raise_for_status()

    for line in resp.iter_lines():
        if not line:
            continue
        chunk = json.loads(line)
        content = chunk.get("message", {}).get("content", "")
        if content:
            yield f"data: {json.dumps({'type': 'content', 'content': content}, ensure_ascii=False)}\n\n"
        if chunk.get("done"):
            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
            return


def build_detection_context(detection_result):
    has_fire = detection_result.get("hasFire", False)
    confidence = detection_result.get("confidence", 0)
    count = detection_result.get("count", 0)
    detect_type = detection_result.get("detectType", "unknown")
    process_time = detection_result.get("processTime", 0)

    type_map = {"image": "图像上传检测", "video": "视频文件检测", "camera": "摄像头实时检测"}
    type_label = type_map.get(detect_type, detect_type)

    if has_fire:
        context = f"""系统刚刚完成一次火灾检测，检测结果如下：
- 检测模式：{type_label}
- 是否检测到火焰：是 ⚠️
- 最高置信度：{confidence:.4f}（{confidence * 100:.1f}%）
- 检测目标数量：{count}个火焰区域
- 处理耗时：{process_time}秒

请根据以上检测结果，给出风险评估和安全建议。"""
    else:
        context = f"""系统刚刚完成一次火灾检测，检测结果如下：
- 检测模式：{type_label}
- 是否检测到火焰：否 ✅
- 最高置信度：{confidence:.4f}（{confidence * 100:.1f}%）
- 检测目标数量：0
- 处理耗时：{process_time}秒

本次检测未发现火灾，请给出安全确认和日常预防建议。"""

    return context


def check_ollama_available():
    try:
        resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            return MODEL_NAME in model_names or any(MODEL_NAME.split(":")[0] in n for n in model_names)
        return False
    except Exception:
        return False