from flask import Flask, request, jsonify, send_file, send_from_directory, Response
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import os
import uuid
import base64
from datetime import datetime
import torch
import pymysql
import hashlib
import json as _json

# ========== 路径配置 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, '..', 'frontend')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
RESULT_FOLDER = os.path.join(BASE_DIR, 'results')
MODEL_PATH = r"C:\code\machine_view\Final\model\weights\best.pt"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')
CORS(app)

# ========== MySQL 数据库配置 ==========
DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': 'Rikka666',
    'database': 'firedec',
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}


def get_db():
    """获取数据库连接"""
    return pymysql.connect(**DB_CONFIG)


def init_db():
    """初始化数据库表"""
    conn = get_db()
    try:
        with conn.cursor() as cursor:
            # 检测记录表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detections (
                    id INT PRIMARY KEY AUTO_INCREMENT,
                    detect_type VARCHAR(20) NOT NULL COMMENT 'image/video/camera',
                    filename VARCHAR(255) COMMENT '原始文件名',
                    has_fire TINYINT NOT NULL DEFAULT 0 COMMENT '0=无火,1=有火',
                    confidence FLOAT COMMENT '最高置信度',
                    count INT DEFAULT 0 COMMENT '检测目标数',
                    process_time FLOAT COMMENT '处理耗时(秒)',
                    result_path VARCHAR(500) COMMENT '结果文件路径',
                    device_ip VARCHAR(50) COMMENT '检测端IP',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='火灾检测记录'
            ''')

            # 检测框详情表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detection_boxes (
                    id INT PRIMARY KEY AUTO_INCREMENT,
                    detection_id INT NOT NULL,
                    class_name VARCHAR(50) COMMENT 'fire/smoke',
                    confidence FLOAT,
                    x1 FLOAT, y1 FLOAT, x2 FLOAT, y2 FLOAT,
                    FOREIGN KEY (detection_id) REFERENCES detections(id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='检测框详情'
            ''')

            # 用户表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INT PRIMARY KEY AUTO_INCREMENT,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    password VARCHAR(255) NOT NULL COMMENT 'SHA256哈希',
                    role VARCHAR(20) DEFAULT 'operator' COMMENT 'admin/operator/viewer',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='系统用户'
            ''')

            # 插入默认管理员（admin/admin123）
            default_pass = hashlib.sha256('admin123'.encode()).hexdigest()
            cursor.execute('''
                INSERT IGNORE INTO users (id, username, password, role)
                VALUES (1, 'admin', %s, 'admin')
            ''', (default_pass,))

        conn.commit()
        print("✅ 数据库初始化完成")
    except Exception as e:
        print(f"❌ 数据库初始化失败: {e}")
        raise
    finally:
        conn.close()


# 启动时初始化数据库
#init_db()


def save_detection(detect_type, filename, has_fire, confidence, count,
                   process_time, result_path, device_ip, boxes=None):
    """
    保存检测记录到数据库
    boxes: [{class_name, confidence, x1, y1, x2, y2}, ...]
    """
    conn = get_db()
    try:
        with conn.cursor() as cursor:
            cursor.execute('''
                INSERT INTO detections 
                (detect_type, filename, has_fire, confidence, count, process_time, result_path, device_ip)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ''', (detect_type, filename, 1 if has_fire else 0, confidence,
                  count, process_time, result_path, device_ip))

            detection_id = cursor.lastrowid

            # 保存每个检测框
            if boxes:
                for box in boxes:
                    cursor.execute('''
                        INSERT INTO detection_boxes 
                        (detection_id, class_name, confidence, x1, y1, x2, y2)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ''', (detection_id, box['class'], box['confidence'],
                          box['bbox'][0], box['bbox'][1], box['bbox'][2], box['bbox'][3]))

        conn.commit()
        return detection_id
    finally:
        conn.close()


def get_detection_history(limit=50, offset=0, has_fire=None):
    """查询检测历史"""
    conn = get_db()
    try:
        with conn.cursor() as cursor:
            sql = 'SELECT * FROM detections WHERE 1=1'
            params = []

            if has_fire is not None:
                sql += ' AND has_fire = %s'
                params.append(1 if has_fire else 0)

            sql += ' ORDER BY created_at DESC LIMIT %s OFFSET %s'
            params.extend([limit, offset])

            cursor.execute(sql, params)
            rows = cursor.fetchall()

            result = []
            for row in rows:
                # 查询该记录的检测框详情
                cursor.execute('SELECT * FROM detection_boxes WHERE detection_id = %s', (row['id'],))
                row['boxes'] = cursor.fetchall()
                result.append(row)

            return result
    finally:
        conn.close()


def get_stats():
    """获取统计数据"""
    conn = get_db()
    try:
        with conn.cursor() as cursor:
            # 总检测次数
            cursor.execute('SELECT COUNT(*) as total FROM detections')
            total = cursor.fetchone()['total']

            # 火灾次数
            cursor.execute('SELECT COUNT(*) as fire_count FROM detections WHERE has_fire = 1')
            fire_count = cursor.fetchone()['fire_count']

            # 今日检测
            cursor.execute('''
                SELECT COUNT(*) as today FROM detections 
                WHERE DATE(created_at) = CURDATE()
            ''')
            today = cursor.fetchone()['today']

            # 按小时统计（最近24小时）
            cursor.execute('''
                SELECT HOUR(created_at) as hour, COUNT(*) as cnt
                FROM detections 
                WHERE created_at >= DATE_SUB(NOW(), INTERVAL 1 DAY)
                GROUP BY HOUR(created_at)
                ORDER BY hour
            ''')
            hourly = cursor.fetchall()

            return {'total': total, 'fire_count': fire_count, 'today': today, 'hourly': hourly}
    finally:
        conn.close()


# ========== 模型加载 ==========
print("正在加载模型...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO(MODEL_PATH)
model.to(device)
print(f"模型加载完成，使用设备: {device}")


# ========== 工具函数 ==========
def save_base64_image(base64_str, save_path):
    img_data = base64.b64decode(base64_str.split(',')[1])
    with open(save_path, 'wb') as f:
        f.write(img_data)
    return save_path


def image_to_base64(image_path):
    with open(image_path, 'rb') as f:
        return 'data:image/jpeg;base64,' + base64.b64encode(f.read()).decode()


# ========== 路由 ==========

@app.route('/')
def index():
    return send_from_directory(FRONTEND_DIR, 'dec.html')


@app.route('/api/detect/image', methods=['POST'])
def detect_image():
    try:
        start_time = datetime.now()
        client_ip = request.remote_addr

        if 'file' in request.files:
            file = request.files['file']
            ext = os.path.splitext(file.filename)[1]
            original_name = file.filename
            filename = f"{uuid.uuid4()}{ext}"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
        elif request.json and 'image' in request.json:
            original_name = 'camera_capture.jpg'
            filename = f"{uuid.uuid4()}.jpg"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            save_base64_image(request.json['image'], filepath)
        else:
            return jsonify({'error': '没有提供图片'}), 400

        # 推理
        results = model(filepath, conf=0.25)
        result = results[0]

        # 保存标注结果
        result_filename = f"result_{filename}"
        result_path = os.path.join(RESULT_FOLDER, result_filename)
        result.save(filename=result_path)

        # 解析结果
        boxes = result.boxes
        has_fire = len(boxes) > 0
        detections_list = []

        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            detections_list.append({
                'class': result.names[cls],
                'confidence': round(conf, 4),
                'bbox': [round(x, 2) for x in xyxy]
            })

        process_time = (datetime.now() - start_time).total_seconds()

        # 保存到数据库
        db_id = save_detection(
            detect_type='image',
            filename=original_name,
            has_fire=has_fire,
            confidence=round(max([d['confidence'] for d in detections_list], default=0), 4),
            count=len(detections_list),
            process_time=round(process_time, 3),
            result_path=result_path,
            device_ip=client_ip,
            boxes=detections_list
        )

        response = {
            'success': True,
            'hasFire': has_fire,
            'detections': detections_list,
            'count': len(detections_list),
            'confidence': round(max([d['confidence'] for d in detections_list], default=0), 4),
            'processTime': round(process_time, 3),
            'resultImage': image_to_base64(result_path),
            'recordId': db_id,
            'timestamp': datetime.now().isoformat()
        }

        return jsonify(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/detect/video', methods=['POST'])
def detect_video():
    try:
        client_ip = request.remote_addr

        if 'file' not in request.files:
            return jsonify({'error': '没有提供视频文件'}), 400

        file = request.files['file']
        original_name = file.filename
        ext = os.path.splitext(file.filename)[1]
        filename = f"{uuid.uuid4()}{ext}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        cap = cv2.VideoCapture(filepath)
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        result_filename = f"detected_{filename}"
        result_path = os.path.join(RESULT_FOLDER, result_filename)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(result_path, fourcc, fps, (w, h))

        frame_count = 0
        fire_frames = 0
        max_confidence = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            res = model(frame, verbose=False)
            annotated = res[0].plot()
            out.write(annotated)

            if len(res[0].boxes) > 0:
                fire_frames += 1
                for box in res[0].boxes:
                    conf = float(box.conf[0])
                    if conf > max_confidence:
                        max_confidence = conf
            frame_count += 1

        cap.release()
        out.release()

        # 保存到数据库
        db_id = save_detection(
            detect_type='video',
            filename=original_name,
            has_fire=fire_frames > 0,
            confidence=round(max_confidence, 4),
            count=fire_frames,
            process_time=None,
            result_path=result_path,
            device_ip=client_ip,
            boxes=None
        )

        return jsonify({
            'success': True,
            'downloadUrl': f'/api/download/{result_filename}',
            'totalFrames': frame_count,
            'fireFrames': fire_frames,
            'recordId': db_id,
            'message': '视频处理完成'
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/detect/camera', methods=['POST'])
def detect_camera():
    try:
        client_ip = request.remote_addr
        data = request.json

        if not data or 'frame' not in data:
            return jsonify({'error': '没有提供帧数据'}), 400

        filename = f"{uuid.uuid4()}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        save_base64_image(data['frame'], filepath)

        results = model(filepath, conf=0.25, verbose=False)
        result = results[0]

        has_fire = len(result.boxes) > 0
        confidence = 0
        boxes_list = []

        if has_fire:
            result_path = os.path.join(RESULT_FOLDER, f"camera_{filename}")
            result.save(filename=result_path)
            result_image = image_to_base64(result_path)

            confidence = round(float(result.boxes.conf.max()), 4)
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                boxes_list.append({
                    'class': result.names[cls],
                    'confidence': round(conf, 4),
                    'bbox': [round(x, 2) for x in xyxy]
                })
        else:
            result_image = None
            result_path = None

        # 保存到数据库
        db_id = save_detection(
            detect_type='camera',
            filename='camera_live.jpg',
            has_fire=has_fire,
            confidence=confidence,
            count=len(result.boxes),
            process_time=None,
            result_path=result_path,
            device_ip=client_ip,
            boxes=boxes_list if has_fire else None
        )

        return jsonify({
            'success': True,
            'hasFire': has_fire,
            'count': len(result.boxes),
            'confidence': confidence,
            'resultImage': result_image,
            'recordId': db_id,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/history', methods=['GET'])
def get_history():
    """获取检测历史"""
    try:
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        has_fire = request.args.get('has_fire', None, type=int)

        data = get_detection_history(limit=limit, offset=offset, has_fire=has_fire)
        return jsonify({'success': True, 'data': data, 'count': len(data)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_statistics():
    """获取统计数据"""
    try:
        stats = get_stats()
        return jsonify({'success': True, 'data': stats})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/download/<filename>', methods=['GET'])
def download_file(filename):
    filepath = os.path.join(RESULT_FOLDER, filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return jsonify({'error': '文件不存在'}), 404


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'running',
        'modelLoaded': True,
        'device': device,
        'cudaAvailable': torch.cuda.is_available()
    })


# ========== 智能体模块 ==========
from agent import chat_with_agent, build_detection_context, check_ollama_available, SYSTEM_PROMPT


@app.route('/api/agent/chat', methods=['POST'])
def agent_chat():
    try:
        data = request.json
        if not data or 'messages' not in data:
            return jsonify({'error': '缺少messages参数'}), 400

        user_messages = data['messages']
        stream = data.get('stream', False)

        full_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        full_messages.extend(user_messages)

        if stream:
            def generate():
                for chunk in chat_with_agent(full_messages, stream=True):
                    yield chunk
            return Response(generate(), mimetype='text/event-stream',
                            headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})
        else:
            reply = chat_with_agent(full_messages, stream=False)
            return jsonify({'success': True, 'reply': reply})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/agent/detect_advice', methods=['POST'])
def agent_detect_advice():
    try:
        data = request.json
        if not data:
            return jsonify({'error': '缺少检测数据'}), 400

        context = build_detection_context(data)
        stream = data.get('stream', False)

        full_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context}
        ]

        if stream:
            def generate():
                for chunk in chat_with_agent(full_messages, stream=True):
                    yield chunk
            return Response(generate(), mimetype='text/event-stream',
                            headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})
        else:
            reply = chat_with_agent(full_messages, stream=False)
            return jsonify({'success': True, 'reply': reply})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/agent/status', methods=['GET'])
def agent_status():
    available = check_ollama_available()
    return jsonify({
        'available': available,
        'model': 'mannix/smallthinker:q2_k',
        'message': '智能体就绪' if available else 'Ollama服务未启动或模型未加载'
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)