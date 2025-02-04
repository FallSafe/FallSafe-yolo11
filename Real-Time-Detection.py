import cv2
import torch
import threading
import multiprocessing
import time
from ultralytics import YOLO
from flask import Flask, render_template, Response, request, jsonify
from Email import send_email_alert
from Whatsapp import send_whatsapp_alert
from Message import send_sms_alert
from icecream import ic
import os
import shutil

model = YOLO("model/model.pt")

app = Flask(__name__)

fall_detected = False
fall_detected_lock = threading.Lock()
fall_detected_time = None
alert_set = False
recipient = ""
tonumber = ""
confidence = 1

def process_predictions(results, frame):
    global fall_detected, fall_detected_time
    boxes = results[0].boxes
    model_names = model.names

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    for box in boxes:
        class_id = int(box.cls.item())
        class_name = model_names[class_id]
        conf = box.conf.item()
        if class_name == "fall":
            fall_detected = True
            fall_detected_time = time.time()

            frame_path = os.path.join(output_dir, f"fall_frame_{fall_detected_time}.jpg")
            if not cv2.imwrite(frame_path, frame):
                print(f"Failed to save frame at {frame_path}")

            p_email = multiprocessing.Process(target=send_email_alert, kwargs={
                "label": "Fall Detected!",
                "confidence_score": conf,
                "receiver_email": recipient,
                "frame_path": frame_path
            })
            
            p_email.start()
            p_email.join()

            p_sms = multiprocessing.Process(target=send_sms_alert, args=(tonumber,))
            p_whatsapp = multiprocessing.Process(target=send_whatsapp_alert, args=(tonumber,))

            p_sms.start()
            p_whatsapp.start()

            p_sms.join()
            p_whatsapp.join()

            ic(f"Fall detected !!!! with confidence: {conf:.2f}")

        elif class_name == "nofall":
            fall_detected = False

    return fall_detected


def generate_frames():
    global alert_set,confidence,cap
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if alert_set:
            results = model.predict(source=frame,conf=confidence)
            process_predictions(results, frame)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html', fall_detected=fall_detected, alert_set=alert_set)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/send_details', methods=['POST'])
def send_alert():
    global alert_set, recipient ,tonumber,confidence
    data = request.get_json()
    recipient = data.get('email', None)
    tonumber = data.get('phone', None)
    confidence = data.get('conf', None)
    confidence = float(confidence)
    ic(data,recipient,tonumber,confidence)
    if recipient and tonumber and confidence:
        alert_set = True
        return jsonify({"message": "Email and Phone saved successfully!"})
    return jsonify({"message": "Invalid details"}), 400

@app.route('/fall_status')
def updateFallStatus():
    return jsonify({"status": fall_detected})

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    output_dir = "output"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    app.run(debug=True, host='0.0.0.0', port=5000)
