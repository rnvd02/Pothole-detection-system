import os
import cv2
import re
import pytesseract
import logging
from flask import Flask, request, send_file, render_template
from plyer import notification
from ultralytics import YOLO

# -------- Configure Tesseract path for Windows --------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# -------- Flask setup --------
app = Flask(__name__, template_folder="templates", static_folder="static")
logging.basicConfig(level=logging.INFO)

# -------- Load Model --------
MODEL_PATH = "best (5).pt"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Missing model at {MODEL_PATH}")
model = YOLO(MODEL_PATH).to('cpu')

# -------- In-memory pothole list --------
pothole_coordinates = []

# -------- Routes --------

@app.route('/favicon.ico')
def favicon():
    return app.send_static_file('favicon.ico')

@app.route('/ping', methods=['GET'])
def ping():
    return "pong", 200

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/route-map')
def route_map():
    print(f"potholes data sent to template: {pothole_coordinates}")
    return render_template('route-map.html', potholes=pothole_coordinates)

@app.route('/upload', methods=['POST'])
def upload_video():
    logging.info(">>> /upload reached")
    global pothole_coordinates
    pothole_coordinates.clear()

    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    os.makedirs('uploads', exist_ok=True)
    os.makedirs('processed', exist_ok=True)
    in_path = os.path.join('uploads', file.filename)
    out_path = os.path.join('processed', 'detected_output.mp4')
    file.save(in_path)

    try:
        logging.info(f"Processing video ▶ {in_path}")
        process_video(in_path, out_path)
    except Exception as e:
        logging.exception("❌ process_video failed")
        return f"Server error:\n{e}", 500

    if not os.path.exists(out_path):
        return "Processed video missing", 500

    return send_file(out_path, mimetype='video/mp4')


# -------- OCR helper --------

def extract_gps_from_frame(frame):
    h = frame.shape[0]
    gps_region = frame[int(h * 0.80):, :]  # Bottom 20%

    gray = cv2.cvtColor(gps_region, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    text = pytesseract.image_to_string(thresh)
    print("📝 OCR text:", text)

    match = re.search(r'Lat\s*([\d.]+)[^\d]+Long\s*([\d.]+)', text)
    if match:
        lat = float(match.group(1))
        lng = float(match.group(2))
        print(f"✅ Extracted GPS: {lat}, {lng}")
        return {"lat": lat, "lng": lng}
    
    print("❌ GPS not found in this frame")
    return None


# -------- Video processing --------

def process_video(input_path, output_path):
    global pothole_coordinates

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open input {input_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not out.isOpened():
        raise IOError(f"Failed to open VideoWriter at {output_path}")

    first_pothole_detected = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection
        results = model(frame)
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()

            # Notification
            if not first_pothole_detected and len(boxes) > 0:
                notification.notify(
                    title="Alert",
                    message="A pothole has been detected!",
                    timeout=2
                )
                first_pothole_detected = True

            # Extract GPS from frame
            gps = extract_gps_from_frame(frame)
            if gps:
                for (x1, y1, x2, y2), c in zip(boxes, confs):
                    pothole_coordinates.append({
                    "lat": gps["lat"],
                    "lng": gps["lng"],
                    "confidence": round(float(c), 2),
                    "box": [int(x1), int(y1), int(x2), int(y2)]
        })


            for (x1, y1, x2, y2), c in zip(boxes, confs):
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"Pothole:{c:.2f}",
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255,255,255), 2)

        out.write(frame)

    cap.release()
    out.release()
    logging.info("✅ Finished writing %s", output_path)
    logging.info(f"Pothole coords collected: {pothole_coordinates}")


    notification.notify(
        title="Processing Complete",
        message="Video has been processed. All potholes detected.",
        timeout=4
    )


# -------- Run --------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
