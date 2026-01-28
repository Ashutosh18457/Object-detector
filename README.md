import cv2
from ultralytics import YOLO
import pyttsx3
from flask import Flask, Response

app = Flask(__name__)

cap = cv2.VideoCapture(0)

# Set video resolution (e.g., 1280x720 for HD)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Set frame rate (e.g., 60 FPS)
cap.set(cv2.CAP_PROP_FPS, 60)

model = YOLO('yolov5s.pt') 

def speak(text):
    """Function to convert text to speech."""
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def generate_frames():
    """Generate video frames for streaming."""
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB for YOLO
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run YOLO model with adjusted confidence and IoU thresholds
        results = model(frame_rgb, conf=0.2, iou=0.45)

        detected = False  # Flag to ensure only one object is processed
        for result in results:
            for box in result.boxes:
                if detected:  # Skip further detections
                    break
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                label = model.names[class_id]

                # Debugging: Print detection details
                print(f"Detected {label} with confidence {confidence:.2f} at {x1, y1, x2, y2}")

                speak(f"Detected {label} with confidence {confidence:.2f}")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                detected = True  # Set the flag to true after detecting one object

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Route to stream video feed."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """Default route to display instructions."""
    return '''
    <h1>Object Detection Stream</h1>
    <p>Open <a href="/video_feed">this link</a> to view the video feed.</p>
    '''

if __name__ == "__main__":
    if not cap.isOpened():
        print("Error: Could not open webcam or video.")
    else:
        print("Camera is now active. Access the stream at http://<your-ip>:5000/video_feed")
    app.run(host='0.0.0.0', port=5000)
    cap.release()
    cv2.destroyAllWindows()
