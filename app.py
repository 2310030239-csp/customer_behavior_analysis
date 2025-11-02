from flask import Flask
import cv2
import os

app = Flask(__name__)

@app.route("/")
def analyze():
    video_path = "sample_footage1.mp4"  # Make sure this file is in your repo
    if not os.path.exists(video_path):
        return "<h1>Error: Video file not found!</h1>"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "<h1>Error: Cannot open video file!</h1>"

    fgbg = cv2.createBackgroundSubtractorMOG2()
    people_counts = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fgmask = fgbg.apply(frame)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        people_count = sum(1 for c in contours if cv2.contourArea(c) > 500)
        people_counts.append(people_count)

    cap.release()

    avg = sum(people_counts) / len(people_counts) if people_counts else 0
    return f"<h1>Average People Detected: {avg:.2f}</h1>"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Use Render's PORT
    app.run(host="0.0.0.0", port=port)
