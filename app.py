from flask import Flask
import cv2

app = Flask(__name__)

@app.route("/")
def analyze():
    video_path = "sample_footage1.mp4"
    cap = cv2.VideoCapture(video_path)
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
    app.run(host="0.0.0.0", port=10000)
