import cv2

# Load video
video_path = "sample_footage1.mp4"
cap = cv2.VideoCapture(video_path)

# Initialize background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

people_counts = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction
    fgmask = fgbg.apply(frame)

    # Find contours
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    people_count = 0
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:  # filter small noise
            people_count += 1

    people_counts.append(people_count)

cap.release()

# Print summary result (you can also log or save this)
average_count = sum(people_counts) / len(people_counts) if people_counts else 0
print(f"Average People Detected: {average_count:.2f}")
