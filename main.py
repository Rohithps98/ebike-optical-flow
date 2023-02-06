import cv2
import numpy as np
import time

def estimate_speed(frame, previous_frame, object_bbox):
    # Calculate the number of pixels moved
    x1, y1, x2, y2 = object_bbox
    object_center = (int((x1+x2)/2), int((y1+y2)/2))
    prev_x, prev_y = previous_frame[object_center[1], object_center[0]]
    curr_x, curr_y = frame[object_center[1], object_center[0]]
    displacement = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)

    # Calculate the time elapsed between frames
    current_time = time.time()
    elapsed_time = current_time - prev_time

    # Convert displacement to km/h
    speed = (displacement * fps * 3.6) / 10

    return speed, elapsed_time

# Load video file
cap = cv2.VideoCapture("/Users/rohith/Desktop/thesis/videos/video.mov")

# Get video properties
fps = 30

# Define the object bounding box (bbox)
object_bbox = (x1, y1, x2, y2)

# Initialize the previous frame
prev_frame = None
prev_time = time.time()

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_frame is None:
        prev_frame = gray
        continue

    # Estimate the speed of the object
    speed, elapsed_time = estimate_speed(gray, prev_frame, object_bbox)
    print("Speed: {:.2f} km/h".format(speed))

    prev_frame = gray
    prev_time = time.time()

# Release the video file
cap.release()
