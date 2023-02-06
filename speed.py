import cv2
import numpy as np
import time
import argparse
from hyperparameters import *


cap = cv2.VideoCapture("/Users/rohith/Desktop/thesis/videos/video.mov")
# params for Shi Tamasi corner detection

feature_params = dict(maxCorners=100,
                      qualityLevel=0.05,
                      minDistance=50,
                      blockSize=7)
# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))



writer = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, (IMAGE_H, IMAGE_W))

li = ["car", "truck", "bicycle"]

def estimate_speed(frame, prev_frame, object_bbox):
    # Select the object region of interest (ROI)
    x, y, w, h = object_bbox
    roi = prev_frame[y:h, x:w]

    # Convert to grayscale and calculate the optical flow
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowPyrLK(prev_gray, gray, None, **lk_params)

    # Calculate the average flow in the ROI
    average_flow = np.mean(flow[y:h, x:w], axis=(0, 1))

    # Calculate the displacement
    displacement = np.sqrt(average_flow[0] ** 2 + average_flow[1] ** 2)

    # Calculate the time elapsed between frames
    current_time = time.time()
    elapsed_time = current_time - prev_time

    # Convert displacement to km/h
    speed = (displacement * fps * 3.6) / 10

    return speed, elapsed_time

def process_video(video_file,
                  yolo_model,
                  output_layers,
                  classes,
                  save_output):
    # initialize the writer for writing the output video to a file
    if video_file:
        cap = cv2.VideoCapture(video_file)
    else:
        cap = cv2.VideoCapture(0)

    wide = 0.3
    flag = True

    time_diff = 0

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (IMAGE_H, IMAGE_W))
        height, width, chan = frame.shape
        start = time.time()

        # Create some random colors
        color = np.random.randint(0, 255, (100, 3))
        old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        # Create a mask image for drawing purposes
        mask = np.zeros_like(frame)

        # Detecting objects
        blob = cv2.dnn.blobFromImage(frame, SCALE_FACTOR, TRANSFORMED_IMAGE_SIZE, MEAN_VALUES, True, crop=False)
        yolo_model.setInput(blob)  # sets the input to the network as blob

        # run forward pass of the network to compute the outputs of the layers listed in output_layers
        outs = yolo_model.forward(output_layers)  # returns an ndarray(bx, by, bh, bw, pc, c)

        # initialize our lists of detected bounding boxes, confidences, and class IDs, respectively
        boxes = []
        confidences = []
        class_ids = []

        # loop over each of the layer outputs
        for out in outs:
            # loop over each of the detections
            for detection in out:
                # extract the class ID and confidence
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # filter out weak predictions
                if confidence > CONFIDENCE_THRESHOLD:
                    center_x = int(detection[0] * IMAGE_H)
                    center_y = int(detection[1] * IMAGE_W)

                    # width and height
                    w = int(detection[2] * IMAGE_H)
                    h = int(detection[3] * IMAGE_W)

                    # Rectangle coordinates top left
                    x = int(center_x - (w / 2))
                    y = int(center_y - (h / 2))

                    # update our list of bounding box coordinates, confidences, and class IDs
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD,
                                   NMS_THRESHOLD)  # saves the array of indices of best boxes in the frame

        if len(indexes) > 0:
            # loop over the indexes we are keeping
            for i in indexes.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                label = str(classes[class_ids[i]])
                # Define the object bounding box (bbox)
                object_bboxes = (x, y, w, h)

                if label in li:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    speeds = []
                    for object_bbox in object_bboxes:
                        speed, elapsed_time = estimate_speed(frame, prev_frame, object_bbox)
                        speeds.append(speed)

                    # Draw the bounding boxes and speed on the frame
                    for i, object_bbox in enumerate(object_bboxes):
                        x1, y1, x2, y2 = object_bbox
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, "{:.2f} km/h".format(speeds[i]), (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    continue
            # transformed = cv2.warpPerspective(frame, transformation_matrix, (IMAGE_W, IMAGE_H))
            # cv2.imshow("Transformed Image", transformed)  # Show results

            cv2.imshow("Frame", frame)
            if save_output:
                writer.write(frame)
            if cv2.waitKey(1) == 27:
                cap.release()
                writer.release()
                break

# Get video properties
fps = 30



# Initialize the previous frame
prev_frame = None
prev_gray = None
prev_time = time.time()


def load_model():
    # reading a deep learning network from the given config files
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # stores all the unconnected output layers
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in
                     net.getUnconnectedOutLayers()]

    return net, output_layers, classes


def main(args):
    path = args.path
    # Calibrating and defining the threshold distance
    # [transformation_matrix, _, threshold_distance, _] = calibrate(path)

    # Load yolo-v3 pretrained model.
    yolo_model, output_layers, classes = load_model()

    process_video(path, yolo_model, output_layers, classes, args.save_output)


if __name__ == "__main__":
    # CLI support
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Enter the path of Video', required=True)
    parser.add_argument('--save_output', help='Enter the path of Video', action="store_true")
    args = parser.parse_args()
    main(args)
