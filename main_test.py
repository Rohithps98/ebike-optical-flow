# importing the required modules
import time

import cv2
import numpy as np
import argparse
from computeDistance import get_distance_between_points
from hyperparameters import *

writer = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, (IMAGE_H, IMAGE_W))

li = ["car", "truck", "bicycle"]




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
    start = end = 0
    flag = True

    time_diff = 0

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (IMAGE_H, IMAGE_W))
        height, width, chan = frame.shape

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

                if label in li:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    center_x = (2 * x + w) / 2
                    center_y = (2 * y + h) / 2
                    actual_dist = wide * (width / 2) / w

                    if flag is True and int(round(center_x)) in (range(0, 80) or range(400, 480)):
                        start = time.time()
                        flag = False

                    if flag is False and int(round(center_x)) in range(int(round(width / 2)) - 10,
                                                                       int(round(width / 2)) + 10):
                        end = time.time()
                        time_diff = end - start
                        # print("End:",end)
                        flag = True
                        s_flag = True

                if time_diff > 0 and s_flag == True:
                    velocity = actual_dist / time_diff
                    # print(round(start),round(end))
                    vel_kmph = round(velocity * 3.6, 2)
                    ttc = (actual_dist * 3600) / (1000 * vel_kmph)
                    print("Speed:", vel_kmph, "kmph")
                    print("Distance from car:", round(actual_dist, 2), "m")
                    print("Time taking to collide: ", ttc, "secs")
                    s_flag = False
            #     classify = get_distance_between_points(projected, actual_points, threshold_distance)
            #
            #     for bx in classify:
            #         box = bx[0][1]
            #         (x, y) = (box[0], box[1])
            #         (w, h) = (box[2], box[3])
            #
            #         # If True then Red boxes
            #         if bx[1]:
            #             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            #         else:
            #             # Green boxes
            #             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # else:
            #     continue
        # transformed = cv2.warpPerspective(frame, transformation_matrix, (IMAGE_W, IMAGE_H))
        # cv2.imshow("Transformed Image", transformed)  # Show results

        cv2.imshow("Frame", frame)
        if save_output:
            writer.write(frame)
        if cv2.waitKey(1) == 27:
            cap.release()
            writer.release()
            break


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
