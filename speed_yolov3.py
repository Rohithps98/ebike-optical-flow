import cv2
import numpy as np
import time

import computeDistance
from computeDistance import get_distance_between_points

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))


def process_video(video_file,
                  yolo_model,
                  output_layers,
                  classes,
                  transformation_matrix,
                  threshold_distance,
                  save_output, CONFIDENCE_THRESHOLD=0.7, NMS_THRESHOLD=0.2, SCALE_FACTOR=0.00392,
                  TRANSFORMED_IMAGE_SIZE=(416, 416), MEAN_VALUES=(0, 0, 0), projected=None):
    video = cv2.VideoCapture("/Users/rohith/Desktop/thesis/videos/video.mov")
    wide = 0.3
    Flag = True
    start = end = 0
    time_diff = 0

    while video.isOpened():
        ret, img = video.read()
        if ret:
            height, width, chan = img.shape

            # Detecting objects
            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            # Showing informations on the screen
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > CONFIDENCE_THRESHOLD:
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

            list_of_obstacles = ['person', 'bicycle', 'car', 'motorbike', 'bus', 'train', 'truck', 'traffic light',
                                 'fire hydrant', 'stop sign',
                                 'parking meter', 'bench', 'parking gate', 'bird', 'cat', 'dog', 'dustbin', 'wall',
                                 'shopping cart', 'skateboard', 'e-bike', 'tree', 'dumpster']

            if len(indexes) > 0:
                # loop over the indexes we are keeping
                for i in indexes.flatten():
                    # extract the bounding box coordinates
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    label = str(classes[class_ids[i]])

                    if label in list_of_obstacles:
                        # p=p+1
                        projected.append([[x + w // 2, y + h], boxes[i]])
                        classify = get_distance_between_points(projected, transformation_matrix, threshold_distance)

                        for bx in classify:
                            box = bx[0][1]
                            (x, y) = (box[0], box[1])
                            (w, h) = (box[2], box[3])

                        # If True then Red boxes
                        if bx[1]:
                            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        else:
                            # Green boxes
                            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    if flag is True and int(round(center_x)) in (range(0, 80) or range(400, 480)):
                        start = time.time()
                        flag = False

                        # print("Start:",start)

                    if flag is False and int(round(center_x)) in range(int(round(width / 2)) - 10,
                                                                       int(round(width / 2)) + 10):
                        end = time.time()
                        time_diff = end - start
                        # print("End:",end)
                        flag = True
                        s_flag = True

                # print("Time Difference:",time_diff)
                if time_diff > 0 and s_flag == True:
                    velocity = computeDistance.get_distance_between_points.actual_dist / time_diff
                    # print(round(start),round(end))
                    vel_kmph = round(velocity * 3.6, 2)
                    ttc = (computeDistance.get_distance_between_points.actual_dist * 3600) / (1000 * vel_kmph)
                    print("Speed:", vel_kmph, "kmph")
                    print("Distance from car:", round(computeDistance.get_distance_between_points.actual_dist, 2), "m")
                    print("Time taking to collide: ", ttc, "secs")
                    s_flag = False

                cv2.line(img, (int(width / 2), 0), (int(width / 2), height), (255, 0, 0), 2)
                cv2.imshow('frame', img)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


            else:
                continue

        else:
            print("No frames observed")

        video.release()
        cv2.destroyAllWindows()