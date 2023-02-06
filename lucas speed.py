# Get video properties
fps = 30

# Define the object bounding box (bbox)
object_bbox = (x1, y1, x2, y2)

# Initialize the previous frame
prev_frame = None
prev_gray = None
prev_time = time.time()

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    if prev_frame is None:
        prev_frame = frame
        prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        continue

    # Estimate the speed of the object
    speed, elapsed_time = estimate_speed(frame, prev_frame, object_bbox)
    print("Speed: {:.2f} km/h".format(speed))

    prev_frame = frame
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_time = time.time()

# Release the video file
cap.release()



import numpy as np
import cv2
import time
import argparse

# parser = argparse.ArgumentParser(description='This sample demonstrates Lucas-Kanade Optical Flow calculation. \
#                                               The example file can be downloaded from: \
#                                               Users/rohith/Desktop/thesis/videos/video_short.mp4')
# parser.add_argument('image', type=str, help='path to image file')
# args = parser.parse_args()


def estimate_speed(frame, previous_frame, object_bbox):
    # Select the object region of interest (ROI)
    x1, y1, x2, y2 = object_bbox
    roi = previous_frame[y1:y2, x1:x2]

    # Convert to grayscale and calculate the optical flow
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Calculate the average flow in the ROI
    average_flow = np.mean(flow[y1:y2, x1:x2], axis=(0, 1))

    # Calculate the displacement
    displacement = np.sqrt(average_flow[0]**2 + average_flow[1]**2)

    # Calculate the time elapsed between frames
    current_time = time.time()
    elapsed_time = current_time - prev_time

    # Convert displacement to km/h
    speed = (displacement * fps * 3.6) / 10

    return speed, elapsed_time

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

# Create some random colors
color = np.random.randint(0, 255, (100, 3))
# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)


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
    flag = True

    time_diff = 0

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (IMAGE_H, IMAGE_W))
        height, width, chan = frame.shape
        start = time.time()

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













while True:
    ret, frame = cap.read()
    mean_u = 0
    mean_v = 0
    alpha = 0.97


    # def optical_flow(old_frame, frame, corners0):
    #     # convert images to grayscale
    #     old_gray_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    #     frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #
    #     # Use the Lucas-Kanade method `cv.calcOpticalFlowPyrLK` for Optical Flow
    #     # Indices of the `status` array which equal 1 signify a corresponding new feature has been found
    #     p1, status, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, corners0, None, **lk_params)
    #
    #     return p1, status == 1
    if ret:
        def optical_flow(old_frame, frame, corners0):
            # convert images to grayscale
            old_gray_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Use the Lucas-Kanade method `cv.calcOpticalFlowPyrLK` for Optical Flow
            # Indices of the `status` array which equal 1 signify a corresponding new feature has been found
            p1, status, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, corners0, None, **lk_params)

            return p1, status == 1
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]
        #draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

        p1, valid = optical_flow(old_frame, frame, p0)
        velocity = ((p1 - p0)[valid == 1]).reshape(-1, 2)

        u, v = velocity[:, 0], velocity[:, 1]

        mean_u = alpha * mean_u + (1 - alpha) * np.mean(u)
        mean_v = alpha * mean_v + (1 - alpha) * np.mean(v)
        vel = ((mean_u**2)+(mean_v**2))**0.5
        print(vel, "mph")
        velocity_1 = vel*1.60934*3.6
        print(velocity_1, "m/s")

        cv2.putText(frame, "Velocity (U) = {0:.2f}".format(velocity_1),
                   (20, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

        img = cv2.add(frame, mask)
        cv2.imshow('frame', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
    else:
        print('No frames grabbed!')
        break

cv2.destroyAllWindows()
