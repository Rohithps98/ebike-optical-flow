import numpy as np
import cv2 as cv

# parser = argparse.ArgumentParser(description='This sample demonstrates Lucas-Kanade Optical Flow calculation. \
#                                               The example file can be downloaded from: \
#                                               Users/rohith/Desktop/thesis/videos/video_short.mp4')
# parser.add_argument('image', type=str, help='path to image file')
# args = parser.parse_args()
cap = cv.VideoCapture("/Users/rohith/Desktop/thesis/videos/video.mov")
# params for Shi Tamasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.05,
                      minDistance=50,
                      blockSize=7)
# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))
# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

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
            old_gray_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # Use the Lucas-Kanade method `cv.calcOpticalFlowPyrLK` for Optical Flow
            # Indices of the `status` array which equal 1 signify a corresponding new feature has been found
            p1, status, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, corners0, None, **lk_params)

            return p1, status == 1
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]
        #draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

        p1, valid = optical_flow(old_frame, frame, p0)
        velocity = ((p1 - p0)[valid == 1]).reshape(-1, 2)

        u, v = velocity[:, 0], velocity[:, 1]

        mean_u = alpha * mean_u + (1 - alpha) * np.mean(u)
        mean_v = alpha * mean_v + (1 - alpha) * np.mean(v)
        vel = ((mean_u**2)+(mean_v**2))**0.5
        print(vel, "mph")
        velocity_1 = vel*1.60934*3.6
        print(velocity_1, "m/s")

        cv.putText(frame, "Velocity (U) = {0:.2f}".format(velocity_1),
                   (20, 20), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), thickness=2, lineType=cv.LINE_AA)

        img = cv.add(frame, mask)
        cv.imshow('frame', img)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
    else:
        print('No frames grabbed!')
        break

cv.destroyAllWindows()
