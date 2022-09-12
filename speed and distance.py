import cv2

# This is the distance from camera to face object
DECLARED_LEN = 30  # cm
# width of the object face
DECLARED_WID = 14.3  # cm
# Definition of the RGB Colors format
GREEN = (0, 255, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)
# Defining the fonts family, size, type
fonts = cv2.FONT_HERSHEY_COMPLEX
# calling the haarcascade_frontalface_default.xml module for face detection.
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


def face_data(image):
    face_width = 0
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# We use 1.3 for less powerful processors but can increase it according to your processing power of your machine.
    faces = face_detector.detectMultiScale(gray_image, 1.3, 5)
# getting the rectangular frame
    for (x, y, h, w) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), WHITE, 1)
        face_width = w

    return face_width
# We use 0 in the VideoCapture function since that calls the default camera, the webcam.
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret:
        face_width_in_frame = face_data(frame)
        cv2.imshow("frame", frame)
    # The string 'q' is will be used for stopping and quiting
        if cv2.waitKey(1) == ord("q"):
            break
cap.release()
cv2.destroyAllWindows()