import cv2 as cv
import mediapipe as mp
import time
from HelpFunctions import rescale_frame
import os


def make_1080p():
    cap.set(3, 1920)
    cap.set(4, 1080)

def make_720p():
    cap.set(3, 1280)
    cap.set(4, 720)

def make_480p():
    cap.set(3, 640)
    cap.set(4, 480)

def change_res(width, height):
    cap.set(3, width)
    cap.set(4, height)


print(os.sep.join(os.path.abspath(__file__).split(os.sep)[:-3]))
cap = cv.VideoCapture("videos/girl.mp4")
pTime = 0
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()
while True:
    success, img = cap.read()
    imgResize = rescale_frame(img)
    imgRGB = cv.cvtColor(imgResize, cv.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    print(results)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv.putText(imgResize, f"FPS: {int(fps)}", (10, 420), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
    cv.imshow("image", imgResize)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break



