import cv2 as cv
import mediapipe as mp
import time
from HelpFunctions import rescale_frame
import os


def make_1080p(cap):
    cap.set(3, 1920)
    cap.set(4, 1080)


def make_720p(cap):
    cap.set(3, 1280)
    cap.set(4, 720)


def make_480p(cap):
    cap.set(3, 640)
    cap.set(4, 480)


def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)


class FaceDetector():
    def __init__(self, MinDetectionCon=0.5):

        self.MinDetectionCon = MinDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.MinDetectionCon)

    def FindFace(self, img, drew=True):

        img = rescale_frame(img)
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        # print(self.results)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                print(detection.location_data.relative_bounding_box)
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score])

                cv.rectangle(img, bbox, (255, 0, 255), 2)
                cv.putText(img, f"{int(detection.score[0] * 100)}%", (bbox[0], bbox[1] - 20), cv.FONT_HERSHEY_PLAIN, 2,
                           (255, 0, 255), 2)
        return img, bboxs


def main():
    cap = cv.VideoCapture("videos/1.mp4")
    pTime = 0
    detector = FaceDetector()
    while True:
        success, img = cap.read()
        img, bboxs = detector.FindFace(img)
        print(bboxs)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv.putText(img, f"FPS: {int(fps)}", (10, 420), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
        cv.imshow("image", img)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
