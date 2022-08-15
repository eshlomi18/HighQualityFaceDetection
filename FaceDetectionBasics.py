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


cap = cv.VideoCapture("videos/1.mp4")
pTime = 0
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.75)
while True:
    success, img = cap.read()
    img = rescale_frame(img)
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    print(results)

    if results.detections:
        for id, detection in enumerate(results.detections):
            # bounding box over the face and points on the nose eyes ears and mouth
            # mpDraw.draw_detection(img, detection)

            # print(id, detection)
            # print(detection.score)
            print(detection.location_data.relative_bounding_box)
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            cv.rectangle(img, bbox, (255, 0, 255), 2)
            cv.putText(img, f"{int(detection.score[0] * 100)}%", (bbox[0], bbox[1] - 20), cv.FONT_HERSHEY_PLAIN, 2,
                       (255, 0, 255), 2)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv.putText(img, f"FPS: {int(fps)}", (10, 420), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
    cv.imshow("image", img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break


def code():
    import cv2
    import mediapipe as mp
    import time

    cap = cv2.VideoCapture("videos/girl.mp4")
    pTime = 0

    mpFaceDetection = mp.solutions.face_detection
    mpDraw = mp.solutions.drawing_utils
    faceDetection = mpFaceDetection.FaceDetection(0.75)

    while True:
        success, img = cap.read()

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = faceDetection.process(imgRGB)
        print(results)

        if results.detections:
            for id, detection in enumerate(results.detections):
                # mpDraw.draw_detection(img, detection)
                # print(id, detection)
                # print(detection.score)
                # print(detection.location_data.relative_bounding_box)
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(img, bbox, (255, 0, 255), 2)
                cv2.putText(img, f'{int(detection.score[0] * 100)}%',
                            (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                            2, (255, 0, 255), 2)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (0, 255, 0), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


def facedetection_with_points():
    import cv2
    import mediapipe as mp
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    # For static images:
    IMAGE_FILES = []
    with mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5) as face_detection:
        for idx, file in enumerate(IMAGE_FILES):
            image = cv2.imread(file)
            # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
            results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Draw face detections of each face.
            if not results.detections:
                continue
            annotated_image = image.copy()
            for detection in results.detections:
                print('Nose tip:')
                print(mp_face_detection.get_key_point(
                    detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
                mp_drawing.draw_detection(annotated_image, detection)
            cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)

    # For webcam input:
    cap = cv2.VideoCapture(0)
    with mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5) as face_detection:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(image)

            # Draw the face detection annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(image, detection)
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == ord("q"):
                break
    cap.release()
