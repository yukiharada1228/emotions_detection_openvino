import logging
import sys

import cv2 as cv
import face_detection_openvino.face_detection.config
import emotions_detection.config

logger = logging.getLogger(__name__)


INDEX_X = 1
INDEX_Y = 0
BLUE = (255, 0, 0)
RATE_X = 0.05
RATE_Y = 0.1
FLIPCODE_HORIZONTAL = 1
KEYCODE_ESC = 27

if len(sys.argv) == 3:
    num, conf = int(sys.argv[1]), float(sys.argv[2])
elif len(sys.argv) == 2:
    num, conf = int(sys.argv[1]), 0.2
else:
    num, conf = 1, 0.2
logger.debug({"num": num, "conf": conf})
capture = cv.VideoCapture(face_detection_openvino.face_detection.config.CAMERA_PORT)
face_detect = face_detection_openvino.face_detection.config.face_detect
emotions_recognition = emotions_detection.config.emotions_recognition
try:
    while capture.isOpened():
        _, frame = capture.read()
        frame = cv.flip(frame, FLIPCODE_HORIZONTAL)
        cv.putText(
            frame,
            "Exit with Esc",
            (
                int(RATE_X * frame.shape[INDEX_X]),
                int(RATE_Y * frame.shape[INDEX_Y]),
            ),
            cv.FONT_HERSHEY_PLAIN,
            2,
            BLUE,
        )
        faces = face_detect.detect(frame, num=num)
        emotions_recognition.recognize(frame, faces)
        output_frame = emotions_recognition.draw(conf=conf)
        cv.imshow("output_frame", output_frame)
        logger.debug({"frame.shape": frame.shape, "faces": faces})
        key = cv.waitKey(1)
        if key == KEYCODE_ESC:
            raise (KeyboardInterrupt)
except KeyboardInterrupt as ex:
    logger.warning({"ex": ex})
finally:
    capture.release()
    cv.destroyAllWindows()