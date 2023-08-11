from ultralytics import YOLO
import cv2

import util
from sort.sort import *
from util import get_car, read_license_plate,write_csv
import easyocr

# sort() object
results = {}
mort_tracker = Sort()
# loading the model

frame_nmr = -1
coco_model = YOLO('models/yolov8n.pt')  # for detecting the cars
license_plate_detector = YOLO('models/license_plate_detector.pt')
# loading the video

cap = cv2.VideoCapture('sample.mp4')
vehicles = [2, 3, 5, 7]  # class id for vehicles in coco model

# reading frames

ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        # detect the vehicles

        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # object tracking {vehicles}

        track_ids = mort_tracker.update(np.asarray(detections_))

        # detect the licence plate

        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # assign license plate to a given car

            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            # croping the number plate

            license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

            # applying filters to make the filter for easy ocr
            # making the image gray scaled

            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255,
                                                         cv2.THRESH_BINARY_INV)  # inverting and then making the image completely black and white

            # read license plate number
            license_plate_text, license_plate_text_score = util.read_license_plate(license_plate_crop_thresh)

            if license_plate_text is not None:
                results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                              'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                'text': license_plate_text,
                                                                'bbox_score': score,
                                                                'text_score': license_plate_text_score}}

        # write results
        write_csv(results, './test.csv')
