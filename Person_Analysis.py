from ultralytics import YOLO
import cv2
import numpy as np
import datetime



class Customer_Analysis:
    def __init__(self, model_person, model_bag, model_face, model_gender):
        self.person = YOLO(model_person)
        self.bag = YOLO(model_bag)
        self.face = YOLO(model_face)
        self.gender = YOLO(model_gender)


    def Analysis(self,cam_url, debug=False):

        # video = cv2.VideoCapture(cam_url)

        # while True:
        #     _, frame = video.read()

        #     cv2.imshow("svf", frame)
        #     if cv2.waitKey(20) == 27:
        #         break
        # cv2.destroyAllWindows()
        return {'gender': 'Null', 'isBagAvailable': 0, 'isPersonIn': 0, 'detectedDateTime': '2023-10-06 21:08:53.223773'}

    
