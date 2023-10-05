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


    def Analysis(self,frames, debug=False):
        result = self.person.predict(source=frames, show=True)
        print(result)

    
