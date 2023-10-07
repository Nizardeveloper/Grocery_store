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


        return {"tenantId": 5,"siteId": 26,"siteCode": "077","siteUserId": 128,
            "data":{
                    "gender": "Female",
                    "isBagAvailable": 0,
                    "isPersonIn": 1,
                    "detectedDateTime": "2023-10-05 15:35:20"
                    }
                }

    
