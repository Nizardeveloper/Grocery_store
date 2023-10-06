from Person_Analysis import *
import cv2
from flask import Flask, request
import requests


Person_detection_model = r"d:\JSN\Grocery_store\Work_area\model\yolov8n.pt"
Bag_detection_model = r"d:\JSN\Grocery_store\Work_area\model\Bag_Person_100n.pt"
Face_detection_model = r"d:\JSN\Grocery_store\Work_area\model\yolov8n-face.pt"
Gender_classification_model = r"d:\JSN\Grocery_store\Work_area\model\Gender_Classification.pt"


Analysis_models = Customer_Analysis(Person_detection_model,Bag_detection_model,Face_detection_model,Gender_classification_model)

app = Flask(__name__)

@app.route("/Retail_Store_Customer_Analysis", methods=["POST"])   
def Retail_store_customer_analysis():
    data = request.get_json(force=True)
    
    if data["cam"] != "":
        try:
            cam_footage = cv2.VideoCapture(data["cam"])
            


        
        except:
            return "Error Occured"
    
    else:
        return "No Camera Footage Found"
            

