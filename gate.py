from Person_Analysis import *
import cv2
from flask import Flask, request
import requests
import json


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
        Json_outputs = Analysis_models.Analysis(data["cam"])
        # return Json_outputs
    
        url = "https://vchekservicedemoapp.azurewebsites.net/v1/PersonAnalytics"

        headers = {
                    'apikey': '415d1f410a424a4ba0e6925991db57b2',
                    'Content-Type': 'application/json',
                    'Cookie': 'ARRAffinity=79e06db539acb57119e709978d2cf1da299e8341753d6f6345007fcab3f69bc5; ARRAffinitySameSite=79e06db539acb57119e709978d2cf1da299e8341753d6f6345007fcab3f69bc5'
                    }
        
        payload = json.dumps(Json_outputs)
        response = requests.request("POST", url, headers=headers, data=payload)


