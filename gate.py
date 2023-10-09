from Retail_strore_customer_analysis import *

cam_path = r"c:\Users\NISAR\Downloads\WhatsApp Video 2023-10-09 at 12.51.35 PM.mp4"

url = "https://vchekservicedemoapp.azurewebsites.net/v1/PersonAnalytics"

headers = {
           'apikey': '415d1f410a424a4ba0e6925991db57b2',
           'Content-Type': 'application/json'
           }


Person_detection_model = r"yolov8n_person.pt"
Bag_detection_model = r"Bag_100n.pt"
Face_detection_model = r"yolov8n-face.pt"
Gender_classification_model = r"Gender_Classification.pt"


Analysis_models = Customer_Analysis(Person_detection_model,Bag_detection_model,Face_detection_model,Gender_classification_model,url,headers)


Analysis_models.Analysis(cam_path)


