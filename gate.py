from Retail_strore_customer_analysis import *


cam_path = r"d:\JSN\Grocery_store\Work_area\testing\Demo_testing.mp4"


# ROI area points
area1=[(421,275),(449,278),(414,479),(365,479)]
area2=[(475,273),(507,273),(515,479),(471,479)]


#Custom models
Person_detection_model = r"yolov8n_person.pt"
Bag_detection_model = r"Bag_100n.pt"


# Database API
db_url = "https://vchekservicedemoapp.azurewebsites.net/v1/PersonAnalytics"
db_headers = {
           'apikey': '415d1f410a424a4ba0e6925991db57b2',
           'Content-Type': 'application/json'
           }

# Custom vision Gender classification  API
cv_url = "https://genderfinderpoc-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/4850fbc3-49fb-43e0-b22a-98766dbffbd7/classify/iterations/Iteration1/image"
cv_headers = {
            'Prediction-Key': 'b4dd23c416bb4d439e5974dfe308f47d',
            'Content-Type': 'application/octet-stream'
            }


Analysis_models = Customer_Analysis(Person_detection_model,Bag_detection_model,db_url,db_headers,cv_url,cv_headers)
Analysis_models.Analysis(cam_path,area1,area2)


