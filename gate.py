from Person_Analysis import *
import cv2


cam_footage = r"d:\JSN\Grocery_store\Work_area\testing\samp1.mp4"



Person_detection_model = r"d:\JSN\Grocery_store\Work_area\model\yolov8n.pt"
Bag_detection_model = r"d:\JSN\Grocery_store\Work_area\model\Bag_Person_100n.pt"
Face_detection_model = r"d:\JSN\Grocery_store\Work_area\model\yolov8n-face.pt"
Gender_classification_model = r"d:\JSN\Grocery_store\Work_area\model\Gender_Classification.pt"


Analysis_models = Customer_Analysis(Person_detection_model,Bag_detection_model,Face_detection_model,Gender_classification_model)

        
def Retail_store_customer_analysis():

        cam = cv2.VideoCapture(cam_footage)
        while True:
            _, frames = cam.read()

            frame = Analysis_models.Analysis(frames)
            
Retail_store_customer_analysis()