from Person_Analysis import *
import cv2



Person_detection = Customer_Analysis(r"d:\JSN\Grocery_store\Work_area\model\yolov8n.pt")
Bag_detection = Customer_Analysis(r"d:\JSN\Grocery_store\Work_area\model\Bag_Person_100n.pt")
Face_detection = Customer_Analysis(r"d:\JSN\Grocery_store\Work_area\model\yolov8n-face.pt")
Gender_classification = Customer_Analysis(r"d:\JSN\Grocery_store\Work_area\model\Gender_Classification.pt")


def Retail_store_customer_analysis():
        pass
            