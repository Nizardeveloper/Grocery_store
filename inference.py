from ultralytics import YOLO
import cv2
import numpy as np

person = YOLO("d:\JSN\Grocery_store\Work_area\model\yolov8n.pt")
bag = YOLO("d:\JSN\Grocery_store\Work_area\model\Bag_Person_100n.pt")
face = YOLO("d:\JSN\Grocery_store\Work_area\model\yolov8n-face.pt")
gender = YOLO("d:\JSN\Grocery_store\Work_area\model\Gender_Classification.pt")

