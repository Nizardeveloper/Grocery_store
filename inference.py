from ultralytics import YOLO
import cv2 as cv
import numpy as np

modelp=YOLO(r"D:\JSN\Grocery_store\Work_area\model\BFP_grocery_100n.pt")
modelc=YOLO(r"D:\JSN\Grocery_store\Work_area\model\Gender_Classification.pt")

vid=cv.VideoCapture("rtsp://admin:CCTV@123@192.168.29.2:554/Streaming/channels/202")

fps=vid.get(cv.CAP_PROP_FPS)
fcount=0
while True:
    ret , frame = vid.read()
    if not ret:
        break
    p=modelp(frame)
    print(p[0].names[0])
    person=p[0].boxes.data.cpu().tolist()
    print(person)
    for per in  person:
        xm1,ym1,xmx1,ymx1= int(per[0]),int(per[1]),int(per[2]),int(per[3])
        name=p[0].names[int(per[-1])]
        cv.putText(frame,str(name),(xm1,ym1-10),cv.FONT_HERSHEY_PLAIN,1,(0,0,255),1)

        cv.rectangle(frame,(xm1,ym1),(xmx1,ymx1),(0,255,0),2)

    cv.imshow('hj',frame)
    d=int(1000/fps)  
    k=cv.waitKey(1)
    if k!=-1:
        break
cv.destroyAllWindows()