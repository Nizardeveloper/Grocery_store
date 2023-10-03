from ultralytics import YOLO
import cv2 as cv
import numpy as np

modelp=YOLO("D:\JSN\Grocery_store\Work_area\model\BFP_grocery_100n.pt")
modelf=YOLO("D:\JSN\Grocery_store\Work_area\model\BFP_grocery_100n.pt")
modelc=YOLO("D:\JSN\Grocery_store\Work_area\model\Gender_Classification.pt")

# rtsp://admin:CCTV@123@192.168.29.2:554/Streaming/channels/001
vid=cv.VideoCapture("rtsp://admin:CCTV@123@192.168.29.2:554/Streaming/channels/202")

fps=vid.get(cv.CAP_PROP_FPS)
fcount=0
while True:
    ret , frame = vid.read()
    if not ret:
        break
    p=modelp(frame)
    person=p[0].boxes.data.cpu().tolist()
    for per in  person:
        xm1,ym1,xmx1,ymx1= int(per[0]),int(per[1]),int(per[2]),int(per[3])
        
        cv.rectangle(frame,(xm1,ym1),(xmx1,ymx1),(0,255,0),2)
    r=modelf(frame)
    
    faces=r[0].boxes.data.cpu().tolist()
    for face in  faces:
        xm,ym,xmx,ymx= int(face[0]),int(face[1]),int(face[2]),int(face[3])
        
        cv.rectangle(frame,(xm,ym),(xmx,ymx),(0,255,0),2)
        crop=frame[ym:ymx,xm:xmx]
        gen=modelc(crop)
        index=gen[0].probs.top5[0]
        val=gen[0].names[index]
        cv.putText(frame,str(val),(xm,ym-10),cv.FONT_HERSHEY_PLAIN,1,(0,0,255),1)
    cv.imshow('hj',frame)
    d=int(1000/fps)  
    k=cv.waitKey(d)
    if k!=-1:
        break
print(r[0].boxes.data.cpu().tolist())
cv.destroyAllWindows()