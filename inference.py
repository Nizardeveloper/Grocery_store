from ultralytics import YOLO
import cv2 as cv

import numpy as np

modelp=YOLO(r'c:\Users\india\Downloads\cus_best.pt')
modelf=YOLO(r'Face_detection.pt')
modelc=YOLO(r'D:\First_project\Grocery\Gender_Classification.pt')
#cm=
vid = cv.VideoCapture(r'c:\Users\india\Downloads\WhatsApp Video 2023-09-25 at 4.00.44 PM.mp4')

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
        name=p[0].names[int(per[-1])]
        if name!='Face':
            xm1,ym1,xmx1,ymx1= int(per[0]),int(per[1]),int(per[2]),int(per[3])
            name=p[0].names[int(per[-1])]
            cv.putText(frame,str(name),(xm1,ym1-10),cv.FONT_HERSHEY_PLAIN,1,(0,0,255),1)

            cv.rectangle(frame,(xm1,ym1),(xmx1,ymx1),(0,255,0),2)
    r=modelf(frame)  
    
    faces=r[0].boxes.data.cpu().tolist()
    for face in  faces:
        xm,ym,xmx,ymx= int(face[0]),int(face[1]),int(face[2]),int(face[3])
        
        cv.rectangle(frame,(xm,ym),(xmx,ymx),(0,255,0),2)
        # names=r[0].names[int(face[-1])]
        # cv.putText(frame,str(names),(xmx,ymx+10),cv.FONT_HERSHEY_PLAIN,1,(0,0,255),1)

        crop=frame[ym:ymx,xm:xmx]
        gen=modelc(crop)
        index=gen[0].probs.top5[0]
        val=gen[0].names[index]
        cv.putText(frame,str(val),(xmx,ymx+10),cv.FONT_HERSHEY_PLAIN,1,(0,0,255),1)
    cv.imshow('hj',frame)
    d=int(1000/fps)  
    k=cv.waitKey(1)
    if k!=-1:
        break
cv.destroyAllWindows()