from ultralytics import YOLO
import cv2 as cv

import numpy as np

modelp=YOLO('person_detection.pt')
modelf=YOLO('Face_detection.pt')
modelc=YOLO('Gender_Classification.pt')

triangle_pts = np.array([[0,100], [800, 100],[800,200],[0, 200]], dtype=np.int32)
vid=cv.VideoCapture(r'people2.mp4')

fps=vid.get(cv.CAP_PROP_FPS)
fcount=0
while True:
    ret , frame = vid.read()
    if not ret:
        break
    cv.polylines(frame, [triangle_pts], isClosed=True, color=(255, 0, 0), thickness=2)
    p=modelp(frame)
    person=p[0].boxes.data.cpu().tolist()
    for per in  person:
        xm1,ym1,xmx1,ymx1= int(per[0]),int(per[1]),int(per[2]),int(per[3])
        
        cv.rectangle(frame,(xm1,ym1),(xmx1,ymx1),(0,255,0),2)
    r=modelf(frame)
    # face,con= cvl.detect_face(frame)
    # for f , c in zip(face , con):
    #     xm,ym,xmx,ymx= f[0],f[1],f[2],f[3]
    #     cv.rectangle(frame,(xm,ym),(xmx,ymx),(0,255,0),2)
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