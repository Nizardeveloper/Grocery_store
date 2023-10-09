import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import datetime
import requests
import json

class Customer_Analysis:
    def __init__(self, model_person, model_bag, model_face, model_gender,Api_url, headers):
        self.person = YOLO(model_person)
        self.bag = YOLO(model_bag)
        self.face = YOLO(model_face)
        self.gender = YOLO(model_gender)
        self.url = Api_url
        self.headers = headers

    def response(self,data):
        payload = json.dumps(data)
        requests.request("POST", self.url, headers=self.headers, data=payload)


    def Analysis(self,cam_url, debug=False):
        object_tracker = DeepSort(max_age=10,
                n_init=2,
                nms_max_overlap=0.7,
                max_cosine_distance=0.3,
                nn_budget=None,
                override_track_class=None,
                embedder="mobilenet",
                half=True,
                bgr=True,
                embedder_gpu=True,
                embedder_model_name=None,
                embedder_wts=None,
                polygon=False,
                today=None)
        

        area1=[(397,331),(423,328),(382,479),(351,479)]
        area2=[(451,317),(484,315),(477,479),(428,479)]
        cap = cv2.VideoCapture(cam_url)
        class_list = ['Bag','Face','Person']

        cus_in=1
        cus_out=1
        male_in=1
        female_in=1
        pur_val=1
        male_count=set()
        female_count=set()
        purchase_count=set()
        people_entering={}
        people_exit={}
        exiting=set()
        entering=set()
        

        while True:   
            
            out={
              "tenantId": 5,
              "siteId": 26,
              "siteCode": "077",
              "siteUserId": 128,
              "data":{
                    "gender": "Null",
                    "isBagAvailable": 0,
                    "isPersonIn": 1,
                    "detectedDateTime": ""
                      }
                }

            ret,frame = cap.read()
            if not ret:
                break
            frame_copy=frame.copy()
  
            results_person=self.person.predict(frame_copy)
            person=results_person[0].boxes.data.cpu().tolist()
            predicted_list=[]

            if  person!=[]:
                for val in person:
                    if val[-1]==0.0:
                        val[-1]=2.0
                        predicted_list.append(val)

            list=[]
            for predictions in predicted_list:
                pxm=int(predictions[0])
                pym=int(predictions[1])
                pxmx=int(predictions[2]) 
                pymx=int(predictions[3])
                d=int(predictions[5])
                if d==2:
                    list.append(([pxm,pym,pxmx-pxm,pymx-pym],predictions[4],class_list[d]))

            tracks = object_tracker.update_tracks(list, frame=frame_copy) 
            for track in tracks:
                if not track.is_confirmed():
                    continue
                id = track.track_id
                ltrb = track.to_ltrb()
                bbox = ltrb
                txm = int(bbox[0])
                tym = int(bbox[1])
                txmx = int(bbox[2])
                tymx = int(bbox[3])


                    ########ENTERING

                result=cv2.pointPolygonTest(np.array(area1,np.int32),(txm,tymx),False)
                if result>=0:
                    people_entering[id]=(txmx,tymx)
                if id in people_entering:
                    result1=cv2.pointPolygonTest(np.array(area2,np.int32),(txm,tymx),False)
                    if result1>=0:
                        results_face=self.face.predict(frame_copy)
                        faces=results_face[0].boxes.data.cpu().tolist()
                        for face in  faces:
                            fxm,fym,fxmx,fymx= int(face[0]),int(face[1]),int(face[2]),int(face[3])
                            crop=frame_copy[fym:fymx,fxm:fxmx]
                            gender=self.gender.predict(crop)
                            index=gender[0].probs.top5[0]
                            val=gender[0].names[index]
                            people_entering[id]=val
                        entering.add(id)
                        if people_entering[id]=='Male':
                            male='m'+id
                            male_count.add(male)
                        elif people_entering[id]=='Female':
                            female='f'+id
                            female_count.add(female)


                        ####Exiting

                result3=cv2.pointPolygonTest(np.array(area2,np.int32),(txmx,tymx),False)
                if result3>=0:
                    people_exit[id]=(txmx,tymx)
                if id in people_exit:
                    result4=cv2.pointPolygonTest(np.array(area1,np.int32),(txmx,tymx),False)
                    if result4>=0:
                        results_bag=self.bag.predict(frame_copy)
                        bag=results_bag[0].boxes.data.cpu().tolist()
                        for b in  bag:
                            xm,ym,xmx,ymx= int(b[0]),int(b[1]),int(b[2]),int(b[3])
                            area3=[(txm,tym),(txmx,tym),(txmx,tymx),(txm,tymx)]
                            result5=cv2.pointPolygonTest(np.array(area3,np.int32),(xmx,ym),False)
                            if result5>=0:
                                purchase='p'+id
                                purchase_count.add(purchase)
                        exiting.add(id)

            if len(entering)==cus_in:  
                if len(male_count)==male_in:
                    out['data']['gender']="Male"
                    male_in+=1
                elif len(female_count)==female_in:
                    out['data']['gender']="Female"
                    female_in+=1
                out['data']['detectedDateTime']=str(datetime.datetime.now())
                cus_in+=1
                Customer_Analysis.response(self,data=out)


            if len(exiting)==cus_out:
                if result4<=0 and len(exiting)==cus_out:
                    out['data']['isPersonIn']=0
                    if len(purchase_count)==pur_val:
                        out['data']['isBagAvailable']=1
                        pur_val+=1
                    out['data']['detectedDateTime']=str(datetime.datetime.now())
                    cus_out+=1
                    Customer_Analysis.response(self,data=out)










