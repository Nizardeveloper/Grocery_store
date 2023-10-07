import cv2

# rtsp://admin:CCTV@123@192.168.29.2:554/Streaming/channels/001   - global vision
# rtsp://admin:CCTV@123@192.168.29.2:554/Streaming/channels/102   - Cam1 
# Cam2 = 202, cam3 = 302, cam4 = 402
video = cv2.VideoCapture("http://192.168.29.204:8080")

fps = video.get(cv2.CAP_PROP_FPS)
print(fps)
_,frame = video.read()
f = []
while True:

    f.append(frame)
    # frame = cv2.resize(frame,(1200,700))
    

    cv2.imshow("Live Stream", f)
    if cv2.waitKey(20) == 27:
        break