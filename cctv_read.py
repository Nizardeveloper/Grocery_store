import cv2

# rtsp://admin:CCTV@123@192.168.29.2:554/Streaming/channels/001   - global vision
# rtsp://admin:CCTV@123@192.168.29.2:554/Streaming/channels/102   - Cam1 
# Cam2 = 202, cam3 = 302, cam4 = 402
video = cv2.VideoCapture("rtsp://admin:CCTV@123@192.168.29.2:554/Streaming/channels/202")

while True:

    _,frame = video.read()

    cv2.imshow("Live Stream", frame)
    if cv2.waitKey(20) == 27:
        break