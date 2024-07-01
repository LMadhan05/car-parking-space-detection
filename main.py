import time

import cv2
import numpy as np
import pandas as pd

from ultralytics import YOLO, solutions

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("parking1.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Video writer
video_writer = cv2.VideoWriter("parking_space_detection.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (1020, 500))

with open("className" , "r") as file:
    class_list = file.read().split("\n")

areas = [
    [(52, 364), (30, 417), (73, 412), (88, 369)],
    [(105, 353), (86, 428), (137, 427), (146, 358)],
    [(159, 354), (150, 427), (204, 425), (203, 353)],
    [(217, 352), (219, 422), (273, 418), (261, 347)],
    [(274, 345), (286, 417), (338, 415), (321, 345)],
    [(336, 343), (357, 410), (409, 408), (382, 340)],
    [(396, 338), (426, 404), (479, 399), (439, 334)],
    [(458, 333), (494, 397), (543, 390), (495, 330)],
    [(511, 327), (557, 388), (603, 383), (549, 324)],
    [(564, 323), (615, 381), (654, 372), (596, 315)],
    [(616, 316), (666, 369), (703, 363), (642, 312)],
    [(674, 311), (730, 360), (764, 355), (707, 308)]
]



while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    # time.sleep(1)
    im0 = cv2.resize(im0 , (1020,500))
    results = model.predict(im0)
    detections = results[0].boxes.data
    px = pd.DataFrame(detections).astype("float")

    occupancy = [0] * len(areas)

    for _, row in px.iterrows():
        x1, y1, x2, y2, _, class_id = map(int,row)
        class_name = class_list[class_id]

        if 'car' in class_name:
            cx, cy = (x1+x2) // 2, (y1+y2) // 2
            for idx , area in enumerate(areas):
                if cv2.pointPolygonTest(np.array(area,np.int32) , (cx,cy) , False) >= 0:
                    cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(im0, (cx,cy), 3, (0,0,255) , -1)
                    cv2.putText(im0 , class_name , (x1,y1) , cv2.FONT_HERSHEY_COMPLEX , 0.5, (255,255,255),1)
                    occupancy[idx] += 1
                    break

    total_occupancy = sum(occupancy)
    available_space = len(areas) - total_occupancy
    print(available_space)


    for idx , area in enumerate(areas):
        color = (0,0,255) if occupancy[idx] else (0,255,0)
        cv2.polylines(im0 , [np.array(area , np.int32)] , True ,color ,2)
        cv2.putText(im0 , str(idx+1) , tuple(area[0]), cv2.FONT_HERSHEY_COMPLEX , 0.5,color , 1)

    cv2.putText(im0,str(available_space),(23,30),cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),2)

    # for result in results.boxes.data.tolist():
    #     x1,y1,x2,y2,score,class_id = result
    #     if score > 0.5:
    #         cv2.rectangle(
    #             im0,
    #             (int(x1) , int(y1)),
    #             (int(x2) , int(y2)),
    #             (0,255,0),
    #             4
    #         )
    cv2.putText(
                im0,
                str(available_space),
                (1000,450),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                (255,255,255),
                3,
                cv2.LINE_AA
    )

    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()