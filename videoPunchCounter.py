#this script counts punches in a video using the trained punch detection model in train.ipynb 

import cv2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog,MetadataCatalog
import os

#import video
vc = cv2.VideoCapture("mcgregorvsdiaz.mp4")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
vw = cv2.VideoWriter("output1.avi",fourcc,20,(1280,720))

#get the model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.NUM_CLASSES=3
cfg.OUTPUT_DIR = 'faster_rcnn_R_50_FPN_3x\\lr=0.001,itr=3000'
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
predictor = DefaultPredictor(cfg)

#set parameters
counter = 0
punches=0
buffer=[]

#start video classification
while True:
    ret,img = vc.read()
    if not ret:
        break


    counter+=1
    # print(counter)

    outputs = predictor(img)

    #draw punch bounding box
    boxes = outputs["instances"].pred_boxes.tensor.tolist()
    labels=outputs["instances"].pred_classes.tolist()
    conf_lvls=outputs["instances"].scores.tolist()

    #value format=(boxes,conf_lvl)
    punchBoxes=[]
    for i,box in enumerate(boxes):
        if labels[i]==2:
            punchBoxes.append([box,conf_lvls[i]])

    # add to punch count if punch is detected in 3 to 10 consecutive frames
    if len(punchBoxes)>=1:         
        # if buffer is empty or has previous frame
        if len(buffer)==0 or buffer[len(buffer)-1]+1==counter:
            buffer.append(counter)
        # clear buffer if frame is not consecutive
        elif buffer[len(buffer)-1]+1!=counter:
            if len(buffer)>=3 and len(buffer)<=10:
                punches+=1
            buffer=[]
            buffer.append(counter)

        if len(buffer)==10:
            if len(buffer)>=3 and len(buffer)<=10:
                punches+=1
            buffer=[]
            buffer.append(counter)

    # print statistics for debugging purposes
    print(counter,buffer)
    
    # draw these boxes on the image
    for box in punchBoxes:
        cv2.rectangle(img,(int(box[0][0]),int(box[0][1])),(int(box[0][2]),int(box[0][3])),(255,0,0),2)

    cv2.putText(img,"Number of punches: {}".format(punches),(50,50),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),2)

    # uncomment this if you want to see which frames were detected as a punch
    '''
    if len(punchBoxes)>0:
        cv2.imwrite("framesDetected\\{}.jpg".format(counter),img)
    '''

    vw.write(img)

vc.release()
vw.release()
