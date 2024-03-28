import os
import cv2
import sys
from Inference_functions import *
import mrcnn.model as modellib
from mrcnn.config import Config
from pathlib import Path
from mrcnn import visualize


ROOT_DIR = Path("..\Mask RCNN\dataset") 
DEFAULT_LOGS_DIR = "../Mask RCNN/logs"
model_path = '../Mask RCNN/logs/object20230822T1713/mask_rcnn_object_0250.h5'
classes = ['BG','Ideal', 'Mild', 'Heavy'] 

#%%

class InferenceConfig(Config):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.95
    DETECTION_NMS_THRESHOLD = 0.3
    NAME = "object"
    NUM_CLASSES = 1 + 3
    
#_______Defigning Configuration and inputing model form weight path______________
inference_config = InferenceConfig()
model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=DEFAULT_LOGS_DIR)
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

#%%
Video_path = r"E:\Backup\Strawberry Detection Model\02. Data Collection\Video Data\iPhone Camera Video\IMG_0061.MOV"
#________LOAD AND PLOT THE ORIGINAL IMAGE_____________

cap = cv2.VideoCapture(Video_path)
index = 0
while cap.isOpened():
    Ret, Mat = cap.read()
    
    if Ret:
        index+= 1
        if index % 20 != 0:
            continue
        img_rgb = Mat.copy()
        img_nrgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        results = model.detect([img_nrgb], verbose=True)
        r = results[0]
        boxes = r['rois']
        masks = r['masks']
        class_ids = r['class_ids']
        scores = r['scores']
        #abcd = visualize.display_instances(img_nrgb, boxes, masks, class_ids, classes)
        polygons=mask2polygon(img_nrgb,masks)
        points= convertpolygonintopoints(polygons)
        maskcenter, minx, maxx = centerofmask(points) #return the centroid of mask and coordinate with min and max x values
        #minmaxup = pointsbetweenminxandmaxx(points,minx,maxx) #return all the coordinate between min max x values
        #minmaxdown = pointsbetweenminxandmaxxdown(points,minx,maxx)
        miny, maxy = minmaxy (points)
        nimage = newdrawimage(img_nrgb, miny, points)
        nimage = cv2.cvtColor(nimage, cv2.COLOR_BGR2RGB)
        
        cv2.imshow("this is the image", nimage)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    else:
        cv2.destroyAllWindows()

cap.release()




