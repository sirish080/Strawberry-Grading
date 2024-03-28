import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
from mrcnn.visualize import display_instances
import mrcnn.model as modellib
from mrcnn.model import log
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from pathlib import Path
from engine import *

#%% 

ROOT_DIR = Path(r'E:/01. Automatic strawberry grading/Dataset') 
DEFAULT_LOGS_DIR = 'logs'
model_path = 'logs\object20240213T1756\mask_rcnn_object_0250.h5'

classes = ['Ideal', 'Mild', 'Heavy']      #['unripe', 'semiripe', 'ripe']  

#%%

class InferenceConfig(Config):
    
    NAME = "object"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    #Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.9
    
    NUM_CLASSES = 1 + 3  # Background + Hard_hat, Safety_vest

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3

#%%
config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config = config,
                          model_dir = DEFAULT_LOGS_DIR)

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

#%%

dataset_val = CustomDataset()
dataset_val.load_custom(ROOT_DIR, "Val", classes)
dataset_val.prepare()

#%%
'''
IOU_threshold = 0.5
gt_tot = np.array([])
pred_tot = np.array([])
#mAP list
mAP_ = []
mIoU = []

for image_id in dataset_val.image_ids:
    image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_val, config, image_id)#, #use_mini_mask=False)
    info = dataset_val.image_info[image_id]

    # Run the model
    results = model.detect([image], verbose=1)
    r = results[0]
    
    class_names = ["BG",'Ideal', 'Mild', 'Heavy']
    #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                #class_names, r['scores'])
    
    #compute gt_tot and pred_tot
    gt, pred = utils.gt_pred_lists(gt_class_id, gt_bbox, r['class_ids'], r['rois'], IOU_threshold)
    gt_tot = np.append(gt_tot, gt)
    pred_tot = np.append(pred_tot, pred)
    
    
    #precision_, recall_, AP_ 
    AP_, precision_, recall_, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                          r['rois'], r['class_ids'], r['scores'], r['masks'], iou_threshold = IOU_threshold)
                 
    
    
    #check if the vectors len are equal
    print("the actual len of the gt vect is : ", len(gt_tot))
    print("the actual len of the pred vect is : ", len(pred_tot))
    
    mAP_.append(AP_)
    print("Average precision of this image : ",AP_)
    print("The actual mean average precision for the whole images", sum(mAP_)/len(mAP_))
 
    #print("Ground truth object : "+dataset.class_names[gt])

    #print("Predicted object : "+dataset.class_names[pred])
    # for j in range(len(dataset.class_names[gt])):
	    # print("Ground truth object : "+j)
        
gt_tot=gt_tot.astype(int)
pred_tot=pred_tot.astype(int)
#save the vectors of gt and pred
save_dir = 'results'
gt_pred_tot_json = {"gt_tot" : gt_tot, "pred_tot" : pred_tot}
df = pd.DataFrame(gt_pred_tot_json)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
df.to_csv(os.path.join(save_dir,"gt_pred_test.csv"))


tp,fp,fn = utils.plot_confusion_matrix_from_data(gt_tot,pred_tot,columns=['BG','Ideal', 'Mild', 'Heavy'] ,fz=18, figsize=(15,15), lw=0.5)



print("tp for each class :",tp)
print("fp for each class :",fp)
print("fn for each class :",fn)

#eliminate the background class from tps fns and fns lists since it doesn't concern us anymore : 
del tp[0]
del fp[0]
del fn[0]
print("\n########################\n")
print("tp for each class :",tp)
print("fp for each class :",fp)
print("fn for each class :",fn)



# Draw precision-recall curve

AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                          r['rois'], r['class_ids'], r['scores'], r['masks'], iou_threshold = IOU_threshold)
visualize.plot_precision_recall(AP, precisions, recalls, IOU_threshold = IOU_threshold)


# # calculate the mAP for a batch of validation images

#%%
'''
'''
Batch testing


path = os.getcwd()
#model_tar = "nuclei_datasets.tar.gz"
data_path = os.path.join(path + '/dataset')
model_path = os.path.join(path + '/logs')
weights_path = os.path.join(model_path + '/mask_rcnn_strawberry_cfg_0250.h5') #My weights file

DEVICE = "/gpu:0" 


config=inference_config
dataset = dataset_val


with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=model_path, config=config)

model.load_weights(weights_path, by_name=True)

    
def compute_batch_ap(image_ids):
    APs = []
    for image_id in image_ids:
        # Load image
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, config, image_id)
        # Run object detection
        results = model.detect([image], verbose=0)
        # Compute AP
        r = results[0]
        AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask, r['rois'], r['class_ids'], r['scores'], r['masks'])
        AP = 1 - AP
        APs.append(AP)
    return APs, precisions, recalls

#dataset.load_nucleus(data_path, 'val')
#dataset.prepare()
print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))
print("Loading weights ", weights_path)

image_ids = np.random.choice(dataset.image_ids, 25)
APs, precisions, recalls = compute_batch_ap(image_ids)
print("mAP @ IoU=50: ", APs)

AP = np.mean(APs)
visualize.plot_precision_recall(AP, precisions, recalls)
plt.show()

'''
#%%
    

###Visualize

import Grading_engine_MRCNN as GMRCNN
import Final_grading as fg
import statistics
from pathlib import Path 
from timeit import default_timer as timer
IMAGE_DIR = Path(r'E:/01. Automatic strawberry grading/Dataset/Test')

# # COCO Class names
# # Index of the class in the list is its ID. For example, to get ID of
# # the teddy bear class, use: class_names.index('teddy bear')
class_names = ["BG",'Visible', 'Mild', 'Heavy']

file_names = next(os.walk(IMAGE_DIR))[2]
#x=random.choice(file_names)
#print("file name:", x)
#image = skimage.io.imread(os.path.join(IMAGE_DIR, x))

# # Run detection

for i in range(len(file_names)):
    path = IMAGE_DIR/file_names[i]
    #print(path)
    image = skimage.io.imread(path)
    #plt.imshow(image)

    results = model.detect([image], verbose=1)

# # Visualize results

    r = results[0]
    overall_info = {'id':[], 'Grade':[], 'major_axis':[], 'Ripeness':[], 'Ripeness_boundary':[], 'shapes':[], 'size_class':[]}
    for i in range(len(r['rois'])):
        buffer = 10 #pixels
        x1, y1, x2, y2 = r['rois'].tolist()[i]
        mask_binary = r['masks'].astype('uint8')[x1:x2, y1:y2, i] 
        RGB_image = image[x1:x2, y1:y2,:]
        
        Major_axis, Shape, Ripeness = GMRCNN.get_attributes(RGB_image, mask_binary )

        
        ##Update to original major axis coordinates
        Head = Major_axis[1][0]
        Apex = Major_axis[1][1]
        
        Head = [Head[1]+y1, Head[0]+x1]
        Apex = [Apex[1]+y1, Apex[0]+x1]
        
        ##Update to original ripeness boundary coordinates
        if Ripeness[1] is None:
            Left, Right = None, None
            
        else:
            Left = Ripeness[1][0]
            Right = Ripeness[1][1]
            
            Left = [Left[1]+y1, Left[0]+x1]
            Right = [Right[1]+y1, Right[0]+x1]
    
            
            
        
        '''
        plt.imshow(image, cmap = 'gray')
        plt.plot([Head[0], Apex[0]], [Head[1], Apex[1]], c = 'green',linewidth = 0.5 )
        plt.plot([Left[0], Right[0]], [Left[1], Right[1]], c = 'green',linewidth = 0.5 )
        plt.show()
        '''

        Grading_attributes = [Shape, Ripeness[0][0], Major_axis[0]]
        Grade, Final_score, scores = fg.grading(Grading_attributes)
        
        overall_info['Grade'].append(Grade)
        overall_info['id'].append(i)
        overall_info['major_axis'].append([Head, Apex])
        overall_info['Ripeness'].append(Ripeness[0])
        overall_info['Ripeness_boundary'].append([Left, Right])
        overall_info['shapes'].append(Shape)
        overall_info['size_class'].append(scores[1])


 
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                class_names, overall_info, dpi = 800, show_mask= False, show_caption= True)

