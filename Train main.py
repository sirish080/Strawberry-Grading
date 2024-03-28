from pathlib import Path
import imgaug
from engine import *
from mrcnn.config import Config
from mrcnn import model as modellib, utils

#%%
ROOT_DIR = Path(r'E:/01. Automatic strawberry grading/Dataset') 
COCO_WEIGHTS_PATH = "Coco weights/mask_rcnn_coco.h5"
DEFAULT_LOGS_DIR = "../Strawberry-Mask-RCNN/logs"
            #Dataset Directory

#%%       

classes = ['Ideal', 'Mild', 'Heavy']                                        #Object Classes
epochs = 250                                   

#%%

class CustomConfig(Config):
    """Configuration for training on the custom  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "object"
    # NUMBER OF GPUs to use. When using only a CPU, this needs to be set to 1.
    GPU_COUNT = 1
    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2                                                         #CHANGE
    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # Background + unripe, semiripe and ripe               #CHANGE
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 1000                                                     #CHANGE
    #Wheather to resize image for training and predicting or not
    #IMAGE_RESIZE_MODE = "none"                                                 #CHANGE
    #IMAGE_MIN_DIM = 800
    #IMAGE_MAX_DIM = 1280
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.95
    LEARNING_RATE = 0.001

#%% Augmentations

augmentation = imgaug.augmenters.Sequential([ 
    imgaug.augmenters.Fliplr(1),
    #imgaug.augmenters.Affine(rotate=(-15, 15)),
    imgaug.augmenters.Affine(scale=(0.5, 1.5)),
    imgaug.augmenters.Crop(px=(0, 10)),
    imgaug.augmenters.AddToHueAndSaturation((-20, 20)), # change hue and saturation
    imgaug.augmenters.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
    imgaug.augmenters.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))])

#%%

if __name__ == "__main__":                              				
    config = CustomConfig()
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=DEFAULT_LOGS_DIR)
    weights_path = COCO_WEIGHTS_PATH
    if not os.path.exists(weights_path):
      utils.download_trained_weights(weights_path)

    model.load_weights(weights_path, by_name=True, exclude=[
                "mrcnn_class_logits", "mrcnn_bbox_fc",
                "mrcnn_bbox", "mrcnn_mask"])
    
    train(model, ROOT_DIR, classes, epoch = epochs, lr = Config.LEARNING_RATE)
 

