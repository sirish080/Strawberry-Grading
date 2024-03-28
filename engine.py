import os
import json
import numpy as np
import skimage.draw
#from mrcnn.config import Config
from mrcnn import utils

#%% DATASET

class CustomDataset(utils.Dataset):

    def load_custom(self, dataset_dir, subset, classes):
        """Load a subset of the Dog-Cat dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
      
        # Add classes. We have only three class to add.
        #self.add_class("object", 1, "unripe")   #CHANGE
        #self.add_class("object", 2, "semiripe")
        #self.add_class("object", 3, "ripe")
        
        for i in range(len(classes)):
            self.add_class("object", i, "classes[i]")
            

     
        # Train or validation dataset?
        assert subset in ["Train", "Val"]
        #dataset_dir = os.path.join(dataset_dir, subset)
        dataset_dir = dataset_dir/subset
        
        #FORMAT OF THE ANNOTATION JSON FILE
        # Load annotations
        # VGG Image Annotator saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        #annotations1 = json.load(open("D:/mrcnnpractice/Aarohicode/dataset/train/train.json")) Bhola changed this code to something like below
        
        if subset == "Train":
            annotations1 = json.load(open(os.path.join(dataset_dir, 'train.json')))
        elif subset == "Val":
            annotations1 = json.load(open(os.path.join(dataset_dir, 'Val.json')))
        
        # print(annotations1)
        annotations =list(annotations1.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]
        
        # Add images
        for a in annotations:
            # print(a)
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            polygons = [r['shape_attributes'] for r in a['regions']] 
            objects = [s['region_attributes']['Strawberry'] for s in a['regions']]
            print("objects:",objects)
            name_dict = {"Ideal": 1, "Mild": 2, "Heavy" : 3}  #{"unripe": 1,"semiripe": 2,"ripe": 3}                   #CHANGE CLASS ID AND NAME

            # key = tuple(name_dict)
            num_ids = [name_dict[a] for a in objects]
     
            # num_ids = [int(n['Event']) for n in objects]
            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            print("numids",num_ids)
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "object",  ## for a single class just add the name here
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                num_ids=num_ids
                )

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a Dog-Cat dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        if info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)
        num_ids = info['num_ids']
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
        	rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])

        	mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        # Map class names to class IDs.
        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids #np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
            
#%% MODEL TRAINING FUNCTIONS  
         
def train(model, data_dir, classes, epoch, lr):
    """Train the model."""
    # Training dataset.
    dataset_train = CustomDataset()
    dataset_train.load_custom(data_dir, "Train", classes)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CustomDataset()
    dataset_val.load_custom(data_dir, "Val", classes)
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    
    # print("Training network heads")
    #model.train(dataset_train, dataset_val,
                #learning_rate=config.LEARNING_RATE,
                #epochs=300,
                #layers='heads')
                
    model.train(dataset_train, dataset_val,
                learning_rate= lr,
                epochs=epoch,                               #CHANGE HERE
                layers='heads')
#				
#	
#   
# Another way of using imgaug    
# augmentation = imgaug.Sometimes(5/6,aug.OneOf(
                                            # [
                                            # imgaug.augmenters.Fliplr(1), 
                                            # imgaug.augmenters.Flipud(1), 
                                            # imgaug.augmenters.Affine(rotate=(-45, 45)), 
                                            # imgaug.augmenters.Affine(rotate=(-90, 90)), 
                                            # imgaug.augmenters.Affine(scale=(0.5, 1.5))
                                             # ]
                                        # ) 
                                   # )
                                   
#%%
