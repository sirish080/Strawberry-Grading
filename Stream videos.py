import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt
import keyboard 
#import pandas as pd
from skimage import io
import re
import glob
from datetime import datetime

exposure =200
USB = 2.0


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
       
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)
    
if device_product_line == 'D400':
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
else:
    
    if USB ==3.0:
        
    #This is while using USB 3.0
        config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
    
    else:
        
    #This is while using USB 2.0
        config.enable_stream(rs.stream.depth, 320, 240, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    

# Start streaming
profile = pipeline.start(config)
#s = profile.get_device().query_sensors()[1]
#s.set_option(rs.option.exposure, exposure)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

align_to = rs.stream.color
align = rs.align(align_to)

count = 0
Image_dir = r'../Strawberry images/123//'
Files = []
'''
for i in glob.glob(Image_dir +  '/*.jpg'):
    number = int((re.findall(r'\d+', i))[0])
    if number > count:
        count = number
'''
try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        
        # Align the depth frame to color frame
        aligned_frames = align.process(frames)
        
        # Get aligned frames
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        height = rs.depth_frame.get_height(depth_frame)
        width= rs.depth_frame.get_width(depth_frame)
        
        
        # Apply filter to fill the Holes in the depth image
        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.holes_fill, 3)
        filtered_depth = spatial.process(depth_frame)
        hole_filling = rs.hole_filling_filter()
        depth_frame_filled = rs.frame.as_depth_frame(hole_filling.process(filtered_depth))


        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_image = np.expand_dims(depth_image, axis = -1)
        
        depth_image_filled = np.asanyarray(depth_frame_filled.get_data())
        depth_image_filled = np.expand_dims(depth_image_filled, axis = -1)

        color_image_BGR = np.asanyarray(color_frame.get_data())
        color_image_RGB = cv2.cvtColor(color_image_BGR, cv2.COLOR_BGR2RGB)


        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        depth_filled_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_filled, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image_RGB.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image_RGB, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.concatenate((resized_color_image, depth_image), axis = 2)
        else:
            images = np.concatenate((color_image_RGB, depth_image_filled), axis = 2)

        #images1 = np.hstack((color_image, depth_colormap))
        #images2 = np.hstack((depth_colormap, depth_filled_colormap))
        
        img = color_image_BGR.copy()
        scale_percent =50               #Scaling image
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            
        # Show images
        cv2.namedWindow('Color', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Color', img)
        cv2.waitKey(1)
        
        cv2.namedWindow('Depth', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Depth', depth_filled_colormap)
        cv2.waitKey(1)
        
        '''
        
        if keyboard.is_pressed("end"):
            print('Stopping the streaming')
            cv2.destroyAllWindows()
            break
        elif keyboard.is_pressed(' '):
            
            now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            
            
            print('Printing sample images!')
            plt.imshow(color_image_RGB)
            plt.show()
            
            depth_array = depth_image_filled * depth_scale
            
            depth_list = []
            for i in range(height):
                for j in range(width):
                    #print(i,j)
                    depth = rs.depth_frame.get_distance(depth_frame_filled,j,i)
                    #print(depth)
                    depth_list.append(depth)
            depth_array = np.array(depth_list).reshape(height, width)
            depth_array = np.expand_dims(depth_array, axis = -1)
            
            
            plt.imshow(depth_array, cmap = 'gray')
            plt.show()
            
    
            count += 1
            print('Sample image number starting from {}'.format(count))
            file_name = 'Image'+ now
            print('Saving image: {}'.format(file_name))
            
            
            io.imsave((Image_dir+'RGBimg'+ now + '.bmp'), color_image_RGB)
            io.imsave((Image_dir+ 'Depthimg' + now + '.tiff'), depth_array)
            

        else:

            continue
        '''
            
finally:
    # Stop streaming
    pipeline.stop()
    