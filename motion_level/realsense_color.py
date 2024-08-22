## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
from PIL import Image 
import pdb
# Configure depth and color streams
import json 
def start_pipeline(width=640, height=480, serial_number=None):
    pipeline = rs.pipeline()
    config = rs.config()
    
    # 238222073605 for left side camera
    # 851112062064 for top camera
    # ctx = rs.context()
    # if len(ctx.devices) > 0:
    #     for d in ctx.devices:
    #         print ('Found device: ', \
    #                 d.get_info(rs.camera_info.name), ' ', \
    #                 d.get_info(rs.camera_info.serial_number))
    #     else:
    #         print("No Intel Device connected")
    # pdb.set_trace()
    # Get device product line for setting a supporting resolution
    
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)
    if (serial_number is not None):
        config.enable_device(serial_number)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
    
    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        print(device_product_line, width, height)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)
    return pipeline 

def end_pipeline(pipeline):
    pipeline.stop()


def get_rgbd(pipeline):
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        # Convert images to numpy arrays
        # pdb.set_trace()
        color_image = np.asanyarray(color_frame.get_data())
        color_image = (color_image.astype(np.uint8) - 128.0) / 128
        color_image = color_image[:,:,::-1] # BGR to RGB
        depth_image = np.asanyarray(depth_frame.get_data()) / 1000 # convert to meters
        depth_image = depth_image[...,np.newaxis]
        image = np.concatenate((color_image, depth_image), axis=2)
        image = cv2.resize(image, (256, 256))
        return image
    
 
def get_latest_frame(pipeline, num_warm_up=30):
    frame_buffers = []
    for i in range(num_warm_up):
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue 
        color_image = np.asanyarray(color_frame.get_data())
        frame_buffers.append(color_image)
    return frame_buffers[-1]
    
def get_latest_frame_cli(pipeline, num_warm_up=30):
    frame_num = 0
    while True:
        command = input("Enter command: \n")
        if command == "c":
            frame_buffers = []
            for i in range(30):
                # Wait for a coherent pair of frames: depth and color
                frames = pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                
                color_image = np.asanyarray(color_frame.get_data())

                image = Image.fromarray(color_image)

                crop_rectangle = (80, 36, 501, 396)

                # Crop the image
                cropped_image = image.crop(crop_rectangle)
                cropped_image = np.asanyarray(cropped_image)
                
                frame_buffers.append(cropped_image)
            #print(f"frame_buffers[-1] {frame_buffers[-1]}")
            #print(f"save in /home/hongyic/Documents/table_top_push/kortex/vlm_actions/wood_top_down/test_move_image_{frame_num}.png")
            cv2.imwrite(f"/home/hongyic/Documents/table_top_push/kortex/vlm_actions/wood_top_down/test_move_image_{frame_num}.png", frame_buffers[-1])
            frame_num += 1

        elif command == "exit":
            break 
    return

if __name__=="__main__":
    import argparse 
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-fn","--frame_num",type=int,default=0)
    # args = parser.parse_args()
    pipeline=start_pipeline()
    get_latest_frame_cli(pipeline)
    end_pipeline(pipeline)