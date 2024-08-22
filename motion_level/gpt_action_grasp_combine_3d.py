# caller.py
from multiprocessing import Process, Pipe
from table_top_push import kinova_control
from realsense_color import start_pipeline, end_pipeline, get_latest_frame
from gpt_utils import make_gpt_call, encode_numpy_image, RotatedRect
import numpy as np 
import cv2
import json 
import pathlib
import random
import math
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
import sys
sys.path.insert(0, './Detic/third_party/CenterNet2/')
sys.path.insert(0, './Detic/')
from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.predictor import VisualizationDemo
 
SYSTEM_MESSAGE_PLACEHOLDER = "SYSTEM_MESSAGE_PLACEHOLDER"
LOCATION_MESSAGE_PLACEHOLDER = "LOCATION_MESSAGE_PLACEHOLDER"
SUMMARY_MESSAGE_PLACEHOLDER = "SUMMARY_MESSAGE_PLACEHOLDER"
# DEFAULT_ROT_DEGREE = 20

openai_key_file = pathlib.Path(__file__).parent / "openai_key.txt"
with open(openai_key_file, "r") as f:
    openai_key = f.read().strip()
    
# 298, 214; 
# size: 640, 480
corner_offset = np.array((640//2 - 294, 480//2 - 214))
step_size = 0.03
forward_step_size = 0.03
top_hor_step_size = 0.03
rotate_degree = 5
ACTION_NAME_TO_MOVEMENT={
    "move up": [0,0,step_size],
    "move down": [0,0,-step_size],
    "move forward": [forward_step_size,0,0],
    "move backward": [-forward_step_size,0,0],
    "move left": [0,top_hor_step_size,0],
    "move right": [0,-top_hor_step_size,0],
    "stop": [0, 0, 0]
}

TARGET_XY = [47.2*0.01,2.6*0.01]

max_x = 60 * 0.01 #74
max_y = 30 * 0.01 #40
min_y = -30 * 0.01 #-36
min_x = 20 * 0.01 #24

goal_image = cv2.imread("correct.png")

def sample_points_in_rectangle_regular(top_right, bottom_left, rows, cols):
    # Calculate rectangle dimensions
    width = top_right[0] - bottom_left[0]
    height = top_right[1] - bottom_left[1]
    
    # Calculate spacing between points
    x_spacing = width / cols
    y_spacing = height / rows
    
    # Generate and return points
    points = []
    for row in range(rows):
        for col in range(cols):
            # Calculate point coordinates
            x = bottom_left[0] + (col + 0.5) * x_spacing
            y = bottom_left[1] + (row + 0.5) * y_spacing

            # make relative to target
            relative_x = x - TARGET_XY[0]
            relative_y = y - TARGET_XY[1]
            points.append([relative_x, relative_y])
    return points

def sample_around_target_xy_regular(x_min, x_max, y_min, y_max, target_center, target_center_deadzone, num_rows, num_cols):
    quad1_top_right, quad1_bottom_left = [x_max, y_max], [target_center[0]+target_center_deadzone, target_center[1]+target_center_deadzone]
    quad2_top_right, quad2_bottom_left = [target_center[0]-target_center_deadzone, y_max], [x_min, target_center[1]+target_center_deadzone]
    quad3_top_right, quad3_bottom_left = [target_center[0]-target_center_deadzone, target_center[1]-target_center_deadzone], [x_min, y_min]
    quad4_top_right, quad4_bottom_left = [x_max, target_center[1]-target_center_deadzone], [target_center[0]+target_center_deadzone, y_min]
    
    quad1_samples = sample_points_in_rectangle_regular(quad1_top_right, quad1_bottom_left, num_rows, num_cols)
    quad2_samples = sample_points_in_rectangle_regular(quad2_top_right, quad2_bottom_left, num_rows, num_cols)
    quad3_samples = sample_points_in_rectangle_regular(quad3_top_right, quad3_bottom_left, num_rows, num_cols)
    quad4_samples = sample_points_in_rectangle_regular(quad4_top_right, quad4_bottom_left, num_rows, num_cols)
    return quad4_samples


def simple_parse_action(summarized_action, scale_factor=1.0):
    print(summarized_action)
    final_action = [0, 0]
    encountered_similar = False 
    if "upward" in summarized_action.lower() or "above" in summarized_action.lower():
        final_action = ACTION_NAME_TO_MOVEMENT["move up"]
    elif "downward" in summarized_action.lower() or "below" in summarized_action.lower():
        final_action = ACTION_NAME_TO_MOVEMENT["move down"]
    elif "forward" in summarized_action.lower():
        final_action = ACTION_NAME_TO_MOVEMENT["move forward"]
    elif "backward" in summarized_action.lower():
        final_action = ACTION_NAME_TO_MOVEMENT["move backward"]
    elif "left" in summarized_action.lower():
        final_action = ACTION_NAME_TO_MOVEMENT["move left"]
    elif "right" in summarized_action.lower():
        final_action = ACTION_NAME_TO_MOVEMENT["move right"]
    elif "stop" in summarized_action.lower() or "similar" in summarized_action.lower():
        final_action = ACTION_NAME_TO_MOVEMENT["stop"]
        encountered_similar = True 
    else:
        final_action = ACTION_NAME_TO_MOVEMENT["stop"]  
    if encountered_similar:
        print("Encountered similar")
    return final_action, encountered_similar
    
def simple_stop_query(gpt_history, summarized_prompt):
    local_history = gpt_history    
    response_summary_query = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": summarized_prompt
            }
        ]
    }
    local_history.append(response_summary_query)
    summarized_action = make_gpt_call(local_history, openai_key)
    summarized_action_msg = summarized_action.json()['choices'][0]['message']['content']
    local_history.append(summarized_action.json()['choices'][0]['message'])
    
    if "(yes)" in summarized_action_msg.lower():
        return True
    else: 
        return False

def single_image_ask_gpt_movement(image, system_message, query_message, gpt_history):
    encoded_image = encode_numpy_image(image)
    encoded_goal_image = encode_numpy_image(goal_image)
    sys_message = {
        "role": "system",
        "content": [
                {
                "type": "text",
                "text": system_message
                },
            ]
    }
    gpt_history.append(sys_message)
    
    query_message = {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
            },
            {
                "type": "text",
                "text": query_message
            },
        ]
    }
    gpt_history.append(query_message)
    response = make_gpt_call(gpt_history, openai_key)
    response_content = response.json()['choices'][0]['message']['content']
    gpt_history.append(response.json()['choices'][0]['message'])
    return response_content, response


def parse_to_action(response, summarized_prompt, gpt_history, no_history=False, scale_factor=None, scale_factor_idx=0, is_stop_query=False):
    local_history = gpt_history    
    if no_history:
        local_history = []
    
    response_summary_query = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": summarized_prompt
            }
        ]
    }
    local_history.append(response_summary_query)
    summarized_action = make_gpt_call(local_history, openai_key)
    summarized_action_msg = summarized_action.json()['choices'][0]['message']['content']
    local_history.append(summarized_action.json()['choices'][0]['message'])

    action_value, enountered_similar = simple_parse_action(summarized_action_msg, scale_factor=scale_factor[scale_factor_idx])
    # if enountered_similar and scale_factor is not None:
    #     if scale_factor[scale_factor_idx] == 1.0:
    #         scale_factor[scale_factor_idx] = 0.2
    return action_value, summarized_action_msg

def detect_grasp_object(current_image, top=False):
    if top:
        lower_pink = np.array([50, 50, 150])
        upper_pink = np.array([100, 100, 200])
    else:
        lower_pink = np.array([0, 0, 90])
        upper_pink = np.array([40, 40, 140])
    frame_threshed = cv2.inRange(current_image, lower_pink, upper_pink)

    # apply morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    clean = cv2.morphologyEx(frame_threshed, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    clean = cv2.morphologyEx(frame_threshed, cv2.MORPH_CLOSE, kernel)

    # get external contours
    contours = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    c = max(contours, key = cv2.contourArea)

    # get rotated rectangle from contour
    rot_rect = cv2.minAreaRect(c)
    rotation = rot_rect[2]
    #pdb.set_trace()
    box = cv2.boxPoints(rot_rect)
    box = np.intp(box)
    # draw rotated rectangle on copy of img
    # cv2.drawContours(current_image,[box],0,(0,0,0),2)
    current_box = RotatedRect(int((box[0][0]+box[1][0]+box[2][0]+box[3][0])/4), int((box[0][1]+box[1][1]+box[2][1]+box[3][1])/4), int(math.dist(box[0],box[1])), int(math.dist(box[0],box[1])), rotation)
    box_area = int(math.dist(box[0],box[1])) ** 2
    return current_image, current_box, box_area, box


def setup_cfg(args):
    cfg = get_cfg()
    if args.cpu:
        cfg.MODEL.DEVICE="cpu"
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand' # load later
    if not args.pred_all_class:
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="./Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", help="Take inputs from webcam.")
    parser.add_argument("--cpu", action='store_true', help="Use CPU only.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--vocabulary",
        default="custom",
        choices=['lvis', 'openimages', 'objects365', 'coco', 'custom'],
        help="",
    )
    parser.add_argument(
        "--custom_vocabulary",
        default="bottle", # teddy bear, hippo , bottlecap
        help="",
    )
    parser.add_argument("--pred_all_class", action='store_true')
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=['MODEL.WEIGHTS', './Detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'],
        nargs=argparse.REMAINDER,
    )
    return parser


def run_callee(prompt_configuration,loggin_dict, log_save_dir, connection_kwargs, max_steps=10, num_seeds=1, fake_marker_type=None, rows=1, cols=1, rotation_type=None, width=640, height=480):
    
    h_sys_msg = prompt_configuration["hor_sys_msg"]
    h_query_msg = prompt_configuration["horizontal_prompt"]
    h_sum_msg = prompt_configuration["horizontal_summary_prompt"]
    
    v_sys_msg = prompt_configuration["ver_sys_msg"]
    v_query_msg = prompt_configuration["vertical_prompt"]
    v_sum_msg = prompt_configuration["vertical_summary_prompt"]
    
    th_sys_msg = prompt_configuration.get("top_hor_sys_msg", None)
    th_query_msg = prompt_configuration.get("top_horizontal_prompt", None)
    th_sum_msg = prompt_configuration.get("top_horizontal_summary_prompt", None)
    
    s_sys_msg = prompt_configuration.get("stop_sys_msg", None)
    stop_query_msg = prompt_configuration.get("stop_query_msg", None)
    stop_sum_msg = prompt_configuration.get("stop_sum_msg", None)
    

    parent_conn, child_conn = Pipe()  # Create a pipe for IPC
    is_rotate = rotation_type is not None
    p = Process(target=kinova_control, args=(child_conn,connection_kwargs, is_rotate, ))  # Pass the child connection to the subprocess
    # get pid of the process
    p.start()
    pipeline_left = start_pipeline(width, height, "238222075236")
    pipeline_top = start_pipeline(width, height, "851112062064")
    # Detic object detection model
    args = get_parser().parse_args()
    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg, args)
    
    def save_logs(seed, step, log_save_dir, hor_raw_msg, summarized_hor_action, ver_raw_msg, summarized_ver_action, top_hor_raw_msg, summarized_top_hor_action, combined_action, current_image_left, current_image_top):
        # logging 
        img_save_dir = f"{log_save_dir}/images_s{seed}/step_{step}_left.png"
        cv2.imwrite(img_save_dir, current_image_left)
        img_save_dir = f"{log_save_dir}/images_s{seed}/step_{step}_top.png"
        cv2.imwrite(img_save_dir, current_image_top)
    
        if seed not in loggin_dict["communications"]:
            loggin_dict["communications"][seed]={}
        loggin_dict["communications"][seed][step] = {
            "horizontal_response": hor_raw_msg,
            "horizontal_action": summarized_hor_action,
            "vertical_response": ver_raw_msg,
            "vertical_action": summarized_ver_action,
            "top_hor_raw_msg": top_hor_raw_msg,
            "summarized_top_hor_action": summarized_top_hor_action,
            "combined_action": combined_action,
            "image_dir": img_save_dir,
        }
        with open(f"{log_save_dir}/comm_log.json", "w") as f:
            json.dump(loggin_dict, f, indent=4)
    
    decay_ratio = 0.97
    for seed in range(4):
        step_decay = 1.0
        scale_factors = [1.0, 1.0]
        os.makedirs(f"{log_save_dir}/images_s{seed}", exist_ok=True)
        for step in range(max_steps):
            print(f"step {step}")
            
            # left/right movement
            current_image_top = get_latest_frame(pipeline_top)
            _, after_box, box_area, box = detect_grasp_object(current_image_top, top=True)
            predictions, visualized_output = demo.run_on_image(current_image_top)
            current_image_top = visualized_output.get_image() #[:, :, ::-1]
            cv2.rectangle(current_image_top, (after_box.cx-10, after_box.cy-10), (after_box.cx+10, after_box.cy+10), (255, 0, 0), -1) # fill 
            current_image_top = current_image_top[:,:,::-1]
            gpt_history = []
            #pdb.set_trace()
            top_hor_raw_msg, _ = single_image_ask_gpt_movement(current_image_top, th_sys_msg, th_query_msg, gpt_history)
            top_hor_action, summarized_top_hor_action = parse_to_action(top_hor_raw_msg, th_sum_msg, gpt_history, scale_factor=scale_factors, scale_factor_idx=0)
            
            gpt_history = []
            hor_raw_msg, hor_response = single_image_ask_gpt_movement(current_image_top, h_sys_msg, h_query_msg, gpt_history)
            hor_action, summarized_hor_action = parse_to_action(hor_raw_msg, h_sum_msg, gpt_history, scale_factor=scale_factors, scale_factor_idx=0)
            
            
            # forward/backward and upward/downward movement
            current_image_left = get_latest_frame(pipeline_left)
            _, after_box, box_area, box = detect_grasp_object(current_image_left)
            predictions, visualized_output = demo.run_on_image(current_image_left)
            current_image_left = visualized_output.get_image() #[:, :, ::-1]
            cv2.rectangle(current_image_left, (after_box.cx-15, after_box.cy-5), (after_box.cx+15, after_box.cy+5), (255, 0, 0), -1) # fill 
            current_image_left = current_image_left[:,:,::-1]
            
            
            gpt_history = []
            ver_raw_msg, ver_response = single_image_ask_gpt_movement(current_image_left, v_sys_msg, v_query_msg, gpt_history)
            ver_action, summarized_ver_action = parse_to_action(ver_raw_msg, v_sum_msg, gpt_history, scale_factor=scale_factors, scale_factor_idx=0)

            
            combined_action = (np.array(hor_action) + np.array(ver_action) + np.array(top_hor_action)).tolist()
            combined_action[0] = combined_action[0] * step_decay #-1 * 
            combined_action[1] = combined_action[1] * step_decay #-1 * 
            combined_action[2] = combined_action[2] * step_decay
            print(f"Combined action: {combined_action}")

            save_logs(seed, step, log_save_dir, hor_raw_msg, summarized_hor_action, ver_raw_msg, summarized_ver_action, top_hor_raw_msg, summarized_top_hor_action, combined_action, current_image_left, current_image_top)
            
            cmd = "##step relative##: "
            cmd += str(combined_action)
            parent_conn.send(cmd)  # Send command through the pipe
            response = parent_conn.recv()  # Wait for the response
            print(f"Response from robot controller: {response}")
            
            # # final image save
            # if step == max_steps-1:
            #     after_image = get_latest_frame(pipeline_left)
            #     predictions, visualized_output = demo.run_on_image(after_image)
            #     after_image = visualized_output.get_image()[:, :, ::-1]
            #     save_logs(seed, step+1, log_save_dir, hor_raw_msg, summarized_hor_action, ver_raw_msg, summarized_ver_action, combined_action, after_image_left, after_image_top)
                
            step_decay *= decay_ratio

    parent_conn.send("##backB##")
    reset_success = parent_conn.recv()
    print(f"Reset success: {reset_success}")  
    print(f"Response from robot controller: {response}")
    parent_conn.send("exit")  # Send exit command
    end_pipeline(pipeline_left)
    p.join()  # Wait for the subprocess to finish

from table_push_default_cfg_final import prompt_configs
from datetime import datetime
import argparse
import os 
import pdb
EXP_BASE_DIR = "/home/hongyic/Documents/table_top_push/llm_table_top_push/automate_push_logs"
os.makedirs(EXP_BASE_DIR, exist_ok=True)
if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("-e","--exp_name",type=str,default="test")
    parser.add_argument("-m","--max_steps",type=int,default=8) 
    parser.add_argument("-s","--seeds", type=int, default=1)
    parser.add_argument("-r","--rows", type=int, default=1)
    parser.add_argument("-c","--cols", type=int, default=1)
    parser.add_argument("-fmt","--fake_marker_type", default="hollow", type=str)  
    parser.add_argument("-rt", "--rotation", type=int, default=None)
    parser.add_argument("-pt","--prompt_type", type=str, default="combine_3d_grasp_prompt")
    parser.add_argument("-st","--seperate_stop_query", action="store_true")
    args_exp = parser.parse_args()
    cur_time_str=  datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_save_dir = os.path.join(EXP_BASE_DIR, f"{cur_time_str}_{args_exp.exp_name}")
    os.makedirs(exp_save_dir, exist_ok=True)
    img_save_dir = os.path.join(exp_save_dir, "images")
    os.makedirs(img_save_dir, exist_ok=True)
    selected_prompt_config = prompt_configs.get(args_exp.prompt_type, None)
    #print(selected_prompt_config)
    if selected_prompt_config is None:
        raise ValueError("Prompt type not found")
    if args_exp.fake_marker_type is not None:
        for key in selected_prompt_config:
            try: 
                selected_prompt_config[key] = selected_prompt_config[key].format(FILL_TYPE=args_exp.fake_marker_type)
            except:
                pass 
    if args_exp.seperate_stop_query:
        stop_query_config = prompt_configs["stop_config"]
        for key in stop_query_config:
            try: 
                stop_query_config[key] = stop_query_config[key].format(FILL_TYPE=args_exp.fake_marker_type)
            except:
                pass
        selected_prompt_config.update(stop_query_config)
    print(selected_prompt_config)
    logging_dict = {
        "num_seeds": args_exp.seeds,
        "prompt_config": selected_prompt_config,
        "communications": {}
    }
    connection_args = {
        "ip" : "192.168.1.10",
        "username": "admin",
        "password": "admin"
    }
    run_callee(selected_prompt_config, logging_dict, exp_save_dir, connection_args, args_exp.max_steps, args_exp.seeds, args_exp.fake_marker_type, args_exp.rows, args_exp.cols, args_exp.rotation)
    
    with open(f"{exp_save_dir}/comm_log.json", "w") as f:
        json.dump(logging_dict, f, indent=4)
        