# caller.py
from multiprocessing import Process, Pipe
from table_top_push import kinova_control
from realsense_color import start_pipeline, end_pipeline, get_latest_frame
from gpt_utils import make_gpt_call, encode_numpy_image, RotatedRect
import numpy as np 
import cv2
import json 
import pathlib
import math
from waypoints import CENTER_POS
import re 
 
SYSTEM_MESSAGE_PLACEHOLDER = "SYSTEM_MESSAGE_PLACEHOLDER"
LOCATION_MESSAGE_PLACEHOLDER = "LOCATION_MESSAGE_PLACEHOLDER"
SUMMARY_MESSAGE_PLACEHOLDER = "SUMMARY_MESSAGE_PLACEHOLDER"
DEFAULT_ROT_DEGREE = 20

openai_key_file = pathlib.Path(__file__).parent / "openai_key.txt"
with open(openai_key_file, "r") as f:
    openai_key = f.read().strip()
    
# 298, 214; 
# size: 640, 480
corner_offset = np.array((640//2 - 294, 480//2 - 214))
step_size = 0.04
ACTION_NAME_TO_MOVEMENT={
    "move up": [step_size, 0],
    "upward": [step_size, 0],
    "move down": [-step_size, 0],
    "downward": [-step_size, 0],
    "move left": [0, step_size],
    "left": [0, step_size],
    "move right": [0, -step_size],
    "right": [0, -step_size],
    "stop": [0, 0]
}

LOWER_THRESH = np.array([80,0,0])
UPPER_THRESH = np.array([179, 255, 99])

TARGET_XY = [46*0.01,2.2*0.01]

max_x = 74 * 0.01 #74
max_y = 38 * 0.01 #40
min_x = 24 * 0.01 #24
min_y = -38 * 0.01 #-36

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

def sample_around_target_xy_regular(x_min, x_max, y_min, y_max, target_center, target_center_deadzone, num_rows, num_cols, row_deadzone_factor=1):
    quad1_top_right, quad1_bottom_left = [x_max, y_max], [target_center[0]+target_center_deadzone/row_deadzone_factor, target_center[1]+target_center_deadzone]
    quad2_top_right, quad2_bottom_left = [target_center[0]-target_center_deadzone/row_deadzone_factor, y_max], [x_min, target_center[1]+target_center_deadzone]
    quad3_top_right, quad3_bottom_left = [target_center[0]-target_center_deadzone/row_deadzone_factor, target_center[1]-target_center_deadzone], [x_min, y_min]
    quad4_top_right, quad4_bottom_left = [x_max, target_center[1]-target_center_deadzone], [target_center[0]+target_center_deadzone/row_deadzone_factor, y_min]
    
    quad1_samples = sample_points_in_rectangle_regular(quad1_top_right, quad1_bottom_left, num_rows, num_cols)
    quad2_samples = sample_points_in_rectangle_regular(quad2_top_right, quad2_bottom_left, num_rows, num_cols)
    quad3_samples = sample_points_in_rectangle_regular(quad3_top_right, quad3_bottom_left, num_rows, num_cols)
    quad4_samples = sample_points_in_rectangle_regular(quad4_top_right, quad4_bottom_left, num_rows, num_cols)
    
    # get horizontal points 
    largest_y = max([quad1_top_right[1], quad2_top_right[1], quad3_top_right[1], quad4_top_right[1]])
    smallest_y = min([quad1_bottom_left[1], quad2_bottom_left[1], quad3_bottom_left[1], quad4_bottom_left[1]])
    hor_points_1 = (target_center[0], largest_y)
    hor_points_2 = (target_center[0], smallest_y)
    
    
    #return quad3_samples + [hor_points_1, hor_points_2]
    return  quad1_samples + quad2_samples + quad3_samples + quad4_samples + [hor_points_1, hor_points_2]

def sample_in_rect(x_min, x_max,y_min,y_max, num_samples, min_dist_to_target=0.4):
    samples = []
    for i in range(num_samples):
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        while np.abs(x-TARGET_XY[0]) < min_dist_to_target:
            x = np.random.uniform(x_min, x_max)
        while np.abs(y-TARGET_XY[1]) < min_dist_to_target:
            y = np.random.uniform(y_min, y_max)
        relative_x = x - TARGET_XY[0]
        relative_y = y - TARGET_XY[1]
        samples.append([relative_x, relative_y])
    return samples

def sample_around_target_xy(x_min,x_max, y_min,y_max, samples_per_quadrant=5):
    top_left_samples = sample_in_rect(x_min, TARGET_XY[0], TARGET_XY[1], y_max, samples_per_quadrant)
    top_right_samples = sample_in_rect(TARGET_XY[0], x_max, TARGET_XY[1], y_max, samples_per_quadrant)
    bottom_left_samples = sample_in_rect(x_min, TARGET_XY[0], y_min, TARGET_XY[1], samples_per_quadrant)
    bottom_right_samples = sample_in_rect(TARGET_XY[0], x_max, y_min, TARGET_XY[1], samples_per_quadrant)
    final_list = []
    final_list.extend(top_left_samples)
    final_list.extend(top_right_samples)
    final_list.extend(bottom_left_samples)
    final_list.extend(bottom_right_samples)
    return final_list
    

def sample_point_rectangle(width, height, minimum_w_disp=0.15, minimum_h_disp=0.15):
    x = np.random.uniform(-width/2, width/2)
    y = np.random.uniform(-height/2, height/2)
    while np.abs(x) < minimum_w_disp:
        x = np.random.uniform(-width/2, width/2)
    while np.abs(y) < minimum_h_disp:
        y = np.random.uniform(-height/2, height/2)
    return [x, y]

def parse_discrete_action(action_response, scale_factor=1.0):
    if not action_response:
        return (0, 0)
    
    # Regular expression to find tuples of the form (a,b) where a and b are numbers
    pattern = r'\(\s*(left|right)\s*,\s*(upward|downward)\s*\)'

    match = re.search(pattern, action_response)
    if match:
        a, b = match.groups()
        return (ACTION_NAME_TO_MOVEMENT[a][1] * scale_factor, ACTION_NAME_TO_MOVEMENT[b][0] * scale_factor)
    
    return (0, 0)
 
def parse_direct_action(action_response, scale_factor=1.0, clip = 0.04):
    if not action_response:
        return (0, 0)
    
    # Regular expression to find tuples of the form (a,b) where a and b are numbers
    pattern = r'\(\s*([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)\s*\)'
    
    match = re.search(pattern, action_response)
    if match:
        a, b = match.groups()
        return (np.clip(float(a), -clip, clip) * scale_factor, np.clip(float(b), -clip, clip)* scale_factor)
    
    return (0, 0)


def simple_parse_action(summarized_action, scale_factor=1.0):
    print(summarized_action)
    final_action = [0, 0]
    encountered_similar = False 
    if "upward" in summarized_action.lower() or "above" in summarized_action.lower():
        final_action = ACTION_NAME_TO_MOVEMENT["move up"]
    elif "downward" in summarized_action.lower() or "below" in summarized_action.lower():
        final_action = ACTION_NAME_TO_MOVEMENT["move down"]
    elif "left" in summarized_action.lower():
        final_action = ACTION_NAME_TO_MOVEMENT["move left"]
    elif "right" in summarized_action.lower():
        final_action = ACTION_NAME_TO_MOVEMENT["move right"]
    elif "stop" in summarized_action.lower() or "similar" in summarized_action.lower():
        final_action = ACTION_NAME_TO_MOVEMENT["stop"]
        encountered_similar = True 
    else:
        final_action = ACTION_NAME_TO_MOVEMENT["stop"]  
    final_action_scaled = [final_action[0]*scale_factor, final_action[1]*scale_factor]
    if encountered_similar:
        print("Encountered similar")
    return final_action_scaled, encountered_similar
    
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

def single_image_ask_gpt_movement(image, system_message, query_text_message, gpt_history):
    encoded_image = encode_numpy_image(image)
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
                "text": query_text_message
            },
        ]
    }
    
    gpt_history.append(query_message)
    response = make_gpt_call(gpt_history, openai_key)
    response_content = response.json()['choices'][0]['message']['content']
    gpt_history.append(response.json()['choices'][0]['message'])
    return response_content, response
    
def rough_crop_center_image(image, crop_size = (160,120)):
    center_x = image.shape[1]//2
    center_y = image.shape[0]//2
    crop_top_left = (center_x - crop_size[0]//2, center_y - crop_size[1]//2)
    crop_bottom_right = (center_x + crop_size[0]//2, center_y + crop_size[1]//2)
    cropped_image = image[crop_top_left[1]:crop_bottom_right[1], crop_top_left[0]:crop_bottom_right[0]]
    # scale back to original size
    cropped_image = cv2.resize(cropped_image, (image.shape[1], image.shape[0]))
    return cropped_image

def parse_to_action(response, summarized_prompt, gpt_history, no_history=False, scale_factor=None, scale_factor_idx=0, directly_apply_simple=False, combined_relative=False):
    local_history = gpt_history    
    if no_history:
        local_history = []
    if summarized_prompt is not None:
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
    else:
        if directly_apply_simple:
            if not combined_relative:
                summarized_action_msg = response
                action_value = parse_direct_action(response, scale_factor=scale_factor[scale_factor_idx])
                summarized_action_msg = False
            else:
                summarized_action_msg = response
                action_value = parse_discrete_action(response, scale_factor=scale_factor[scale_factor_idx])
                summarized_action_msg = False
        else: 
            summarized_action_msg = response
            action_value, enountered_similar = simple_parse_action(summarized_action_msg, scale_factor=scale_factor[scale_factor_idx])
    # if enountered_similar and scale_factor is not None:
    #     if scale_factor[scale_factor_idx] == 1.0:
    #         scale_factor[scale_factor_idx] = 0.2
    return action_value, summarized_action_msg

def detect_grasp_object(current_image):
    current_image_copy = current_image.copy()
    lower_pink = LOWER_THRESH
    upper_pink = UPPER_THRESH
    
    frame_threshed = cv2.inRange(current_image_copy, lower_pink, upper_pink)

    # apply morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    clean = cv2.morphologyEx(frame_threshed, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

    # get external contours
    contours = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    if len(contours) == 0:
        return current_image_copy, None, 0

    c = max(contours, key=cv2.contourArea)

    # get rotated rectangle from contour
    rot_rect = cv2.minAreaRect(c)
    rotation = rot_rect[2]
    box = cv2.boxPoints(rot_rect)
    box = np.int0(box)
    # draw rotated rectangle on copy of img
    cv2.drawContours(current_image_copy, [box], 0, (255, 255, 255), 2)
    current_box = RotatedRect(int((box[0][0]+box[1][0]+box[2][0]+box[3][0])/4), 
                              int((box[0][1]+box[1][1]+box[2][1]+box[3][1])/4), 
                              int(math.dist(box[0], box[1])), 
                              int(math.dist(box[0], box[1])), 
                              rotation)
    box_area = int(math.dist(box[0], box[1])) ** 2
    return current_image_copy, current_box, box_area

def run_callee(prompt_configuration,loggin_dict, log_save_dir, connection_kwargs, max_steps=10, num_seeds=1, fake_marker_type=None, rows=1, cols=1, rotation_type=None, width=640, height=480):
    
    h_sys_msg = prompt_configuration["hor_sys_msg"]
    h_query_msg = prompt_configuration["horizontal_prompt"]
    h_sum_msg = prompt_configuration["horizontal_summary_prompt"]
    v_sys_msg = prompt_configuration["ver_sys_msg"]
    v_query_msg = prompt_configuration["vertical_prompt"]
    v_sum_msg = prompt_configuration["vertical_summary_prompt"]
    
    s_sys_msg = prompt_configuration.get("stop_sys_msg", None)
    hor_stop_query_msg = prompt_configuration.get("hor_stop_query_msg", None)
    ver_stop_query_msg = prompt_configuration.get("ver_stop_query_msg", None)
    hor_stop_sum_msg = prompt_configuration.get("hor_stop_sum_msg", None)
    ver_stop_sum_msg = prompt_configuration.get("ver_stop_sum_msg", None)
    

    regular_xys = sample_around_target_xy_regular(min_x, max_x, min_y, max_y, TARGET_XY, 0.15, rows,cols, row_deadzone_factor=1)

    print(regular_xys)
    parent_conn, child_conn = Pipe()  # Create a pipe for IPC
    is_rotate = rotation_type is not None
    p = Process(target=kinova_control, args=(child_conn,connection_kwargs, is_rotate, ))  # Pass the child connection to the subprocess
    # get pid of the process
    p.start()
    pipeline = start_pipeline(width, height)
    
    width_offset_ratio = width/640
    height_offset_ratio = height/480
    
    topleft = np.array([int(width//2 - corner_offset[0] * width_offset_ratio), int(height//2 - corner_offset[1] * height_offset_ratio)])
    bottomright = np.array([int(width//2 + corner_offset[0] * width_offset_ratio), int(height//2 + corner_offset[1] * height_offset_ratio)])
    
    # cx, cy, w, h, angle
    target_box = RotatedRect(int((bottomright[0]+topleft[0])/2), int((bottomright[1]+topleft[1])/2), bottomright[0]-topleft[0], bottomright[1]-topleft[1], 0)
    
    def save_logs(seed,step, log_save_dir, hor_raw_msg, summarized_hor_action, ver_raw_msg, summarized_ver_action, combined_action, current_image, bb_image, raw_image, coverage, pixel_dist, stop_responses=None, stop_sums=None):
        # logging 
        
        img_save_dir = f"{log_save_dir}/images_s{seed}/step_{step}.png"
        bb_img_save_dir = f"{log_save_dir}/images_s{seed}/bb_step_{step}.png"
        raw_img_save_dir = f"{log_save_dir}/images_s{seed}/raw_step_{step}.png"
        print(f"img_save_dir {img_save_dir}")
        cv2.imwrite(img_save_dir, current_image)
        cv2.imwrite(bb_img_save_dir, bb_image)
        cv2.imwrite(raw_img_save_dir, raw_image)
        loggin_dict["positions"] = regular_xys
        if seed not in loggin_dict["communications"]:
            loggin_dict["communications"][seed]={}
        loggin_dict["communications"][seed][step] = {
            "horizontal_response": hor_raw_msg,
            "horizontal_action": summarized_hor_action,
            "vertical_response": ver_raw_msg,
            "vertical_action": summarized_ver_action,
            "combined_action": combined_action,
            "image_dir": img_save_dir,
            "bb_image_dir": bb_img_save_dir,
            "coverage": coverage,
            "pixel_dist": pixel_dist,
            "stop_responses": stop_responses,
            "stop_sums": stop_sums
        }
        with open(f"{log_save_dir}/comm_log_s{seed}.json", "w") as f:
            json.dump(loggin_dict, f, indent=4)
    
    decay_ratio = prompt_configuration.get("decay_ratio", 0.9)
    coverages_all = []
    dists_all = []
    for loc_id, rand_location in enumerate(regular_xys):
        for seed in range(num_seeds):
            # rand_location[1] *= -0.5
            # rand_location[0] *= -1 #0.5
            coverages = []
            dists = []
            step_decay = 1.0
            # query if reset is needed
            parent_conn.send("##query reset##")  # Send command through the pipe
            response = parent_conn.recv()  # Wait for the response
            if response == "reseted: False":
                print("Reset is needed")
                parent_conn.send("##reset##")
                reset_success = parent_conn.recv()
                print(f"Reset success: {reset_success}")  
            # random initial position
            test_img = get_latest_frame(pipeline)
            horizon_init_loc = [rand_location[0], 0.0]
            vertical_init_loc = [0.0, rand_location[1]]
            
            inv_horizon_init_loc = [-rand_location[0], 0.0]
            inv_vertical_init_loc = [0.0, -rand_location[1]]
            
            cmd = f"##step relative##: {vertical_init_loc}"
            # Send commands to the callee function
            parent_conn.send(cmd)
            init_success = parent_conn.recv()
            print(f"Initial movement success: {init_success}")
            
            cmd = f"##step relative##: {horizon_init_loc}"
            # Send commands to the callee function
            parent_conn.send(cmd)
            init_success = parent_conn.recv()
            print(f"Initial movement success: {init_success}")
            
            # Send command through the pipe
            # action scale factor 
            scale_factors = [1.0, 1.0]
            loc_log_save_dir = f"{log_save_dir}/loc{loc_id}"
            os.makedirs(f"{loc_log_save_dir}/images_s{seed}", exist_ok=True)
            stop_count = 0
            
        
            for step in range(max_steps):
                
                current_image = get_latest_frame(pipeline)
                current_image_copy = current_image.copy()
                print(f"step {step}, fake_marker_type {fake_marker_type}")
                if prompt_configuration.get("visual_opt_type", "marker") in ["both", "opt"]:
                    cv2.rectangle(current_image, (topleft[0]+int((bottomright[0]-topleft[0])/4), topleft[1]+int((bottomright[1]-topleft[1])/4)), (topleft[0]+int((bottomright[0]-topleft[0])*0.75), topleft[1]+int((bottomright[1]-topleft[1])*0.75)), (0, 0, 255), -1) # fill   
                if prompt_configuration.get("visual_opt_type", "marker")  in ["both", "marker"]:
                    cv2.rectangle(current_image, (topleft[0], topleft[1]), (bottomright[0], bottomright[1]), (0,255,0), 3)
     
                gpt_history = []
                hor_raw_msg, hor_response = single_image_ask_gpt_movement(current_image, h_sys_msg, h_query_msg, gpt_history)
                hor_action, summarized_hor_action = parse_to_action(hor_raw_msg, h_sum_msg, gpt_history, scale_factor=scale_factors, scale_factor_idx=0, directly_apply_simple=prompt_configuration.get("directly_apply_simple", False), combined_relative=prompt_configuration.get("combined_relative", False))
                ver_raw_msg, summarized_ver_action = None, None
                if v_query_msg is not None:
                    gpt_history = []
                    ver_raw_msg, ver_response = single_image_ask_gpt_movement(current_image, v_sys_msg, v_query_msg, gpt_history)
                    ver_action, summarized_ver_action = parse_to_action(ver_raw_msg, v_sum_msg, gpt_history, scale_factor=scale_factors, scale_factor_idx=1,directly_apply_simple=prompt_configuration.get("directly_apply_simple", False), combined_relative=prompt_configuration.get("combined_relative", False))
                    
                    combined_action = (np.array(hor_action) + np.array(ver_action)).tolist()
                    combined_action[0] = combined_action[0] * step_decay #-1 * 
                    combined_action[1] = combined_action[1] * step_decay #-1 * 
                
                else:
                    combined_action = (hor_action[0] * step_decay, hor_action[1] * step_decay)
                print(f"Combined action: {combined_action}")

                after_image, after_box, box_area = detect_grasp_object(current_image_copy)
                if after_box is not None:
                    cv2.rectangle(after_image, (topleft[0], topleft[1]), (bottomright[0], bottomright[1]), (0, 255, 0), 3)
                    cv2.rectangle(after_image, (topleft[0]+int((bottomright[0]-topleft[0])/4), topleft[1]+int((bottomright[1]-topleft[1])/4)), (topleft[0]+int((bottomright[0]-topleft[0])*0.75), topleft[1]+int((bottomright[1]-topleft[1])*0.75)), (0, 0, 255), -1) # fill
                    dist = math.dist([after_box.cx, after_box.cy], [target_box.cx, target_box.cy])
                    dists.append(dist)
                    print(f"Distance to target: {dist}")
                    coverage = after_box.intersection(target_box).area / box_area
                    print(f"coverage {coverage}")
                    coverages.append(coverage)

                else: 
                    dist = -1
                    dists.append(dist)
                    coverage = -1
                    coverages.append(coverage)
             
                cmd = "##step relative##: "
                cmd += str(combined_action)
                parent_conn.send(cmd)  # Send command through the pipe
                response = parent_conn.recv()  # Wait for the response
                print(f"Response from robot controller: {response}")

                stop_raw_msg, stop_response = None, None
                stop_episode_hor, stop_episode_ver = False, False

                stop_episode = False
                hor_stop_raw_msg, ver_stop_raw_msg = None, None
                if s_sys_msg is not None and hor_stop_query_msg is not None and ver_stop_query_msg is not None and hor_stop_sum_msg is not None and ver_stop_sum_msg is not None:
                    gpt_history = []
                    hor_stop_raw_msg, stop_response = single_image_ask_gpt_movement(current_image, s_sys_msg, hor_stop_query_msg, gpt_history)
                    stop_episode_hor = simple_stop_query(gpt_history, hor_stop_sum_msg) #simple_stop_query(gpt_history)
                    gpt_history = []
                    ver_stop_raw_msg, stop_response = single_image_ask_gpt_movement(current_image, s_sys_msg, ver_stop_query_msg, gpt_history)
                    stop_episode_ver = simple_stop_query(gpt_history, ver_stop_sum_msg) #simple_stop_query(gpt_history)
                    stop_episode = stop_episode_hor and stop_episode_ver

                save_logs(seed, step, loc_log_save_dir, hor_raw_msg, summarized_hor_action, ver_raw_msg, summarized_ver_action, combined_action, current_image=current_image, bb_image=after_image, raw_image=current_image_copy, coverage=coverage, pixel_dist=dist, stop_responses=[hor_stop_raw_msg, ver_stop_raw_msg], stop_sums=[stop_episode_hor, stop_episode_ver])

                # final image save
                if step == max_steps-1 or stop_episode:
                    current_image = get_latest_frame(pipeline)
                    after_image, after_box, box_area = detect_grasp_object(current_image)
                    cv2.rectangle(after_image, (topleft[0], topleft[1]), (bottomright[0], bottomright[1]), (0,255,0), 3)
                    cv2.rectangle(after_image, (topleft[0]+int((bottomright[0]-topleft[0])/4), topleft[1]+int((bottomright[1]-topleft[1])/4)), (topleft[0]+int((bottomright[0]-topleft[0])*0.75), topleft[1]+int((bottomright[1]-topleft[1])*0.75)), (0, 0, 255), -1) # fill
                    dist = math.dist([after_box.cx, after_box.cy], [target_box.cx, target_box.cy])
                    print(f"Distance to target: {dist}")
                    coverage = after_box.intersection(target_box).area / box_area
                    print(f"coverage {coverage}")
                    coverages.append(coverage)
                    save_logs(seed, step+1, loc_log_save_dir, hor_raw_msg, summarized_hor_action, ver_raw_msg, summarized_ver_action, combined_action, current_image=current_image, bb_image=after_image, raw_image=current_image_copy, coverage=coverage, pixel_dist=dist)
                    coverages_all.append(coverages)
                    dists_all.append(dists)
                    print(coverages_all)
                    print(dists_all)
                
                if stop_episode:
                    break
                step_decay *= decay_ratio
            

    parent_conn.send("##reset##")
    reset_success = parent_conn.recv()
    print(f"Reset success: {reset_success}")  
    print(f"Response from robot controller: {response}")
    parent_conn.send("exit")  # Send exit command
    end_pipeline(pipeline)
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
    parser.add_argument("-m","--max_steps",type=int,default=10) 
    parser.add_argument("-s","--seeds", type=int, default=1)
    parser.add_argument("-r","--rows", type=int, default=1)
    parser.add_argument("-c","--cols", type=int, default=1)
    parser.add_argument("-fmt","--fake_marker_type", default="hollow", type=str)  
    parser.add_argument("-rt", "--rotation", type=int, default=None)
    parser.add_argument("-pt","--prompt_type", type=str, default="relative_simple_prompt")
    parser.add_argument("-st","--seperate_stop_query", action="store_true")
    args_exp = parser.parse_args()
    cur_time_str=  datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_save_dir = os.path.join(EXP_BASE_DIR, f"{cur_time_str}_{args_exp.exp_name}")
    os.makedirs(exp_save_dir, exist_ok=True)
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
    run_callee(selected_prompt_config, logging_dict, log_save_dir=exp_save_dir, connection_kwargs=connection_args, max_steps=args_exp.max_steps, num_seeds=args_exp.seeds, rows=args_exp.rows, cols=args_exp.cols, rotation_type=args_exp.rotation)
    
    with open(f"{exp_save_dir}/comm_log.json", "w") as f:
        json.dump(logging_dict, f, indent=4)