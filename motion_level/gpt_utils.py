import base64 
import numpy as np 
import requests 
from time import sleep
import cv2

import shapely.geometry
import shapely.affinity

class RotatedRect:
    def __init__(self, cx, cy, w, h, angle):
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h
        self.angle = angle

    def get_contour(self):
        w = self.w
        h = self.h
        c = shapely.geometry.box(-w/2.0, -h/2.0, w/2.0, h/2.0)
        rc = shapely.affinity.rotate(c, self.angle)
        return shapely.affinity.translate(rc, self.cx, self.cy)

    def intersection(self, other):
        return self.get_contour().intersection(other.get_contour())

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def encode_numpy_image(np_image):
    im_arr = cv2.imencode('.jpg', np_image)[1]
    im_bytes = im_arr.tobytes()
    return base64.b64encode(im_bytes).decode('utf-8')

def parse_yes_no_negate(message):
    if "(yes)" in message.lower():
        return False
    elif "(no)" in message.lower():
        return True
    else:
        return False


def parse_reward_response(response):
    reward_split = response.split("reward (")[-1]
    reward_val = reward_split.split(")")[0].strip()
    try:
        reward_numeric = float(reward_val)
    except:
        reward_numeric = 0
    return int(np.round(reward_numeric))

def hor_to_vert_translate(direction, clock_wise=True):
    if clock_wise:
        if "up" in direction:
            return "move left"
        elif "down" in direction:
            return "move right"
        elif "stop" in direction:
            return "stop"
    else:
        if "up" in direction:
            return "move right"
        elif "down" in direction:
            return "move left"
        elif "stop" in direction:
            return "stop"
    return None
            
def get_system_msg(system_msg_text):
    system_msg = {
        "role": "system",
        "content": [
            {
            "type": "text",
            "text": system_msg_text
            },
        ]
    }
    return system_msg 
  
def make_gpt_call(gpt_msg_history, openai_api_key):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }
    payload = {
        "model": "gpt-4o",
        "messages": gpt_msg_history,
        "max_tokens": 512
    }
            
    sent = False
    while not sent:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        if "error" not in response.json():
            sent = True 
        else:
            print("Error found in response, retrying in 60 seconds")
            print(response.json())
            sleep(10)
    return response