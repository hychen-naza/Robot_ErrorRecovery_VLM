import base64 
import pathlib 
import cv2 
from time import sleep
import requests

def get_api_key(key_file="openai_key.txt"):
    openai_key_file = pathlib.Path(__file__).parent / key_file
    with open(openai_key_file, "r") as f:
        openai_key = f.read().strip()
    return openai_key

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def encode_numpy_image(np_image):
    im_arr = cv2.imencode('.jpg', np_image)[1]
    im_bytes = im_arr.tobytes()
    return base64.b64encode(im_bytes).decode('utf-8')

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
  
def make_gpt_call_request(gpt_msg_history, openai_api_key, model_type="gpt-4o"):
    assert model_type in ["gpt-4o", "gpt-4-vision-preview"]
    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }
    payload = {
        "model": model_type,
        "messages": gpt_msg_history,
        "max_tokens": 512
    }
            
    sent = False
    while not sent:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=header, json=payload)
        if "error" not in response.json():
            sent = True 
        else:
            print("Error found in response, retrying in 60 seconds")
            print(response.json())
            sleep(60)
    return response



def ask_gpt(query_message, image_paths=None,gpt_history=None, system_message=None, openai_key=None, image_folder=None):
    assert openai_key is not None, "openai_key is required"
    import os
    import pdb 
    #pdb.set_trace()
    encoded_images = [encode_image(os.path.join(image_folder, image_path)) for image_path in image_paths] if image_paths is not None else None
    if gpt_history is None or len(gpt_history) == 0:
        assert system_message is not None, "system_message is required if gpt_history is None"
        gpt_history = []
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
    
    query_payload = {
        "role": "user",
        "content": [
        ]
    }
    if encoded_images is not None:
        for encoded_image in encoded_images:
            query_payload["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
            })
    query_payload["content"].append({
        "type": "text",
        "text": query_message
    })
    
    gpt_history.append(query_payload)
    response = make_gpt_call_request(gpt_history, openai_key)
    response_content = response.json()['choices'][0]['message']['content']
    gpt_history.append(response.json()['choices'][0]['message'])
    return gpt_history, response_content, response
