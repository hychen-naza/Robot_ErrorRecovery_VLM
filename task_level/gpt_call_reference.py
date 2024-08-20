import json
import os 
from copy import deepcopy
from datetime import datetime
from utils import *
import pdb 

CURRENT_TIME=datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
IMAGE_FOLDER = None

def llm_error_correction_detection_only(openai_key, prompts, image_path, correct_image_path=None, gpt_history=None):
    gpt_history = []
    error_detection = prompts["error_detection"]
    system_msg = prompts["system_msg"]
    if not isinstance(image_path, list):
        image_path = [image_path]
    if correct_image_path is not None and prompts.get("correct_example",None) is not None:
        if not isinstance(correct_image_path, list):
            correct_image_path = [correct_image_path]
        correct_example = prompts["correct_example"]
        gpt_history, response_content, response = ask_gpt(correct_example, correct_image_path, gpt_history, system_msg, openai_key, IMAGE_FOLDER)
    gpt_history, response_content, response = ask_gpt(error_detection, image_path, gpt_history, system_msg, openai_key, IMAGE_FOLDER)
    return response_content

def llm_error_correction_detection_correction(openai_key, system_msg, error_detection, error_correction, image_path, correct_image_path=None, gpt_history=None):
    pass 

def llm_error_correction_detection_description(openai_key, system_msg, error_detection, error_type_classification, image_path, correct_image_path=None, gpt_history=None):
    pass 

def llm_error_correction_detection_description_correction(openai_key, system_msg, error_detection, error_type_classification, error_correction, image_path, correct_image_path=None, gpt_history=None):
    pass 

def parse_case_folder(case_folder):
    case_dict = {}
    prompt_file = os.path.join(case_folder, "prompt.json")
    if not os.path.exists(prompt_file):
        print(f"prompt.json not found in {case_folder}")
        return None
    with open(prompt_file, "r") as f:
        prompt = json.load(f)
    case_dict["prompt"] = prompt
    # correct image is the only image in "correct_image" folder
    correct_image_dir = os.path.join(case_folder, "correct_image")
    correct_image = None
    if os.path.exists(correct_image_dir):
        correct_image = os.path.join(correct_image_dir, os.listdir(correct_image_dir)[0])
    case_dict["correct_image"] = correct_image
    error_images = os.path.join(case_folder, "error_images")
    case_dict["error_images"] = [os.path.join(error_images, img_name) for img_name in os.listdir(error_images)]
    return case_dict

def smart_ask_llm_error_correction(prompts, error_image, correct_image, openai_key, gpt_history=None):
    # system_msg, error_detection, error_type_classification, error_correction = prompts
    if prompts.get("error_type_classification",None) is not None:
        if prompts.get("error_correction", None) is not None:
            return llm_error_correction_detection_description_correction(openai_key, prompts, error_image, correct_image, gpt_history)
        return llm_error_correction_detection_description(openai_key, prompts, error_image, correct_image, gpt_history)
    else:
        if prompts.get("error_correction",None) is not None:
            return llm_error_correction_detection_correction(openai_key, prompts, error_image, correct_image, gpt_history)
        return llm_error_correction_detection_only(openai_key, prompts, error_image, correct_image, gpt_history)
    

def run_case_experiment(case_folder, openai_key, repeat_num=5, save_to_case_dir=True):
    case_dict = parse_case_folder(case_folder)
    if case_dict is None:
        return None
    local_dict = {}
    local_dict["input"] = deepcopy(case_dict)
    # make the image directories local
    local_dict["input"]["correct_image"] = "/".join(local_dict["input"]["correct_image"].split("/")[-2:])
    local_dict["input"]["error_images"] = ["/".join(img.split("/")[-2:]) for img in local_dict["input"]["error_images"]]

    local_dict["output"] = {}
    # error image can have multiple variations
    for erro_img_variation in case_dict["error_images"]:
        erro_img_variation_last_two = "/".join(erro_img_variation.split("/")[-2:])
        local_dict["output"][erro_img_variation_last_two] = []
        for i in range(repeat_num):
            response = smart_ask_llm_error_correction(case_dict["prompt"], erro_img_variation, case_dict["correct_image"], openai_key)
            local_dict["output"][erro_img_variation_last_two].append(response)
    if save_to_case_dir:
        case_output_dir = os.path.join(case_folder, "output")
        os.makedirs(case_output_dir, exist_ok=True)
        case_output_file = os.path.join(case_output_dir, f"response_c_{CURRENT_TIME}.json")
        with open(case_output_file, "w") as f:
            json.dump(local_dict, f, indent=4)
    return local_dict

def run_step_experiment(step_folder, openai_key, repeat_num=5, save_to_step_dir=True, selected_case=None):
    step_dict = {}
    for case_dir in os.listdir(step_folder):
        if selected_case is not None and case_dir not in selected_case:
            continue
        full_case_dir = os.path.join(step_folder, case_dir)
        response = run_case_experiment(full_case_dir, openai_key, repeat_num, save_to_step_dir)
        if response is not None:
            step_dict[case_dir] = response
    if save_to_step_dir:
        step_output_dir = os.path.join(step_folder, "output")
        os.makedirs(step_output_dir, exist_ok=True)
        step_output_file = os.path.join(step_output_dir, f"response_s_{CURRENT_TIME}.json")
        with open(step_output_file, "w") as f:
            json.dump(step_dict, f, indent=4)
    return step_dict

def run_experiment(exp_folder, openai_key, repeat_num=5, save_to_exp_dir=True, selected_step=None, selected_case=None):
    exp_dict = {}
    for steps in os.listdir(exp_folder):
        if selected_step is not None and steps not in selected_step:
            continue
        step_full_dir = os.path.join(exp_folder, steps)
        response = run_step_experiment(step_full_dir, openai_key, repeat_num, save_to_exp_dir, selected_case)
        exp_dict[steps] = response
    if save_to_exp_dir:
        exp_output_dir = os.path.join(exp_folder, "output")
        os.makedirs(exp_output_dir, exist_ok=True)
        exp_output_file = os.path.join(exp_output_dir, f"response_e_{CURRENT_TIME}.json")
        with open(exp_output_file, "w") as f:
            json.dump(exp_dict, f, indent=4)
    return exp_dict

def parse_standard_prompt(prompt_dict):
    system_msg = prompt_dict["system_msg"]
    error_detection = prompt_dict["error_detection"]
    error_type_classification = prompt_dict.get("error_type_classification", None)
    error_correction = prompt_dict.get("error_correction", None)
    return system_msg, error_detection, error_type_classification, error_correction
 
def check_dir_hierarchy(target_folder):
    # it is case folder if it directly contains prompt.json
    # it is step folder if it directly contains at least one case folders
    # it is experiment folder if it directly contains at least one step folder
    # it is invalid if it does not contain any of the above
    if "prompt.json" in os.listdir(target_folder):
        return "case"
    for item in os.listdir(target_folder):
        sub_dir = os.path.join(target_folder, item)
        if os.path.isdir(sub_dir) and "prompt.json" in os.listdir(sub_dir):
            return "step"
    for item in os.listdir(target_folder):
        sub_dir = os.path.join(target_folder, item)
        if os.path.isdir(sub_dir):
            for sub_item in os.listdir(sub_dir):
                sub_sub_dir = os.path.join(sub_dir, sub_item)
                if os.path.isdir(sub_sub_dir) and "prompt.json" in os.listdir(sub_sub_dir):
                    return "experiment"
    return "invalid"

def run_smart(target_folder, openai_key, repeat_num=5, save_to_exp_dir=True, selected_step=None, selected_case=None):
    hierarchy = check_dir_hierarchy(target_folder)
    assert hierarchy != "invalid", "Invalid folder structure"
    if hierarchy == "case":
        return run_case_experiment(target_folder, openai_key, repeat_num, save_to_exp_dir)
    if hierarchy == "step":
        return run_step_experiment(target_folder, openai_key, repeat_num, save_to_exp_dir, selected_case)
    if hierarchy == "experiment":
        return run_experiment(target_folder, openai_key, repeat_num, save_to_exp_dir, selected_step, selected_case)    


##########################################################################################################################
####################################################### Main part ########################################################
##########################################################################################################################

def parse_input_dict(input_dict):
    system_msg = input_dict["system_msg"]
    inputs = input_dict["inputs"]
    return system_msg, inputs


def ask_general_sequential_llm(openai_key, system_msg, questions):
    conversation_record = [] 
    gpt_history = []
    for query, image in questions:
        conversation_record.append((query, image))
        if not isinstance(image, list) and image is not None:
            image = [image]
        #pdb.set_trace()
        print(f"query {query}, image {image}")
        gpt_history, response_content, response = ask_gpt(query, image, gpt_history, system_msg, openai_key, IMAGE_FOLDER)
        conversation_record.append(response_content)
    return conversation_record

def run_case_experiment_sequential(case_directory, openai_key, repeat_num=5, save_to_case_dir=True):
    query_json = os.path.join(case_directory, "query.json")
    if not os.path.exists(query_json):
        print(f"sequential.json not found in {case_directory}")
        return None
    with open(query_json, "r") as f:
        seq_info = json.load(f)
    if seq_info is None:
        return None
    local_dict = {}
    system_msg = seq_info["system_msg"]
    query = seq_info["query"]
    local_dict["system_msg"] = system_msg
    local_dict["query"] = query
    local_dict["output"] = {}
    for i in range(repeat_num):
        print(f"Running repeat {i}")
        response = ask_general_sequential_llm(openai_key, system_msg, query, )
        local_dict["output"][f"repeat_{i}"] = response
    
    if save_to_case_dir:
        case_output_dir = os.path.join(case_directory, "output")
        os.makedirs(case_output_dir, exist_ok=True)
        case_output_file = os.path.join(case_output_dir, f"response_c_{CURRENT_TIME}.json")
        with open(case_output_file, "w") as f:
            json.dump(local_dict, f, indent=4)
    return local_dict
     

if __name__ == "__main__":
    #global IMAGE_FOLDER
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("target_folder", type=str, help="Path to the target folder")
    parser.add_argument("--legacy", action="store_true", help="Use legacy mode")
    parser.add_argument("-r", "--repeat_num", type=int, default=2, help="Number of times to repeat the experiment")
    # list of selected step or case to run, multiple values can be passed, default is None
    parser.add_argument("-s", "--selected_step", type=str, nargs="+", default=None, help="List of selected steps")
    parser.add_argument("-c", "--selected_case", type=str, nargs="+", default=None, help="List of selected cases")
    parser.add_argument("-ct","--call_type", type=str, default="smart", help="Type of call to make, smart, case, step, experiment")
    args = parser.parse_args()
    openai_key = get_api_key()
    IMAGE_FOLDER = args.target_folder
    
    if args.legacy:
        if args.call_type == "smart":
            run_smart(args.target_folder, openai_key, args.repeat_num, True, args.selected_step, args.selected_case)
        elif args.call_type == "case":
            run_case_experiment(args.target_folder, openai_key, args.repeat_num, True)
        elif args.call_type == "step":
            run_step_experiment(args.target_folder, openai_key, args.repeat_num, True, args.selected_case)
        elif args.call_type == "experiment":
            run_experiment(args.target_folder, openai_key, args.repeat_num, True, args.selected_step, args.selected_case)
        else:
            print("Invalid call type, please select from smart, case, step, experiment")
    else:
        run_case_experiment_sequential(args.target_folder, openai_key, args.repeat_num, True)
    
