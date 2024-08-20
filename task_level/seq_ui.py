import streamlit as st
import json
from PIL import Image
import os
import pandas as pd 
from copy import deepcopy

# manually set the directory
EXP_DIR = "/home/ycyao/Documents/gpt_ask/experiments"
             

# Set page config to make layout wide
st.set_page_config(layout="wide")

# Load the JSON data
def load_data(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
        case_name = filepath.split('/')[-3]
        json_name = filepath.split('/')[-1]
        data['case_name'] = case_name
        data['json_name'] = json_name
    return data

# Display image if exists
def display_image(image_path):
    if os.path.exists(image_path):
        image = Image.open(image_path)
        st.image(image, width=300)  # Adjust width as needed

def find_all_json_files(directory):
    out_json = []
    for case_dir in os.listdir(directory):
        full_dir_path = os.path.join(directory, case_dir)
        if "query.json" not in os.listdir(full_dir_path):
            continue
        if os.path.isdir(full_dir_path):
            output_dir = os.path.join(full_dir_path, 'output')
            output_paths = os.listdir(output_dir)
            if len(output_paths) == 0:
                continue
            sorted_out_path = sorted(output_paths)
            output_json = os.path.join(output_dir, sorted_out_path[-1])
            out_json.append(output_json)
    return out_json



def save_data(filepath, data):
    with open(filepath, 'w') as file:
        json.dump(data, file, indent=4)

def view_conversations(json_files):
    # Navigation through files
    if 'current_file_index' not in st.session_state:
        st.session_state['current_file_index'] = 0  # Initialize state
    col_num = 0

    data = load_data(json_files[st.session_state['current_file_index']])
    col1, col2 = st.columns([2, 2])
    if col1.button('Previous'):
        if st.session_state['current_file_index'] > 0:
            st.session_state['current_file_index'] -= 1
            st.rerun()
            col_num = 0

    if col2.button('Next'):
        if st.session_state['current_file_index'] < len(json_files) - 1:
            st.session_state['current_file_index'] += 1
            st.rerun()
            col_num = 0

    # Display conversations
    if 'output' in data:
        conv_length = len(data["output"]["repeat_0"])
        st.subheader(f'{data["case_name"]} - {data["json_name"]}')
        for i in range(conv_length):
            cols = st.columns(len(data['output'])-1)
            for j in range(len(data['output'])-1):
                data_part = data['output'][f'repeat_{j}']
                with cols[j]:
                    conversation = data_part[i]
                    if isinstance(conversation, list):
                        st.write('Q:', conversation[0])
                        if conversation[1]:
                            st.image(conversation[1][0], width=300)
                    else:
                        st.write('A:', conversation)
            st.divider()
    # if 'output' in data:
    #     cols = st.columns(len(data['output'])-1)
    #     for key, value in data['output'].items():
    #         if key != 'record_points':
    #             with cols[col_num]:
    #                 st.subheader(f'{data["case_name"]} - {data["json_name"]}')
    #                 for item in value:
    #                     if isinstance(item, list):
    #                         st.write('Q:', item[0])
    #                         if item[1]:
    #                             st.image(item[1][0], width=300)
    #                     else:
    #                         st.write('A:', item)
    #             col_num += 1

    if 'record_points' in data["output"]:
        st.header('Recorded Numbers')
        new_numbers = {}
        for key, value in data["output"]['record_points'].items():
            new_value = st.text_input(f"{key}", value)
            new_numbers[key] = new_value

        # Save the updated recorded numbers back to the JSON file if they are changed
        if st.button('Save Changes'):
            data["output"]['record_points'] = new_numbers
            save_data(json_files[st.session_state['current_file_index']], data)
            st.success('Changes saved successfully!')

def summary_page(json_files):
    st.title("Summary of Recorded Numbers")
    # Prepare an empty dictionary to gather data in the form of a dataframe
    summary_data = {}
    column_labels = []  # This will store column names to ensure all rows have the same structure

    # Gather all recorded_numbers into the summary dictionary
    for file_path in json_files:
        data = load_data(file_path)
        recorded_numbers = deepcopy(data["output"]['record_points'])
        conversation_nums = len(data['output']) - 1
        for k,v in recorded_numbers.items():
            recorded_numbers[k] = str(v) + " / " + str(conversation_nums)
        summary_data[file_path] = recorded_numbers
        # Update the column labels with any new keys found in this file's recorded_numbers
        for key in recorded_numbers.keys():
            if key not in column_labels:
                column_labels.append(key)

    # Convert dictionary to DataFrame ensuring all columns are considered
    df = pd.DataFrame.from_dict(summary_data, orient='index', columns=column_labels)

    # Handle missing values if any (replace with a default value or drop etc.)
    df.fillna('Missing', inplace=True)  # Or use any other appropriate default value

    st.table(df)


def main():
    tab1, tab2 = st.tabs(["View Conversations", "Summary"])
    json_files = json_files = find_all_json_files(EXP_DIR)

    with tab1:
        view_conversations(json_files)
    with tab2:
        summary_page(json_files)


if __name__ == '__main__':
    main()

