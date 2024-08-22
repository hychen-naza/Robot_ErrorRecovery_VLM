prompt_configs = dict(
    original_prompt_continuous = dict(
        horizontal_prompt="""
        Given the current image, how should the robot arm gripper move the blut cube it is holding to reach the target location marked by the blue box? Please be concise and provide a horizontal and vertial movement amount in meters. Assume moving rightward and downward are negative and moving leftward and upward are positive. You should provide the movement in the following format:
            - (horizontal movement amount, vertical movement amount)
        Always wrap your final answer with parentheses as shown in the options.
        """,
        decay_ratio = 1.0,
        directly_apply_simple = False,
        vertical_prompt = None,
        horizontal_summary_prompt=None,
        vertical_summary_prompt=None,
        hor_sys_msg = "You are an assistant that helps a human determine the horizontal movements on a tabletop.",
        ver_sys_msg = "You are an assistant that helps a human determine the vertical movements on a tabletop.",
        # stop_sys_msg = "You are an assistant that helps a human determine the stop condition of a movement.",
        # stop_query_msg = """
        # Given the current image, tell me if the blue object reach the red marker?
        # """,
        # stop_sum_msg = """
        # Please summarize your previous description of whether the blue cube reach the red marker into the one of the following two options.
        #     - (yes)
        #     - (no)
        # Always wrap your final answer with parentheses as shown in the options.
        # """
    ),
    prompt_relative = dict(
        visual_opt_type="marker",
        combined_relative=True,
        directly_apply_simple=True,
        horizontal_prompt="""
        Given the current image, in which direction should the robot arm gripper move the blue cube it is holding horizontally (left or right) and vertically (upward or downward)  to reach the target location marked by the green box? You should provide the horizontal and vertical movement in the following format:
            - (horizontal movement, vertical movement)
            
        Please be concise and limit your horizontal movement to the following:
            - (left)
            - (right)
        and limit your vertical movement to the following:
            - (upward)
            - (downward)
        """,
        vertical_prompt = None,
        horizontal_summary_prompt=None,
        vertical_summary_prompt = None,
        hor_sys_msg = "You are an assistant that helps a human determine the horizontal movements on a tabletop.",
        ver_sys_msg = "You are an assistant that helps a human determine the vertical movements on a tabletop.",
        # stop_sys_msg = "You are an assistant that helps a human determine the stop condition of a movement.",
        # stop_query_msg = """
        # Given the current image, tell me if the blue object reach the red marker?
        # """,
        # stop_sum_msg = """
        # Please summarize your previous description of whether the blue cube reach the red marker into the one of the following two options.
        #     - (yes)
        #     - (no)
        # Always wrap your final answer with parentheses as shown in the options.
        # """
    ),
    prompt_relative_v2 = dict(
        visual_opt_type="marker",
        combined_relative=True,
        directly_apply_simple=True,
        horizontal_prompt="""
        Given the current image, what's the horizontal and vertical relative position of the blue cube to the green box and in which direction should the robot arm gripper move the blue cube it is holding horizontally (left or right) and vertically (upward or downward)  to reach the target location marked by the green box? You should provide the horizontal and vertical movement in the following format:
            - (horizontal movement, vertical movement)
            
        Please be concise and limit your horizontal movement to the following:
            - (left)
            - (right)
        and limit your vertical movement to the following:
            - (upward)
            - (downward)
        """,
        vertical_prompt = None,
        horizontal_summary_prompt=None,
        vertical_summary_prompt = None,
        hor_sys_msg = "You are an assistant that helps a human determine the horizontal movements on a tabletop.",
        ver_sys_msg = "You are an assistant that helps a human determine the vertical movements on a tabletop.",
        # stop_sys_msg = "You are an assistant that helps a human determine the stop condition of a movement.",
        # stop_query_msg = """
        # Given the current image, tell me if the blue object reach the red marker?
        # """,
        # stop_sum_msg = """
        # Please summarize your previous description of whether the blue cube reach the red marker into the one of the following two options.
        #     - (yes)
        #     - (no)
        # Always wrap your final answer with parentheses as shown in the options.
        # """
    ),
    prompt_relative_decompose_c1_v2= dict(
        visual_opt_type="marker",
        horizontal_prompt="""
        Given the current image, what's the horizontal relative position of the blue cube to the green box and in which direction should the robot arm gripper move the blue cube it is holding horizontally (left or right) to reach the target location marked by the green box? Please be concise and limit your answer to the following:
            - (left)
            - (right)
        Always wrap your final answer with parentheses as shown in the options.
        """,
        vertical_prompt = """    
        Given the current image, what's the vertical relative position of the blue cube to the green box and in which direction should the robot arm gripper move the blue cube it is holding vertically (upward or downward) to reach the target location marked by the green box? Please be concise and limit your answer to the following:
            - (upward)
            - (downward)
        Always wrap your final answer with parentheses as shown in the options.
        """,
        directly_apply_simple=False,
        horizontal_summary_prompt=None,
        vertical_summary_prompt = None,
        hor_sys_msg = "You are an assistant that helps a human determine the horizontal movements on a tabletop.",
        ver_sys_msg = "You are an assistant that helps a human determine the vertical movements on a tabletop.",
        # stop_sys_msg = "You are an assistant that helps a human determine the stop condition of a movement.",
        # stop_query_msg = """
        # Given the current image, tell me if the blue object reach the red marker?
        # """,
        # stop_sum_msg = """
        # Please summarize your previous description of whether the blue cube reach the red marker into the one of the following two options.
        #     - (yes)
        #     - (no)
        # Always wrap your final answer with parentheses as shown in the options.
        # """
    ),
    prompt_relative_decompose_c2_v2= dict(
        visual_opt_type="marker",
        horizontal_prompt="""
        Given the current image, what's the horizontal relative position of the blue cube to the green box and in which direction should the robot arm gripper move the blue cube it is holding horizontally (left or right) to reach the target location marked by the green box?
        """,
        vertical_prompt = """    
        Given the current image, what's the vertical relative position of the blue cube to the green box and in which direction should the robot arm gripper move the blue cube it is holding vertically (upward or downward) to reach the target location marked by the green box?
        """,
        directly_apply_simple=False,
        horizontal_summary_prompt=""" 
        Please summarize your previous description of the direction the blue cube should move to reach the green box horizontally into the one of the following two options:
            - (left)
            - (right)
        Always wrap your final answer with parentheses as shown in the options.
        """,
        vertical_summary_prompt = """ 
        Please summarize your previous description of the direction the blue cube should move to reach the green box vertically into the one of the following two options:
            - (upward)
            - (downward)
        Always wrap your final answer with parentheses as shown in the options.
        """,
        hor_sys_msg = "You are an assistant that helps a human determine the horizontal movements on a tabletop.",
        ver_sys_msg = "You are an assistant that helps a human determine the vertical movements on a tabletop.",
        # stop_sys_msg = "You are an assistant that helps a human determine the stop condition of a movement.",
        # stop_query_msg = """
        # Given the current image, tell me if the blue object reach the red marker?
        # """,
        # stop_sum_msg = """
        # Please summarize your previous description of whether the blue cube reach the red marker into the one of the following two options.
        #     - (yes)
        #     - (no)
        # Always wrap your final answer with parentheses as shown in the options.
        # """
    ),
    prompt_relative_decompose_c2_visfeature_v2= dict(
        visual_opt_type="both",
        horizontal_prompt="""
        Given the current image, what's the horizontal relative position of the blue cube to the green box and in which direction should the robot arm gripper move the blue cube it is holding horizontally (left or right) to reach the target red square marker inside the green box?
        """,
        vertical_prompt = """    
        Given the current image, what's the vertical relative position of the blue cube to the green box and in which direction should the robot arm gripper move the blue cube it is holding vertically (upward or downward) to reach the target red square marker inside the green box?
        """,
        directly_apply_simple=False,
        horizontal_summary_prompt=""" 
        Please summarize your previous description of the direction the blue cube should move to reach the red marker horizontally into the one of the following two options:
            - (left)
            - (right)
        Always wrap your final answer with parentheses as shown in the options.
        """,
        vertical_summary_prompt = """ 
        Please summarize your previous description of the direction the blue cube should move to reach the red marker vertically into the one of the following two options:
            - (upward)
            - (downward)
        Always wrap your final answer with parentheses as shown in the options.
        """,
        hor_sys_msg = "You are an assistant that helps a human determine the horizontal movements on a tabletop.",
        ver_sys_msg = "You are an assistant that helps a human determine the vertical movements on a tabletop.",
        # stop_sys_msg = "You are an assistant that helps a human determine the stop condition of a movement.",
        # stop_query_msg = """
        # Given the current image, tell me if the blue object reach the red marker?
        # """,
        # stop_sum_msg = """
        # Please summarize your previous description of whether the blue cube reach the red marker into the one of the following two options.
        #     - (yes)
        #     - (no)
        # Always wrap your final answer with parentheses as shown in the options.
        # """
    ),
    prompt_relative_decompose_c1= dict(
        visual_opt_type="marker",
        horizontal_prompt="""
        Given the current image, in which direction should the robot arm gripper move the blue cube it is holding horizontally (left or right) to reach the target location marked by the green box? Please be concise and limit your answer to the following:
            - (left)
            - (right)
        Always wrap your final answer with parentheses as shown in the options.
        """,
        vertical_prompt = """    
        Given the current image, in which direction should the robot arm gripper move the blue cube it is holding vertically (upward or downward) to reach the target location marked by the green box? Please be concise and limit your answer to the following:
            - (upward)
            - (downward)
        Always wrap your final answer with parentheses as shown in the options.
        """,
        directly_apply_simple=False,
        horizontal_summary_prompt=None,
        vertical_summary_prompt = None,
        hor_sys_msg = "You are an assistant that helps a human determine the horizontal movements on a tabletop.",
        ver_sys_msg = "You are an assistant that helps a human determine the vertical movements on a tabletop.",
        # stop_sys_msg = "You are an assistant that helps a human determine the stop condition of a movement.",
        # stop_query_msg = """
        # Given the current image, tell me if the blue object reach the red marker?
        # """,
        # stop_sum_msg = """
        # Please summarize your previous description of whether the blue cube reach the red marker into the one of the following two options.
        #     - (yes)
        #     - (no)
        # Always wrap your final answer with parentheses as shown in the options.
        # """
    ),
    prompt_relative_decompose_c2= dict(
        visual_opt_type="marker",
        horizontal_prompt="""
        Given the current image, in which direction should the robot arm gripper move the blue cube it is holding horizontally (left or right) to reach the target location marked by the green box?
        """,
        vertical_prompt = """    
        Given the current image, in which direction should the robot arm gripper move the blue cube it is holding vertically (upward or downward) to reach the target location marked by the green box?
        """,
        directly_apply_simple=False,
        horizontal_summary_prompt=""" 
        Please summarize your previous description of the direction the blue cube should move to reach the red marker horizontally into the one of the following two options:
            - (left)
            - (right)
        Always wrap your final answer with parentheses as shown in the options.
        """,
        vertical_summary_prompt = """ 
        Please summarize your previous description of the direction the blue cube should move to reach the red marker vertically into the one of the following two options:
            - (upward)
            - (downward)
        Always wrap your final answer with parentheses as shown in the options.
        """,
        hor_sys_msg = "You are an assistant that helps a human determine the horizontal movements on a tabletop.",
        ver_sys_msg = "You are an assistant that helps a human determine the vertical movements on a tabletop.",
        # stop_sys_msg = "You are an assistant that helps a human determine the stop condition of a movement.",
        # stop_query_msg = """
        # Given the current image, tell me if the blue object reach the red marker?
        # """,
        # stop_sum_msg = """
        # Please summarize your previous description of whether the blue cube reach the red marker into the one of the following two options.
        #     - (yes)
        #     - (no)
        # Always wrap your final answer with parentheses as shown in the options.
        # """
    ),
    
    
    relative_grasp_prompt = dict(
        horizontal_prompt="""
        Given the current image, what's the horizontal relative position (left or right) of the blue detected object to the red marker on the gripper and in which direction the red marker should move horizontally (left or right) to get closer to the blue detected object?
        """,
        vertical_prompt = None,
        # to make the red marker reach the blue marker 
        horizontal_summary_prompt=""" 
        Please summarize your previous description of the direction the robot arm gripper, which is the red marker is attached to, should rotate to get closer to the blue bounding box into the one of the following two options:
            - (left)
            - (right)
        Always wrap your final answer with parentheses as shown in the options.
        """,
        vertical_summary_prompt = None,
        hor_sys_msg = "You are an assistant that helps a human determine the horizontal movements of a robot arm.",
        ver_sys_msg = None,
        # stop_sys_msg = "You are an assistant that helps a human determine the stop condition of a movement.",
        # stop_query_msg = """
        # Given the current image, tell me if the blue object reach the red marker?
        # """,
        # stop_sum_msg = """
        # Please summarize your previous description of whether the blue cube reach the red marker into the one of the following two options.
        #     - (yes)
        #     - (no)
        # Always wrap your final answer with parentheses as shown in the options.
        # """
    ),
    combine_grasp_prompt = dict(
        horizontal_prompt="""
        Given the current image, what's the horizontal relative position (left or right) of the blue bounding box to the red square and in which direction the red square marker should move horizontally (left for forward or right for backward) to reach the blue bounding box?
        """,
        vertical_prompt = """    
        Given the current image, what's the vertical relative position (upward or downward) of the blue bounding box to the red square and in which direction the red square marker should move vertically (upward or downward) to reach the blue bounding box?
        """,
        horizontal_summary_prompt=""" 
        Please summarize your previous description of the direction the robot arm gripper, which is the red marker is attached to, should move to reach the blue bounding box into the one of the following two options:
            - (forward)
            - (backward)
        Always wrap your final answer with parentheses as shown in the options.
        """,
        vertical_summary_prompt = """ 
        Please summarize your previous description of the direction the robot arm gripper, which is the red marker is attached to, should move to reach the blue bounding box vertically into the one of the following two options:
            - (upward)
            - (downward)
        Always wrap your final answer with parentheses as shown in the options.
        """,
        hor_sys_msg = "You are an assistant that helps a human determine the horizontal movements of a robot arm.",
        ver_sys_msg = "You are an assistant that helps a human determine the vertical movements of a robot arm.",
        stop_sys_msg = "You are an assistant that helps a human determine the stop condition of a movement.",
        stop_query_msg = """
        Given the current image, tell me if the blue object reach the red marker?
        """,
        stop_sum_msg = """
        Please summarize your previous description of whether the blue cube reach the red marker into the one of the following two options.
            - (yes)
            - (no)
        Always wrap your final answer with parentheses as shown in the options.
        """
    ),
    combine_3d_grasp_prompt = dict(
        horizontal_prompt="""
        Given the current image, what's the horizontal relative position (left or right) of the blue bounding box to the red square and in which direction the red square marker should move horizontally (left for forward or right for backward) to reach the blue bounding box?
        """,
        vertical_prompt = """    
        Given the current image, what's the vertical relative position (upward or downward) of the blue bounding box to the red square and in which direction the red square marker should move vertically (upward or downward) to reach the blue bounding box?
        """,
        top_horizontal_prompt = """    
        Given the current image, what's the horizontal relative position (left or right) of the blue bounding box to the red marker and in which direction the red marker should move horizontally (left or right) to reach the blue bounding box?
        """,
        horizontal_summary_prompt=""" 
        Please summarize your previous description of the direction the robot arm gripper, which is the red marker is attached to, should move to reach the blue bounding box into the one of the following two options:
            - (forward)
            - (backward)
        Always wrap your final answer with parentheses as shown in the options.
        """,
        vertical_summary_prompt = """ 
        Please summarize your previous description of the direction the robot arm gripper, which is the red marker is attached to, should move to reach the blue bounding box vertically into the one of the following two options:
            - (upward)
            - (downward)
        Always wrap your final answer with parentheses as shown in the options.
        """,
        top_horizontal_summary_prompt=""" 
        Please summarize your previous description of the direction the robot arm gripper, which is the red marker is attached to, should move to reach the blue bounding box into the one of the following two options:
            - (left)
            - (right)
        Always wrap your final answer with parentheses as shown in the options.
        """,
        hor_sys_msg = "You are an assistant that helps a human determine the horizontal movements of a robot arm.",
        ver_sys_msg = "You are an assistant that helps a human determine the vertical movements of a robot arm.",
        top_hor_sys_msg = "You are an assistant that helps a human determine the horizontal movements of a robot arm.",
        stop_sys_msg = "You are an assistant that helps a human determine the stop condition of a movement.",
        stop_query_msg = """
        Given the current image, tell me if the blue object reach the red marker?
        """,
        stop_sum_msg = """
        Please summarize your previous description of whether the blue cube reach the red marker into the one of the following two options.
            - (yes)
            - (no)
        Always wrap your final answer with parentheses as shown in the options.
        """
    ),
)
