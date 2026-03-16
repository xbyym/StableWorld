
import torch
import random
import math

def combine_data(data, num_frames=57, keyboard_dim=6, mouse=True):
    assert num_frames % 4 == 1
    keyboard_condition = torch.zeros((num_frames, keyboard_dim))
    if mouse == True:
        mouse_condition = torch.zeros((num_frames, 2))
    
    current_frame = 0
    selections = [12]

    while current_frame < num_frames:
        rd_frame = selections[random.randint(0, len(selections) - 1)]
        rd = random.randint(0, len(data) - 1)
        k = data[rd]['keyboard_condition']
        if mouse == True:
            m = data[rd]['mouse_condition']
        
        if current_frame == 0:
            keyboard_condition[:1] = k[:1]
            if mouse == True:
                mouse_condition[:1] = m[:1]
            current_frame = 1
        else:
            rd_frame = min(rd_frame, num_frames - current_frame)
            repeat_time = rd_frame // 4
            keyboard_condition[current_frame:current_frame+rd_frame] = k.repeat(repeat_time, 1)
            if mouse == True:
                mouse_condition[current_frame:current_frame+rd_frame] = m.repeat(repeat_time, 1)
            current_frame += rd_frame
    if mouse == True:
        return {
                "keyboard_condition": keyboard_condition,
                "mouse_condition": mouse_condition
            }
    return {"keyboard_condition": keyboard_condition}



def combine_data_in_order(
    data,
    num_frames=57,
    keyboard_dim=4,          # forward/back/left/right
    mouse=True,
    frames_per_action=12,    # number of internal frames occupied by each action
    start_with_first_frame=True,
    idle_token="idle",       # use this string to represent an idle state
    template_len_default=4   # default template length for idle entries
):
    """
    data: a list whose elements can be in one of two forms:

      1) dict:
         {
           "keyboard_condition": Tensor[template_len, keyboard_dim],
           "mouse_condition":    Tensor[template_len, 2]   # when mouse=True
         }

      2) a string equal to idle_token (default: 'idle'):
         represents an idle state (all-zero keyboard and mouse input),
         using template_len_default as the template length
    """
    assert isinstance(data, list) and len(data) > 0, "data must not be empty"

    keyboard_condition = torch.zeros((num_frames, keyboard_dim))
    mouse_condition = torch.zeros((num_frames, 2)) if mouse else None

    def _get_templates(item):
        """Generate templates (k, m, template_len) from one element in data."""
        if isinstance(item, str) and item == idle_token:
            k = torch.zeros((template_len_default, keyboard_dim))
            m = torch.zeros((template_len_default, 2)) if mouse else None
            tmpl_len = template_len_default
        else:
            k = item["keyboard_condition"]                     # [template_len, keyboard_dim]
            tmpl_len = k.shape[0]
            assert tmpl_len > 0 and k.shape[1] == keyboard_dim
            if mouse:
                m = item["mouse_condition"]                    # [template_len, 2]
                assert m.shape[0] == tmpl_len and m.shape[1] == 2
            else:
                m = None
        return k, m, tmpl_len

    cur = 0
    idx = 0  # points to the current action

    # Write the starting frame first to match the previous implementation.
    # If the first entry is idle, an all-zero frame will also be written.
    if start_with_first_frame and cur < num_frames:
        k0, m0, _ = _get_templates(data[0])
        keyboard_condition[cur:cur+1] = k0[:1]
        if mouse:
            mouse_condition[cur:cur+1] = m0[:1] if m0 is not None else 0.0
        cur += 1
        idx = 1 % len(data)

    while cur < num_frames:
        k, m, tmpl_len = _get_templates(data[idx])

        to_fill = min(frames_per_action, num_frames - cur)

        # Repeat the template and truncate it
        # so that it fills exactly `to_fill` frames
        rep = math.ceil(to_fill / tmpl_len)
        k_block = k.repeat(rep, 1)[:to_fill]
        keyboard_condition[cur:cur+to_fill] = k_block

        if mouse:
            if m is None:
                # idle case
                mouse_condition[cur:cur+to_fill] = 0.0
            else:
                m_block = m.repeat(rep, 1)[:to_fill]
                mouse_condition[cur:cur+to_fill] = m_block

        cur += to_fill
        idx = (idx + 1) % len(data)

    if mouse:
        return {
            "keyboard_condition": keyboard_condition,
            "mouse_condition": mouse_condition
        }
    else:
        return {"keyboard_condition": keyboard_condition}

def Bench_actions_universal(num_frames, num_samples_per_action=4):
    actions_single_action = [
        "forward",
        # "back",
        "left",
        "right",
    ]
    actions_double_action = [
        "forward_left",
        "forward_right",
        # "back_left",
        # "back_right",
    ]

    actions_single_camera = [   
        "camera_l",
        "camera_r",
        # "camera_ur",
        # "camera_ul",
        # "camera_dl",
        # "camera_dr" 
        # "camera_up",
        # "camera_down",
    ]
    actions_to_test = actions_double_action * 5 + actions_single_camera * 5 + actions_single_action * 5
    for action in (actions_single_action + actions_double_action):
        for camera in (actions_single_camera):
            double_action = f"{action}_{camera}"
            actions_to_test.append(double_action)

    # print("length of actions: ", len(actions_to_test))
    base_action = actions_single_action + actions_single_camera

    KEYBOARD_IDX = { 
        "forward": 0, "back": 1, "left": 2, "right": 3
    }

    CAM_VALUE = 0.1
    CAMERA_VALUE_MAP = {
        "camera_up":  [CAM_VALUE, 0],
        "camera_down": [-CAM_VALUE, 0],
        "camera_l":   [0, -CAM_VALUE],
        "camera_r":   [0, CAM_VALUE],
        "camera_ur":  [CAM_VALUE, CAM_VALUE],
        "camera_ul":  [CAM_VALUE, -CAM_VALUE],
        "camera_dr":  [-CAM_VALUE, CAM_VALUE],
        "camera_dl":  [-CAM_VALUE, -CAM_VALUE],
    }

    data = []

    for action_name in actions_to_test:

        keyboard_condition = [[0, 0, 0, 0] for _ in range(num_samples_per_action)] 
        mouse_condition = [[0,0] for _ in range(num_samples_per_action)] 

        for sub_act in base_action:
            if not sub_act in action_name: # 只处理action_name包含的动作
                continue
            # print(f"action name: {action_name} sub_act: {sub_act}")
            if sub_act in CAMERA_VALUE_MAP:
                mouse_condition = [CAMERA_VALUE_MAP[sub_act]
                                   for _ in range(num_samples_per_action)]

            elif sub_act in KEYBOARD_IDX:
                col = KEYBOARD_IDX[sub_act]
                for row in keyboard_condition:
                    row[col] = 1

        data.append({
            "keyboard_condition": torch.tensor(keyboard_condition),
            "mouse_condition": torch.tensor(mouse_condition)
        })
    # return combine_data(data, num_frames, keyboard_dim=4, mouse=True)

    return combine_data_in_order(data, num_frames, keyboard_dim=4, mouse=True)

def Bench_actions_static(num_frames):
    keyboard_condition = torch.zeros((num_frames, 4))
    mouse_condition    = torch.zeros((num_frames, 2))
    return {"keyboard_condition": keyboard_condition,
            "mouse_condition":   mouse_condition}

def Bench_actions_gta_drive(num_frames, num_samples_per_action=4):
    actions_single_action = [
        "forward",
        "back",
    ]

    actions_single_camera = [   
        "camera_l",
        "camera_r",
    ]
    actions_to_test = actions_single_camera * 2 + actions_single_action * 2
    for action in (actions_single_action):
        for camera in (actions_single_camera):
            double_action = f"{action}_{camera}"
            actions_to_test.append(double_action)

    # print("length of actions: ", len(actions_to_test))
    base_action = actions_single_action + actions_single_camera

    KEYBOARD_IDX = { 
        "forward": 0, "back": 1
    }

    CAM_VALUE = 0.1
    CAMERA_VALUE_MAP = {
        "camera_l":   [0, -CAM_VALUE],
        "camera_r":   [0, CAM_VALUE],
    }
    
    data = []

    for action_name in actions_to_test:

        keyboard_condition = [[0, 0] for _ in range(num_samples_per_action)] 
        mouse_condition = [[0,0] for _ in range(num_samples_per_action)] 

        for sub_act in base_action:
            if not sub_act in action_name: # 只处理action_name包含的动作
                continue
            # print(f"action name: {action_name} sub_act: {sub_act}")
            if sub_act in CAMERA_VALUE_MAP:
                mouse_condition = [CAMERA_VALUE_MAP[sub_act]
                                   for _ in range(num_samples_per_action)]

            elif sub_act in KEYBOARD_IDX:
                col = KEYBOARD_IDX[sub_act]
                for row in keyboard_condition:
                    row[col] = 1

        data.append({
            "keyboard_condition": torch.tensor(keyboard_condition),
            "mouse_condition": torch.tensor(mouse_condition)
        })
    return combine_data(data, num_frames, keyboard_dim=2, mouse=True)

def Bench_actions_templerun(num_frames, num_samples_per_action=4):
    actions_single_action = [
        "jump",
        "slide",
        "leftside",
        "rightside",
        "turnleft",
        "turnright",
        "nomove"
    ]

    actions_to_test = actions_single_action

    base_action = actions_single_action

    KEYBOARD_IDX = { 
        "nomove": 0, "jump": 1, "slide": 2, "turnleft": 3,
        "turnright": 4, "leftside": 5, "rightside": 6
    }

    data = []

    for action_name in actions_to_test:

        keyboard_condition = [[0, 0, 0, 0, 0, 0, 0] for _ in range(num_samples_per_action)] 

        for sub_act in base_action:
            if not sub_act in action_name: # 只处理action_name包含的动作
                continue
            # print(f"action name: {action_name} sub_act: {sub_act}")
            elif sub_act in KEYBOARD_IDX:
                col = KEYBOARD_IDX[sub_act]
                for row in keyboard_condition:
                    row[col] = 1

        data.append({
            "keyboard_condition": torch.tensor(keyboard_condition)
        })
    return combine_data(data, num_frames, keyboard_dim=7, mouse=False)