import pandas as pd
import os
import json
import numpy as np

class npEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int32):
            return int(obj)
        if isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class FinetuningRecorder():
    
    def __init__(self, log_path) -> None:
        # to record all information in one place
        #self.df = pd.DataFrame({"round":[],"step":[],"player":[],"selected_option":[],"LLM_response_plan":[],"LLM_response_action":[],"action_code":[],"plan_position":[],"plan":[],"remaining_steps_plan":[],"reward":[],"cumulative_reward":[],"pos_prev":[],"pos_new":[]})
        self.df = pd.DataFrame()
        self.dict_temp = {}
        self.list_save = []
        self.log_path = log_path

    def add_value(self, key, value):
        self.dict_temp[key]=value
    
    def save_step(self):
        print("SAVING STEP",self.dict_temp)
        self.df._append(self.dict_temp.copy(), ignore_index=True)
        self.list_save.append(self.dict_temp)
        self.dict_temp = {}
        with open(os.path.join(self.log_path, "finetune_track.txt"), "w") as f:
            f.write(json.dumps(self.list_save, indent=4, cls=npEncoder))

    def persist_recording(self):
        self.save_step()
        self.df.to_csv(os.path.join(self.log_path, "fintune_track.csv"))
        self.df.to_excel(os.path.join(self.log_path, "fintune_track.xlsx"))
        with open(os.path.join(self.log_path, "finetune_track.txt"), "w") as f:
            f.write(json.dumps(self.list_save, indent=4, cls=npEncoder))

