import os
from typing import Mapping
import cv2
from datetime import datetime
import shutil
import numpy as np
from skimage import io
from skimage.transform import resize
from game_environment.recorder import recreate_simulation
from game_environment.utils import parse_string_to_matrix, matrix_to_string, connected_elems_map


class Recorder:

    def __init__(self, log_path, init_timestamp, substrate_config, substrate_name):
        self.substrate_name = substrate_name
        self.substrate_config = substrate_config
        self.n_players = self.substrate_config.lab2d_settings.numPlayers
        self.experiment_id = init_timestamp
        self.log_path = os.path.join(log_path, str(self.experiment_id))
        self.step = 0
        self.logs = {}
        self.create_log_tree()

    def create_log_tree(self):
        self.create_or_replace_directory(self.log_path)
        self.create_or_replace_directory(os.path.join(self.log_path, "world"))
        for player_id in range(self.n_players):
            self.create_or_replace_directory(os.path.join(self.log_path, str(player_id)))

    def record(self, timestep, description):
        world_view = timestep.observation["WORLD.RGB"]
        world_path = os.path.join(self.log_path, "world", f"{self.step}.png")
        self.save_image(world_view, world_path)

        for player_id in range(self.n_players):
            agent_observation = timestep.observation[f"{player_id + 1}.RGB"]
            description_image = self.add_description(description[player_id])
            agent_observation = resize(agent_observation, (description_image.shape[0], description_image.shape[1]), anti_aliasing=True)
            agent_observation = (agent_observation * 255).astype(np.uint8)
            agent_observation = np.hstack([agent_observation, description_image])
            avatar_path = os.path.join(self.log_path, str(player_id), f"{self.step}.png")
            self.save_image(agent_observation, avatar_path)

        self.step += 1

    def record_rewards(self, rewards: Mapping[str, float])->None:
        #Writes the rewards to a file
        with open(os.path.join(self.log_path, "rewards_history.txt"), "a") as f:
            rewards = {i: int(rr) for i, rr in enumerate(list(rewards.values()))}
            f.write(f"{self.step}: {rewards}\n")

    def record_elements_status(self, initial_map, current_map):
        # Transform list map to string map
        if self.substrate_name == "commons_harvest_open":
            connected_elements = connected_elems_map(initial_map, ['A'])

            trees = {}

            for elem_key in connected_elements:
                elements = connected_elements[elem_key]['elements']
                if len(elements) > 1:
                    apples_count = len([elem for elem in elements if current_map[elem[0]][elem[1]] == 'A'])
                    trees[elem_key] = apples_count

            with open(os.path.join(self.log_path, "trees_history.txt"), "a") as f:
                f.write(f"{self.step}: {trees}\n")

        elif self.substrate_name == 'clean_up':
            apples = 0
            dirt = 0

            for row in current_map:
                for elem in row:
                    if elem == 'A':
                        apples += 1
                    elif elem == 'D':
                        dirt += 1
            
            with open(os.path.join(self.log_path, "apples_history.txt"), "a") as f:
                f.write(f"{self.step}: A_{apples} - D_{dirt}\n ")





    def save_log(self):
        recreate_simulation.recreate_records(record_path=self.log_path, players=self.substrate_config.player_names, is_focal_player=self.substrate_config.is_focal_player)

    @staticmethod
    def save_image(image, path):
        io.imsave(path, image)

        
    def add_description(self, description):
        canvas = np.ones((600, 600, 3), dtype=np.uint8) * 255
        sub_str = "You were taken out of the game by "
        observation = description["observation"]
        other_agents = description["agents_in_observation"]
        if sub_str in observation:
            murder = observation.replace(sub_str, "")
            self.put_text_on_image(canvas, sub_str, x=20, y=20)
            self.put_text_on_image(canvas, murder, x=20, y=40)
        else:
            lines = observation.split("\n")
            y = 30
            for line in lines:
                x = 10
                for char in line:
                    self.put_text_on_image(canvas, char, x=x, y=y)
                    x += 50
                y += 40
            for agent_id, name in other_agents.items():
                self.put_text_on_image(canvas, f"{agent_id}: {name}", x=10, y=y)
                y += 40
        return canvas

    @staticmethod
    def put_text_on_image(image, text, x, y):
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, 2)

    @staticmethod
    def create_or_replace_directory(directory_path):
        if os.path.exists(directory_path):
            shutil.rmtree(directory_path)
        os.makedirs(directory_path)
