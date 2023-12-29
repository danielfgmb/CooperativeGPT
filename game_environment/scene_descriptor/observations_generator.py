"""

File: observations_descriptor.py
Description: Implements required functions for the observations descriptor

"""

import numpy as np
import ast
from scipy.ndimage import label, center_of_mass
from collections import defaultdict
import re
from game_environment.utils import connected_elems_map


class ObservationsGenerator (object):
    """
    Description: Implements required functions for the observations descriptor. 
            This class is used to generate the descriptions of the observations of the agents based on the 
            scene descriptor module which provides the observations in ascci format
    """

    def __init__(self, global_map:str, players_names: list, substrate_name:str):
        """
        Description: Initializes the observations generator

        Args:
            global_map (str): Global map in ascci format
            players_names (list): List with the names of the players
            substrate_name (str): Name of the substrate
        """
        
        self.global_map = global_map
        self.players_names = players_names
        self.self_symbol = '#'
        self.other_players_symbols = [str(i) for i in range(len(players_names))]
        self.substrate_name = substrate_name

        if self.substrate_name == 'commons_harvest_open':
            self.global_trees = connected_elems_map(self.global_map, ['A', 'G'])
        elif self.substrate_name == 'clean_up':
            self.river_bank =  connected_elems_map(self.global_map, ['=','+']) # Chars that represent the river bank
            self.apple_field_edge = connected_elems_map(self.global_map, ['^','T']) # Chars that represent the apple field edge


    def get_element_global_pos(self, el_local_pos, self_local_pos, self_global_pos, agent_orientation=0) -> list[int]:
        """
        Description: Returns the global position of an element given its local position and the global position of the agent

        Args:
            el_local_pos (tuple): Local position of the element
            self_local_pos (tuple): Local position of the agent 
            self_global_pos (tuple): Global position of the agent
            agent_orientation (int, optional): Orientation of the agent. Defaults to 0.

        Returns:
            list[int]: Global position of the element
        """
        if agent_orientation == 0:
            element_global = (el_local_pos[0] - self_local_pos[0]) + self_global_pos[0],\
                                (el_local_pos[1] - self_local_pos[1]) + self_global_pos[1]
        elif agent_orientation == 1:
            element_global = (el_local_pos[1] - self_local_pos[1]) + self_global_pos[0],\
                               -1 * (el_local_pos[0] - self_local_pos[0]) + self_global_pos[1]
        elif agent_orientation == 2:
            element_global = -1 * (el_local_pos[0] - self_local_pos[0]) + self_global_pos[0],\
                                -1 * (el_local_pos[1] - self_local_pos[1]) + self_global_pos[1]
        elif agent_orientation == 3:
            element_global = -1 * (el_local_pos[1] - self_local_pos[1]) + self_global_pos[0],\
                                (el_local_pos[0] - self_local_pos[0]) + self_global_pos[1]
        return list(element_global)

    


    def get_all_observations_descriptions(self,  agents_observations_str: str) -> dict[str, list[str]]:
        """
        Description: Returns a dictionary with the descriptions of the observations of the agents

        Args:
            agents_observations_str (str): Observations of the agents in ascci format
            
        Returns:
            dict[str, list[str]]: Dictionary with the descriptions of the observations in a list by agent name
        """
        agents_observations = ast.literal_eval(agents_observations_str)
        observations_description_per_agent = {}
        for agent_id, agent_dict in agents_observations.items():
            agent = self.players_names[agent_id]
            observations_description_per_agent[agent] = self.get_observations_per_agent(agent_dict)
            
        return observations_description_per_agent
    

    def get_observations_per_agent(self, agent_dict: dict):
        """
        Description: Returns a list with the descriptions of the observations of the agent

        Args:
            agent_dict (dict): Dictionary with the observations of the agent
        
        Returns:
            list: List with the descriptions of the observations of the agent
        """
        list_of_observations = []
        if agent_dict['observation'].startswith('There are no observations: You were taken '):
            list_of_observations.append(str(agent_dict['observation'] + ' at position {}'.format(agent_dict['global_position'])))
            return list_of_observations
        else:
            local_observation_map = agent_dict['observation']
            local_map_position = (9,5)
            global_position = agent_dict['global_position']
            agent_orientation = agent_dict['orientation']


            if self.substrate_name == 'commons_harvest_open':
                # Get trees descriptions
                trees_descriptions = self.get_trees_descriptions(local_observation_map, local_map_position, global_position, agent_orientation) 
                list_of_observations.extend(trees_descriptions)
            elif self.substrate_name == 'clean_up':
                # Get objects of clean up descriptions
                items_descriptions = self.get_clean_up_descriptions(local_observation_map, local_map_position, global_position, agent_orientation)
                list_of_observations.extend(items_descriptions)

            # Get agents observed descriptions
            for i, row in enumerate(local_observation_map.split('\n')):
                for j, char in enumerate(row):
                    if re.match(r'^[0-9]$', char):
                        agent_id = int(char)
                        agent_global_pos = self.get_element_global_pos((i,j), local_map_position, global_position, agent_orientation)
                        list_of_observations.append("Observed agent {} at position {}".format(agent_id, agent_global_pos))
        
        return list_of_observations

    def get_trees_descriptions(self, local_map:str, local_position:tuple, global_position:tuple, agent_orientation:int):
        """
        Description: Returns a list with the descriptions of the trees observed by the agent

        Args:
            local_map (str): Local map in ascci format
            local_position (tuple): Local position of the agent
            global_position (tuple): Global position of the agent
            agent_orientation (int): Orientation of the agent
            
        Returns:
            list: List with the descriptions of the trees observed by the agent
        """
        tree_elements = ['A', 'G']
        elements_to_find = tree_elements + self.other_players_symbols + [self.self_symbol]
        local_tree_elements = connected_elems_map(local_map, elements_to_find=elements_to_find)
        list_trees_observations = []
        trees_observed = {}

        for global_tree_id, global_tree_data in self.global_trees.items():
            apple_count, grass_count = 0, 0
            for local_tree_data in local_tree_elements.values():
                # Check if the group is a tree element
                first_element = local_tree_data['elements'][0]
                element_type = local_map.split('\n')[first_element[0]][first_element[1]]
                if element_type not in tree_elements:
                    continue

                # Continue if the tree has already been observed
                if global_tree_id in trees_observed.get(element_type, []): 
                    continue

                local_tree_center = local_tree_data['center']
                local_center_real_pos = self.get_element_global_pos(local_tree_center, local_position, global_position, agent_orientation)

                # Check if the local tree corresponds to the global tree
                if local_center_real_pos not in global_tree_data['elements']:
                    continue

                # Find the cluster tree of the local tree
                trees_observed[element_type] = trees_observed.get(element_type, []) + [global_tree_id]
    
                for apple in local_tree_data['elements']:
                    apple_global_pos = self.get_element_global_pos(apple, local_position, global_position, agent_orientation)
                    if local_map.split('\n')[apple[0]][apple[1]] == 'G':
                        list_trees_observations.append("Observed grass to grow apples at position {}. This grass belongs to tree {}"
                                                    .format(apple_global_pos, global_tree_id))
                        grass_count += 1
                    elif local_map.split('\n')[apple[0]][apple[1]] == 'A':
                        list_trees_observations.append("Observed an apple at position {}. This apple belongs to tree {}"
                                                    .format(apple_global_pos, global_tree_id ))
                        apple_count += 1

            if apple_count > 0 or grass_count > 0:      
                list_trees_observations.append("Observed tree {} at position {}. This tree has {} apples remaining and {} grass for apples growing on the observed map. The tree might have more apples and grass on the global map"
                                                .format(global_tree_id, list(global_tree_data['center']), apple_count, grass_count))
        return list_trees_observations


    def get_clean_up_descriptions (self, local_map:str, local_position:tuple, global_position:tuple, agent_orientation:int):
        """
        Description: Returns a list with the descriptions of the objects observed by the agent

        Args:
            local_map (str): Local map in ascci format
            local_position (tuple): Local position of the agent
            global_position (tuple): Global position of the agent
            agent_orientation (int): Orientation of the agent
            
        Returns:
            list: List with the descriptions of the objects observed by the agent
        """
        
        items_observed = []
        # Get apples (A) and dirt (D) observed descriptions
        for i, row in enumerate(local_map.split('\n')):
            for j, char in enumerate(row):
                if char == 'A':
                    apple_global_pos = self.get_element_global_pos((i,j), local_position, global_position, agent_orientation)
                    items_observed.append("Observed an apple at position {}".format(apple_global_pos))

                elif char == 'D':
                    dirt_global_pos = self.get_element_global_pos((i,j), local_position, global_position, agent_orientation)
                    items_observed.append("Observed dirt on the river at position {}".format(dirt_global_pos))

                for elm in self.river_bank.values():
                    if (i,j) in elm['elements']:
                        river_bank_global_pos = self.get_element_global_pos((i,j), local_position, global_position, agent_orientation)
                        items_observed.append("Observed river bank at position {}".format(river_bank_global_pos))
                
                for elm in self.apple_field_edge.values():
                    if (i,j) in elm['elements']:
                        apple_field_edge_global_pos = self.get_element_global_pos((i,j), local_position, global_position, agent_orientation)
                        items_observed.append("Observed apple field edge at position {}".format(apple_field_edge_global_pos))

        return items_observed
