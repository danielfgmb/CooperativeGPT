from game_environment.scene_descriptor.observations_generator import ObservationsGenerator
from game_environment.substrates.python.commons_harvest_language import ASCII_MAP

players = ['agent1', 'agent2', 'agent3']
obs_gen = ObservationsGenerator(ASCII_MAP, players)
orientation_map = {0: 'North', 1: 'East', 2: 'South', 3: 'West'}

def test_get_element_global_pos():
    self_global_pos = (10, 5)

    # Test case 1: agent_orientation = 0
    el_local_pos = (5, 3)
    self_local_pos = (0, 3)
    agent_orientation = 0
    expected_output = [15, 5]
    element_global_pos = obs_gen.get_element_global_pos(el_local_pos, self_local_pos, self_global_pos, agent_orientation)
    assert element_global_pos == expected_output, f'Expected {expected_output}, got {element_global_pos}.Failed with agent_orientation = {agent_orientation}: {orientation_map[agent_orientation]}'

    # Test case 2: agent_orientation = 1
    el_local_pos = (0, 0)
    self_local_pos = (0, 5)
    agent_orientation = 1
    expected_output = [5, 5]
    element_global_pos = obs_gen.get_element_global_pos(el_local_pos, self_local_pos, self_global_pos, agent_orientation)
    assert element_global_pos == expected_output, f'Expected {expected_output}, got {element_global_pos}.Failed with agent_orientation = {agent_orientation}: {orientation_map[agent_orientation]}'

    # Test case 3: agent_orientation = 2
    el_local_pos = (2, 3)
    self_local_pos = (7, 3)
    agent_orientation = 2
    expected_output = [15, 5]
    element_global_pos = obs_gen.get_element_global_pos(el_local_pos, self_local_pos, self_global_pos, agent_orientation)
    assert element_global_pos == expected_output, f'Expected {expected_output}, got {element_global_pos}.Failed with agent_orientation = {agent_orientation}: {orientation_map[agent_orientation]}'

    # Test case 4: agent_orientation = 3
    el_local_pos = (7, 9)
    self_local_pos = (7, 4)
    agent_orientation = 3
    expected_output = [5, 5]
    element_global_pos = obs_gen.get_element_global_pos(el_local_pos, self_local_pos, self_global_pos, agent_orientation)
    assert element_global_pos == expected_output, f'Expected {expected_output}, got {element_global_pos}.Failed with agent_orientation = {agent_orientation}: {orientation_map[agent_orientation]}'

    print("All test cases pass")
    
def test_connected_elems_map():
    # Test case 1: Single element
    observed_map = "A"
    elements_to_find = ["A"]
    expected_output = {1: {'center': (0, 0), 'elements': [[0, 0]]}}
    assert obs_gen.connected_elems_map(observed_map, elements_to_find) == expected_output, f"Failed for a single element"

    # Test case 2: A single connected component
    observed_map = "AB\nBA"
    elements_to_find = ["A", "B"]
    expected_output = {1: {'center': (0, 0), 'elements': [[0, 0], [0, 1], [1, 0], [1, 1]]}} # All elements are connected
    elements_found = obs_gen.connected_elems_map(observed_map, elements_to_find)
    assert elements_found == expected_output, f"Expected {expected_output}, got {elements_found}. Failed for a single connected component"

    # Test case 3: Multiple connected components
    observed_map = [
        "-AA---",
        "GA----",
        "----AG",
        "----GA",
        ]
    observed_map = "\n".join(observed_map)
    elements_to_find = ["A", "G"]
    expected_output = {1: {'center': (0, 1), 'elements': [[0, 1], [0, 2], [1, 0], [1, 1]]}, 2: {'center': (2, 4), 'elements': [[2, 4], [2, 5], [3, 4], [3, 5]]}}
    elements_found = obs_gen.connected_elems_map(observed_map, elements_to_find)
    assert elements_found == expected_output, f"Expected {expected_output}, got {elements_found}. Failed for multiple connected components"


    # Test case 4: No elements found
    observed_map = "AB\nCD"
    elements_to_find = ["E"]
    expected_output = {}
    elements_found = obs_gen.connected_elems_map(observed_map, elements_to_find)
    assert elements_found == expected_output, f"Expected {expected_output}, got {elements_found}. Failed for multiple connected components"

    print("All test cases pass")

def test_get_trees_descriptions():
    # Test case 1: No trees found
    observed_map = '-----------\n-----------\n-----------\n-----------\n-----------\nWWWWWWWWWWW\nAFFFFAFFFFF\nFFFFAAAFFFF\nFFFGAG2AFFF\nFFFFA#AFFFF\nFFFFFGFFFFF'
    local_map_position = (9,5)
    global_position = (8, 19)
    agent_orientation = 1
    expected_output = ['Observed an apple at position [3, 22]. This apple belongs to tree 4', 'Observed tree 4 at position [1, 21]. This tree has 1 apples remaining and 0 grass for apples growing on the observed map. The tree might have more apples and grass on the global map', 'Observed an apple at position [8, 22]. This apple belongs to tree 6', 'Observed an apple at position [7, 21]. This apple belongs to tree 6', 'Observed an apple at position [8, 21]. This apple belongs to tree 6', 'Observed an apple at position [9, 21]. This apple belongs to tree 6', 'Observed grass to grow apples at position [6, 20]. This grass belongs to tree 6', 'Observed an apple at position [7, 20]. This apple belongs to tree 6', 'Observed grass to grow apples at position [8, 20]. This grass belongs to tree 6', 'Observed an apple at position [7, 19]. This apple belongs to tree 6', 'Observed grass to grow apples at position [8, 18]. This grass belongs to tree 6', 'Observed tree 6 at position [8, 20]. This tree has 6 apples remaining and 3 grass for apples growing on the observed map. The tree might have more apples and grass on the global map']
    trees_descriptions = obs_gen.get_trees_descriptions(observed_map, local_map_position, global_position, agent_orientation)
    assert trees_descriptions == expected_output, f"Expected {expected_output}, got {trees_descriptions}. Failed for no trees found"