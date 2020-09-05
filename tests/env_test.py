import gym
import numpy as np

from env.roroDeck import RoRoDeck

#from stable_baselines.common.env_checker import check_env
np.random.seed(0)


# TODO test action_space and reward_space
def test_open_ai_compliance():
    env = gym.make('RORODeck-v0')
    #env.reset()
    #env.step(env.action_space_sample())
#    check_env(env)

def test_roro_deck():
    env = RoRoDeck(False)
    env.reset()
    done = False
    i = 0

    while not done:
        observation_, reward, done, info = env.step(env.action_space_sample())
        i += 1
        assert i <= 100


def test_env_parameter():
    env = RoRoDeck(False)
    env.reset()

    # ToDO
    print(env.frontier)
    print(env.capacity)
    print(env.mandatory_cargo_mask)
    print(env.vehicle_Counter)
    print(env.loaded_Vehicles)
    print(env.grid_vehicle_type)
    print(env.end_of_lanes)

    assert env.vehicle_data.shape[0] == 6
    assert 0 <= env.current_Lane <= env.lanes
    assert env.grid.shape == (env.rows, env.lanes)
    assert env.sequence_no >= 0 and isinstance(env.sequence_no, int)
    assert len(env.reward_system) == 4


# After the reset all variables should have the same value as after __init__
def test_reset_method():
    env = RoRoDeck(False)
    vehicle_data = env.vehicle_data.copy()
    end_of_lanes = env.end_of_lanes.copy()
    grid = env.grid.copy()
    grid_destination = env.grid_destination.copy()
    grid_vehicle_type = env.grid_vehicle_type.copy()
    # TODO add a expanded grid for vehicle Type
    capacity = env.capacity.copy()
    vehicle_counter = env.vehicle_Counter.copy()
    mandatory_cargo_mask = env.mandatory_cargo_mask.copy()
    loaded_vehicles = env.loaded_Vehicles.copy()
    reward_system = env.reward_system.copy()
    frontier = env.frontier.copy()
    sequence_no = env.sequence_no
    current_lane = env.current_Lane.copy()
    action_space = env.action_space.copy()
    possible_actions = env.possible_actions.copy()
    number_of_vehicles_loaded = env.number_of_vehicles_loaded.copy()

    env.reset()
    env.step(env.action_space_sample())
    env.step(env.action_space_sample())
    env.reset()

    _vehicleData = env.vehicle_data
    _endOFLanes = env.end_of_lanes
    _grid = env.grid
    _gridDestination = env.grid_destination
    _gridVehicleType = env.grid_vehicle_type
    _capacity = env.capacity
    _vehicleCounter = env.vehicle_Counter
    _mandatoryCargoMask = env.mandatory_cargo_mask
    _loadedVehicles = env.loaded_Vehicles
    _rewardSystem = env.reward_system
    _frontier = env.frontier
    _sequence_no = env.sequence_no
    _currentLane = env.current_Lane
    _actionSpace = env.action_space
    _possibleActions = env.possible_actions
    _numberOfVehiclesLoaded = env.number_of_vehicles_loaded

    assert (_vehicleData == vehicle_data).all()
    assert (_endOFLanes == end_of_lanes).all()
    assert (_grid == grid).all()
    assert (_gridDestination == grid_destination).all()
    assert (_gridVehicleType == grid_vehicle_type).all()
    assert (_capacity == capacity).all()
    assert (_vehicleCounter == vehicle_counter).all()
    assert (_mandatoryCargoMask == mandatory_cargo_mask).all()
    assert (_loadedVehicles == loaded_vehicles).all()
    assert (_rewardSystem == reward_system).all()
    assert (_frontier == frontier).all()
    assert _sequence_no == sequence_no
    assert (_currentLane == current_lane).all()
    assert (_actionSpace == action_space).all()
    assert (_possibleActions == possible_actions).all()
    assert (_numberOfVehiclesLoaded == number_of_vehicles_loaded).all()


def test_stepMethod():
    env = RoRoDeck(False)
    env.reset()

    vehicle_data = env.vehicle_data.copy()
    end_of_lanes = env.end_of_lanes.copy()
    grid = env.grid.copy()
    grid_destination = env.grid_destination.copy()
    grid_vehicle_type = env.grid_vehicle_type.copy()
    capacity = env.capacity.copy()
    vehicle_counter = env.vehicle_Counter.copy()
    mandatory_cargo_mask = env.mandatory_cargo_mask.copy()
    loaded_vehicles = env.loaded_Vehicles.copy()
    reward_system = env.reward_system.copy()
    frontier = env.frontier.copy()
    sequence_no = env.sequence_no
    current_lane = env.current_Lane.copy()
    action_space = env.action_space.copy()
    possible_actions = env.possible_actions.copy()
    number_of_vehicles_loaded = env.number_of_vehicles_loaded.copy()

    np.random.seed(0)

    action = action_space[mandatory_cargo_mask][0]
    if action not in possible_actions:
        action = env.action_space_sample()

    env.step(action)

    destination = vehicle_data[3][action]
    mandatory = vehicle_data[2][action]
    length = vehicle_data[4][action]

    for i in range(length):
        grid.T[current_lane][end_of_lanes[current_lane] + i] = sequence_no
        grid_destination.T[current_lane][end_of_lanes[current_lane] + i] = destination

    loaded_vehicles[current_lane][vehicle_counter[current_lane]] = action

    assert (env.grid == grid).all()
    assert (env.grid_destination == grid_destination).all()
    assert env.end_of_lanes[current_lane] == end_of_lanes[current_lane] + length
    assert env.sequence_no == sequence_no + 1
    assert (env.loaded_Vehicles == loaded_vehicles).all()
    assert (env.vehicle_Counter[current_lane] == vehicle_counter[current_lane] + 1)

    # TODO teste weiter ab

    _vehicleData = env.vehicle_data
    _grid = env.grid
    _gridDestination = env.grid_destination
    _gridVehicleType = env.grid_vehicle_type
    _capacity = env.capacity
    _vehicleCounter = env.vehicle_Counter
    _mandatoryCargoMask = env.mandatory_cargo_mask
    _loadedVehicles = env.loaded_Vehicles
    _rewardSystem = env.reward_system
    _frontier = env.frontier
    _sequence_no = env.sequence_no
    _currentLane = env.current_Lane
    _actionSpace = env.action_space
    _possibleActions = env.possible_actions
    _numberOfVehiclesLoaded = env.number_of_vehicles_loaded

    assert (_vehicleData == vehicle_data).all()
    assert (_rewardSystem == reward_system).all()


def test_possible_actions():
    env = RoRoDeck(True)
    env.vehicle_data[5][4] = 0  # disable reefer for test
    env.reset()
    number_of_vehicle = env.vehicle_data[1][0]

    assert len(env.vehicle_data.T) == len(env.possible_actions) == 5

    for i in range(number_of_vehicle):
        assert len(env.vehicle_data.T) == len(env.possible_actions) == 5
        env.step(0)

    assert len(env.possible_actions) == len(env.vehicle_data.T) - 1


def test_reefer_positions():
    env = RoRoDeck(True)
    env.vehicle_data[5][4] = 1  # enable reefer for test
    env.reset()

    env.current_Lane = 0
    assert env.grid_reefer.T[env.current_Lane][-1] == 1
    assert len(env.vehicle_data.T) == len(env.get_possible_actions_of_state()) == 5

    env.current_Lane = 4
    assert env.grid_reefer.T[env.current_Lane][-1] == 0
    assert len(env.get_possible_actions_of_state()) == 4
    assert 4 not in env.get_possible_actions_of_state()


def test_end_of_lane_method():
    env = RoRoDeck(True)
    env.reset()
    end_of_lanes = env.end_of_lanes.copy()
    assert np.all(end_of_lanes == env.initial_end_of_lanes)

    current_lane = env.current_Lane
    length = env.vehicle_data[4][env.possible_actions[0]]

    env.step(env.possible_actions[0])

    assert env.end_of_lanes[current_lane] == end_of_lanes[current_lane] + length


# TODO
def test_shifts_caused():
    pass


def test_find_current_lane():
    env = RoRoDeck(lanes=8, rows=12)
    env.reset()
    assert env.current_Lane == 3
    env.step(env.action_space_sample())
    assert env.current_Lane == 4


# TODO delete
def test_switch_current_lane():
    pass


def test_is_action_legal():
    env = RoRoDeck(True)
    env.reset()
    number_vehicle = env.vehicle_data[1][0]

    assert env._is_action_legal(0)
    for i in range(number_vehicle):
        assert env._is_action_legal(0)
        env.step(0)
    assert not env._is_action_legal(0)


def test_is_terminal_state():
    env = RoRoDeck(True)
    assert not env._is_terminal_state()

    env.reset()
    assert not env._is_terminal_state()

    for i in range(4):
        state, reward, done, info = env.step(env.action_space_sample())
    assert not env._is_terminal_state()

    while not done:
        state, reward, done, info = env.step(env.action_space_sample())
    assert env._is_terminal_state()


def test_get_current_state():
    env = RoRoDeck(False)

    state = env.current_state
    assert np.shape(state) == (25,)
    env.reset()
    state = env.current_state
    assert np.shape(state) == (25,)

    env.step(env.action_space_sample())
    assert not np.all(state == env.current_state)
