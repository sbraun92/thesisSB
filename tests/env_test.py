from env.roroDeck import RoRoDeck
import numpy as np
import pytest
import numpy as np
import gym
import env
#from stable_baselines.common.env_checker import check_env
np.random.seed(0)

def test_OpenAiCompliance():
    env = gym.make('RORODeck-v0')
    #check_env(env)
    env.reset()
    env.step(env.actionSpaceSample())

def test_RORODeck():
    env = RoRoDeck(False)
    env.reset()
    done = False
    i = 0

    while(not done):
        observation_, reward, done, info = env.step(env.action_space_sample())
        i+=1
        assert i <= 100

def test_envParam():
    env = RoRoDeck(False)
    env.reset()

    #ToDO
    print(env.frontier)
    print(env.capacity)
    print(env.mandatory_cargo_mask)
    print(env.vehicle_Counter)
    print(env.loaded_Vehicles)
    print(env.grid_vehicle_type)
    print(env.end_of_lanes)


    assert env.vehicle_Data.shape[0] == 6
    assert env.current_Lane >= 0 and env.current_Lane <= env.lanes
    assert env.grid.shape == (env.rows,env.lanes)
    assert env.sequence_no >= 0 and type(env.sequence_no)==type(0)
    assert len(env.reward_system) == 4
    #TODO assert reward range

#After the reset all variables should have the same value as after __init__
def test_resetMethod():
    env = RoRoDeck(False)
    vehicleData = env.vehicle_Data.copy()
    endOFLanes = env.end_of_lanes.copy()
    grid = env.grid.copy()
    gridDestination = env.grid_destination.copy()
    gridVehicleType = env.grid_vehicle_type.copy()
    #TODO add a expanded grid for vehicle Type
    capacity = env.capacity.copy()
    vehicleCounter =env.vehicle_Counter.copy()
    mandatoryCargoMask = env.mandatory_cargo_mask.copy()
    loadedVehicles = env.loaded_Vehicles.copy()
    rewardSystem = env.reward_system.copy()
    frontier = env.frontier.copy()
    sequence_no = env.sequence_no
    currentLane= env.current_Lane.copy()
    actionSpace = env.actionSpace.copy()
    possibleActions = env.possible_actions.copy()
    numberOfVehiclesLoaded = env.number_of_vehicles_loaded.copy()

    env.reset()
    env.step(env.action_space_sample())
    env.step(env.action_space_sample())
    env.reset()

    _vehicleData = env.vehicle_Data
    _endOFLanes = env.end_of_lanes
    _grid = env.grid
    _gridDestination = env.grid_destination
    _gridVehicleType = env.grid_vehicle_type
    _capacity = env.capacity
    _vehicleCounter =env.vehicle_Counter
    _mandatoryCargoMask = env.mandatory_cargo_mask
    _loadedVehicles = env.loaded_Vehicles
    _rewardSystem = env.reward_system
    _frontier = env.frontier
    _sequence_no = env.sequence_no
    _currentLane= env.current_Lane
    _actionSpace = env.actionSpace
    _possibleActions = env.possible_actions
    _numberOfVehiclesLoaded = env.number_of_vehicles_loaded


    assert (_vehicleData == vehicleData).all()
    assert (_endOFLanes == endOFLanes).all()
    assert (_grid == grid).all()
    assert (_gridDestination == gridDestination).all()
    assert (_gridVehicleType == gridVehicleType).all()
    assert (_capacity == capacity).all()
    assert (_vehicleCounter == vehicleCounter).all()
    assert (_mandatoryCargoMask == mandatoryCargoMask).all()
    assert (_loadedVehicles == loadedVehicles).all()
    assert (_rewardSystem == rewardSystem).all()
    assert (_frontier == frontier).all()
    assert _sequence_no == sequence_no
    assert (_currentLane == currentLane).all()
    assert (_actionSpace == actionSpace).all()
    assert (_possibleActions == possibleActions).all()
    assert (_numberOfVehiclesLoaded == numberOfVehiclesLoaded).all()


def test_stepMethod():
    env = RoRoDeck(False)
    env.reset()

    vehicleData = env.vehicle_Data.copy()
    endOFLanes = env.end_of_lanes.copy()
    grid = env.grid.copy()
    gridDestination = env.grid_destination.copy()
    gridVehicleType = env.grid_vehicle_type.copy()
    # TODO add a expanded grid for vehicle Type
    capacity = env.capacity.copy()
    vehicleCounter = env.vehicle_Counter.copy()
    mandatoryCargoMask = env.mandatory_cargo_mask.copy()
    loadedVehicles = env.loaded_Vehicles.copy()
    rewardSystem = env.reward_system.copy()
    frontier = env.frontier.copy()
    sequence_no = env.sequence_no
    currentLane = env.current_Lane.copy()
    actionSpace = env.actionSpace.copy()
    possibleActions = env.possible_actions.copy()
    numberOfVehiclesLoaded = env.number_of_vehicles_loaded.copy()

    np.random.seed(0)


    action = actionSpace[mandatoryCargoMask][0]
    if action not in possibleActions:
        action = env.action_space_sample()

    env.step(action)



    destination = vehicleData[1][action]
    mandatory = vehicleData[2][action]
    length = vehicleData[3][action]

    for i in range(length):
        grid.T[currentLane][endOFLanes[currentLane]+i] = sequence_no
        gridDestination.T[currentLane][endOFLanes[currentLane]+i] = destination

    loadedVehicles[currentLane][vehicleCounter[currentLane]] = action


    print(env.grid)
    print(grid)

    assert (env.grid == grid).all()
    assert (env.grid_destination == gridDestination).all()
    assert env.end_of_lanes[currentLane] == endOFLanes[currentLane] + length
    assert env.sequence_no == sequence_no+1
    assert (env.loaded_Vehicles == loadedVehicles).all()
    assert (env.vehicle_Counter[currentLane] == vehicleCounter[currentLane] + 1)


    #TODO teste weiter ab


    _vehicleData = env.vehicle_Data
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
    _actionSpace = env.actionSpace
    _possibleActions = env.possible_actions
    _numberOfVehiclesLoaded = env.number_of_vehicles_loaded


    assert (_vehicleData == vehicleData).all()
    assert (_rewardSystem == rewardSystem).all()



    #TODO weitere TEst
    #teste reefer position
    #testen wenn eine illigale Action gewählt wurde -> Was soll da überhaupt passieren
    #testen wenn done dann env unveränderbar
    # ....


def test_possible_actions():
    env = RoRoDeck(True)
    env.vehicle_Data[5][4]=0 #disable reefer for test
    env.reset()
    number_of_vehicle = env.vehicle_Data[4][0]

    assert len(env.vehicle_Data.T) == len(env.possible_actions) == 5

    for i in range(number_of_vehicle):
        assert len(env.vehicle_Data.T) == len(env.possible_actions) == 5
        env.step(0)

    assert len(env.possible_actions) == len(env.vehicle_Data.T) - 1

def test_Reefer_Postions():
    env = RoRoDeck(True)
    env.vehicle_Data[5][4] = 1  # enable reefer for test
    env.reset()

    env.current_Lane = 0
    assert env.grid_reefer.T[env.current_Lane][-1]==1
    assert len(env.vehicle_Data.T) == len(env.get_possible_actions_of_state()) == 5

    env.current_Lane = 4
    assert env.grid_reefer.T[env.current_Lane][-1]==0
    assert len(env.get_possible_actions_of_state()) == 4
    assert 4 not in env.get_possible_actions_of_state()

def test_end_of_lane_Method():
    env = RoRoDeck(True)
    env.reset()
    end_of_lanes = env.end_of_lanes
    assert np.all(end_of_lanes == env.inital_end_of_lanes)

    current_lane = env.current_Lane
    length = env.vehicle_Data[3][env.possible_actions[0]]

    env.step(env.possible_actions[0])

    assert env.end_of_lanes[current_lane] == end_of_lanes[current_lane]+length


def test_shifts_caused():
    pass

def test_findCurrentLane():
    pass

def test_switch_current_lane():
    pass

def test_is_action_legal():
    env = RoRoDeck(True)
    env.reset()
    number_vehicle = env.vehicle_Data[4][0]

    assert env._is_action_legal(0) == True
    for i in range(number_vehicle):
        assert env._is_action_legal(0) == True
        env.step(0)
    assert env._is_action_legal(0) == False




def test_is_terminal_state():
    env = RoRoDeck(True)
    assert env._is_terminal_state() == False

    env.reset()
    assert env._is_terminal_state() == False

    for i in range(4):
        state, reward, done, info = env.step(env.action_space_sample())
    assert env._is_terminal_state() == False

    while not done:
        state, reward, done, info = env.step(env.action_space_sample())
    assert env._is_terminal_state() == True





def test_get_Current_State():
    env = RoRoDeck(True)

    state = env.current_state
    assert np.shape(state) == (13,)
    env.reset()
    state = env.current_state
    assert np.shape(state) == (13,)

    env.step(env.actionSpaceSample())
    assert not np.all(state==env.current_state)

    env = RoRoDeck(False)

    state = env.current_state
    assert np.shape(state) == (26,) #was 83 TODO
    env.reset()
    state = env.current_state
    assert np.shape(state) == (26,)

    env.step(env.actionSpaceSample())
    assert not np.all(state == env.current_state)
