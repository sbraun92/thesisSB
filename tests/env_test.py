from env.roroDeck import RoRoDeck
import numpy as np
import pytest
import numpy as np

np.random.seed(0)

def test_RORODeck():
    env = RoRoDeck(False)
    env.reset()
    done = False
    i = 0

    while(not done):
        observation_, reward, done, info = env.step(env.actionSpaceSample())
        i+=1
        assert i <= 100

def test_envParam():
    env = RoRoDeck(False)
    env.reset()

    #ToDO
    print(env.frontier)
    print(env.capacity)
    print(env.mandatory_Cargo_Mask)
    print(env.vehicle_Counter)
    print(env.loaded_Vehicles)
    print(env.grid_Vehicle_Type)
    print(env.end_of_Lanes)


    assert env.vehicle_Data.shape[0] == 5
    assert env.current_Lane >= 0 and env.current_Lane <= env.lanes
    assert env.grid.shape == (env.rows,env.lanes)
    assert env.sequence_no >= 0 and type(env.sequence_no)==type(0)
    assert len(env.reward_System) == 4
    #TODO assert reward range

#After the reset all variables should have the same value as after __init__
def test_resetMethod():
    env = RoRoDeck(False)
    vehicleData = env.vehicle_Data.copy()
    endOFLanes = env.end_of_Lanes.copy()
    grid = env.grid.copy()
    gridDestination = env.grid_Destination.copy()
    gridVehicleType = env.grid_Vehicle_Type.copy()
    #TODO add a expanded grid for vehicle Type
    capacity = env.capacity.copy()
    vehicleCounter =env.vehicle_Counter.copy()
    mandatoryCargoMask = env.mandatory_Cargo_Mask.copy()
    loadedVehicles = env.loaded_Vehicles.copy()
    rewardSystem = env.reward_System.copy()
    frontier = env.frontier.copy()
    sequence_no = env.sequence_no
    currentLane= env.current_Lane.copy()
    actionSpace = env.actionSpace.copy()
    possibleActions = env.possibleActions.copy()
    numberOfVehiclesLoaded = env.number_of_vehicles_loaded.copy()

    env.reset()
    env.step(env.actionSpaceSample())
    env.step(env.actionSpaceSample())
    env.reset()

    _vehicleData = env.vehicle_Data
    _endOFLanes = env.end_of_Lanes
    _grid = env.grid
    _gridDestination = env.grid_Destination
    _gridVehicleType = env.grid_Vehicle_Type
    _capacity = env.capacity
    _vehicleCounter =env.vehicle_Counter
    _mandatoryCargoMask = env.mandatory_Cargo_Mask
    _loadedVehicles = env.loaded_Vehicles
    _rewardSystem = env.reward_System
    _frontier = env.frontier
    _sequence_no = env.sequence_no
    _currentLane= env.current_Lane
    _actionSpace = env.actionSpace
    _possibleActions = env.possibleActions
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
    endOFLanes = env.end_of_Lanes.copy()
    grid = env.grid.copy()
    gridDestination = env.grid_Destination.copy()
    gridVehicleType = env.grid_Vehicle_Type.copy()
    # TODO add a expanded grid for vehicle Type
    capacity = env.capacity.copy()
    vehicleCounter = env.vehicle_Counter.copy()
    mandatoryCargoMask = env.mandatory_Cargo_Mask.copy()
    loadedVehicles = env.loaded_Vehicles.copy()
    rewardSystem = env.reward_System.copy()
    frontier = env.frontier.copy()
    sequence_no = env.sequence_no
    currentLane = env.current_Lane.copy()
    actionSpace = env.actionSpace.copy()
    possibleActions = env.possibleActions.copy()
    numberOfVehiclesLoaded = env.number_of_vehicles_loaded.copy()

    np.random.seed(0)


    action = actionSpace[mandatoryCargoMask][0]
    if action not in possibleActions:
        action = env.actionSpaceSample()

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
    assert (env.grid_Destination == gridDestination).all()
    assert env.end_of_Lanes[currentLane] == endOFLanes[currentLane] + length
    assert env.sequence_no == sequence_no+1
    assert (env.loaded_Vehicles == loadedVehicles).all()
    assert (env.vehicle_Counter[currentLane] == vehicleCounter[currentLane] + 1)


    #TODO teste weiter ab


    _vehicleData = env.vehicle_Data
    _grid = env.grid
    _gridDestination = env.grid_Destination
    _gridVehicleType = env.grid_Vehicle_Type
    _capacity = env.capacity
    _vehicleCounter = env.vehicle_Counter
    _mandatoryCargoMask = env.mandatory_Cargo_Mask
    _loadedVehicles = env.loaded_Vehicles
    _rewardSystem = env.reward_System
    _frontier = env.frontier
    _sequence_no = env.sequence_no
    _currentLane = env.current_Lane
    _actionSpace = env.actionSpace
    _possibleActions = env.possibleActions
    _numberOfVehiclesLoaded = env.number_of_vehicles_loaded


    assert (_vehicleData == vehicleData).all()
    assert (_rewardSystem == rewardSystem).all()



    #TODO weitere TEst
    #testen wenn eine illigale Action gewählt wurde -> Was soll da überhaupt passieren
    #testen wenn done dann env unveränderbar
    # ....


def test_possible_actions():
    pass


def test_end_of_lane_Method():
    pass

def test_shifts_caused():
    pass

def test_findCurrentLane():
    pass

def test_switch_current_lane():
    pass

def test_is_action_legal():
    pass

def test_is_terminal_state():
    env = RoRoDeck(True)
    assert env._isTerminalState() == False

    env.reset()
    assert env._isTerminalState() == False

    for i in range(4):
        state, reward, done, info = env.step(env.actionSpaceSample())
    assert env._isTerminalState() == False

    while not done:
        env.step(env.actionSpaceSample())
    assert env._isTerminalState() == True





def test_get_Current_State():
    env = RoRoDeck(True)

    state = env.current_state
    assert np.shape(state) == (12,)
    env.reset()
    state = env.current_state
    assert np.shape(state) == (12,)

    env = RoRoDeck(False)

    state = env.current_state
    assert np.shape(state) == (32,)
    env.reset()
    state = env.current_state
    assert np.shape(state) == (32,)