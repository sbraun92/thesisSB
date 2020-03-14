from env.roroDeck import RoRoDeck
import numpy as np
import pytest


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
    print(env.mandatoryCargoMask)
    print(env.vehicleCounter)
    print(env.loadedVehicles)
    print(env.gridVehicleType)
    print(env.endOfLanes)


    assert env.vehicleData.shape[0] == 5
    assert env.currentLane >= 0 and env.currentLane <= env.lanes
    assert env.grid.shape == (env.rows,env.lanes)
    assert env.sequence_no >= 0 and type(env.sequence_no)==type(0)
    assert len(env.rewardSystem) == 4
    #TODO assert reward range

#After the reset all variables should have the same value as after __init__
def test_resetMethod():
    env = RoRoDeck(False)
    vehicleData = env.vehicleData.copy()
    endOFLanes = env.endOfLanes.copy()
    grid = env.grid.copy()
    gridDestination = env.gridDestination.copy()
    gridVehicleType = env.gridVehicleType.copy()
    #TODO add a expanded grid for vehicle Type
    capacity = env.capacity.copy()
    vehicleCounter =env.vehicleCounter.copy()
    mandatoryCargoMask = env.mandatoryCargoMask.copy()
    loadedVehicles = env.loadedVehicles.copy()
    rewardSystem = env.rewardSystem.copy()
    frontier = env.frontier.copy()
    sequence_no = env.sequence_no
    currentLane= env.currentLane.copy()
    actionSpace = env.actionSpace.copy()
    possibleActions = env.possibleActions.copy()
    numberOfVehiclesLoaded = env.numberOfVehiclesLoaded.copy()

    env.reset()
    env.step(env.actionSpaceSample())
    env.step(env.actionSpaceSample())
    env.reset()

    _vehicleData = env.vehicleData
    _endOFLanes = env.endOfLanes
    _grid = env.grid
    _gridDestination = env.gridDestination
    _gridVehicleType = env.gridVehicleType
    _capacity = env.capacity
    _vehicleCounter =env.vehicleCounter
    _mandatoryCargoMask = env.mandatoryCargoMask
    _loadedVehicles = env.loadedVehicles
    _rewardSystem = env.rewardSystem
    _frontier = env.frontier
    _sequence_no = env.sequence_no
    _currentLane= env.currentLane
    _actionSpace = env.actionSpace
    _possibleActions = env.possibleActions
    _numberOfVehiclesLoaded = env.numberOfVehiclesLoaded


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

    vehicleData = env.vehicleData.copy()
    endOFLanes = env.endOfLanes.copy()
    grid = env.grid.copy()
    gridDestination = env.gridDestination.copy()
    gridVehicleType = env.gridVehicleType.copy()
    # TODO add a expanded grid for vehicle Type
    capacity = env.capacity.copy()
    vehicleCounter = env.vehicleCounter.copy()
    mandatoryCargoMask = env.mandatoryCargoMask.copy()
    loadedVehicles = env.loadedVehicles.copy()
    rewardSystem = env.rewardSystem.copy()
    frontier = env.frontier.copy()
    sequence_no = env.sequence_no
    currentLane = env.currentLane.copy()
    actionSpace = env.actionSpace.copy()
    possibleActions = env.possibleActions.copy()
    numberOfVehiclesLoaded = env.numberOfVehiclesLoaded.copy()

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
    assert (env.gridDestination == gridDestination).all()
    assert env.endOfLanes[currentLane] == endOFLanes[currentLane]+length
    assert env.sequence_no == sequence_no+1
    assert (env.loadedVehicles == loadedVehicles).all()
    assert (env.vehicleCounter[currentLane] == vehicleCounter[currentLane]+1)


    #TODO teste weiter ab


    _vehicleData = env.vehicleData
    _grid = env.grid
    _gridDestination = env.gridDestination
    _gridVehicleType = env.gridVehicleType
    _capacity = env.capacity
    _vehicleCounter = env.vehicleCounter
    _mandatoryCargoMask = env.mandatoryCargoMask
    _loadedVehicles = env.loadedVehicles
    _rewardSystem = env.rewardSystem
    _frontier = env.frontier
    _sequence_no = env.sequence_no
    _currentLane = env.currentLane
    _actionSpace = env.actionSpace
    _possibleActions = env.possibleActions
    _numberOfVehiclesLoaded = env.numberOfVehiclesLoaded


    assert (_vehicleData == vehicleData).all()
    assert (_rewardSystem == rewardSystem).all()



    #TODO weitere TEst
    #testen wenn eine illigale Action gewählt wurde -> Was soll da überhaupt passieren
    #testen wenn done dann env unveränderbar
    # ....
