from env.roroDeck import RoRoDeck
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


    assert env.vehicleData.shape[0] == 5
    assert env.currentLane >= 0 and env.currentLane <= env.lanes
    assert env.grid.shape == (env.rows,env.lanes)
    assert env.sequence_no >= 0 and type(env.sequence_no)==type(0)
    assert len(env.rewardSystem) == 4



