from env.RoRoDeck import RoRoDeck
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




