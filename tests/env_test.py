from env.RoRoDeck import RoRoDeck
import pytest


def test_RORODeck():
    env = RoRoDeck(False)
    env.reset()
    env.step(1)

    done = False
    i = 0
    while(not done):
        env.render()
        observation_, reward, done, info = env.step(0)
        i+=1
        assert i <= 100



