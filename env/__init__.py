from gym.envs.registration import register
from env.roroDeck import RoRoDeck

register(
    id='RORODeck-v0',
    entry_point='env:RoRoDeck',
)