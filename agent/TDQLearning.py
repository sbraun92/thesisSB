from env.RoRoDeck import RoRoDeck
import numpy as np
import matplotlib.pyplot as plt

# TODO with numpy and it is also wrong
def maxAction(Q, state, actions):
    # find intersection of ActionSpace and possible actions and return the

    argSorted_qValues = np.flipud(np.argsort(Q[state.tobytes()]))

    for ix_q_values in argSorted_qValues:
        if ix_q_values in actions:
            return ix_q_values

    '''ix_max = 0
    max_value = 0

    for action in actions:
        q_value = Q[state.tobytes()][ix_Actions[action]]
        if q_value > max_value:
            max_value = q_value
            ix_max = ix_Actions[action]

    return ix_max'''


if __name__ == '__main__':

    env = RoRoDeck()
    initState = env.reset()

    actionSpace_length = len(env.actionSpace)
    ix_Actions = np.arange(len(env.actionSpace))
    action_list = []

    # for ix, i in enumerate(env.actionSpace.keys()):
    #    ix_Actions[i] = ix
    #   action_list += [i]

    q_table = {initState.tobytes(): np.zeros(actionSpace_length)}

    ALPHA = 0.1
    GAMMA = 1.0
    EPS = 1.0

    numGames = 30000
    totalRewards = np.zeros(numGames)
    env.render()

    for i in range(numGames):
        if i % 1000 == 0:
            print('learning process epoch:', i)

        done = False
        epReward = 0
        observation = env.reset()

        while not done:
            # Show for visualisation the last training epoch

            rand = np.random.random()
            action = maxAction(q_table, observation, env.possibleActions) if rand < (1 - EPS) \
                else env.actionSpaceSample()

            observation_, reward, done, info = env.step(action)

            if observation_.tobytes() not in q_table:
                q_table[observation_.tobytes()] = np.zeros(actionSpace_length)

            epReward += reward

            action_ = maxAction(q_table, observation_, env.possibleActions)

            # TD-Q-Learning with Epsilon-Greedy
            q_table[observation.tobytes()][ix_Actions[action]] += ALPHA * (
                        reward + GAMMA * q_table[observation.tobytes()][action_]
                        - q_table[observation.tobytes()][ix_Actions[action]])

            observation = observation_

            if i == numGames - 1:
                print(action)
                env.render()

        # Epsilon decreases lineary during training
        if 1. - i / (numGames-200) > 0:
            EPS -= 1. / (numGames-200)
        else:
            EPS = 0
        totalRewards[i] = epReward

    plt.figure(figsize=(50, 30))
    plt.plot(totalRewards)

    plt.show()