import numpy as np
from evaluator.evaluator import Evaluator
from env.RoRoDeck import RoRoDeck
from algorithms.analyser import Analysor
import pandas as pd
import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
rewardSystem1 = np.array([2,  # simple Loading
                        -6,  # caused shifts
                        -10,  # terminal: Space left unsed
                        -20])  # terminal: mand. cargo not loaded
rewardSystem2 = np.array([2,  # simple Loading
                        -6,  # caused shifts
                        -6,  # terminal: Space left unsed
                        -20])  # terminal: mand. cargo not loaded

rewardSystem3 = np.array([0,  # simple Loading
                        -6,  # caused shifts
                        -6,  # terminal: Space left unsed
                        -20])  # terminal: mand. cargo not loaded


def calculateRewardsystems():


    reward_shifts = np.arange(-10, 0)
    reward_simpleLoading = np.arange(0, 4, 2)
    reward_terminalSpace = np.arange(-20, -5,3)
    reward_terminalMand = np.arange(-20, -10,2)

    systems = np.array(np.meshgrid(reward_shifts, reward_simpleLoading, reward_terminalSpace, reward_terminalMand)).T.reshape(-1, 4)

    return systems


if __name__ == '__main__':
    np.random.seed(0)
    systems = calculateRewardsystems()
    df = pd.DataFrame()
    print(len(systems))
    for systemid, system in tqdm.tqdm(enumerate(systems)):
        numberOfStowagePlans = 15
        cumRewards = np.zeros(numberOfStowagePlans)
        evaluations = []

        env = RoRoDeck(True)
        env.rewardSystem = system
        evaluator = Evaluator(env.vehicleData,env.grid)
        for i in range(numberOfStowagePlans):
            cumReward = 0
            env.reset()
            done = False

            while not done:
                action = env.actionSpaceSample()
                observation_, reward, done, info = env.step(action)
                cumReward += reward

            stowagePlan = env.getStowagePlan()
            evaluation = evaluator.evaluate(stowagePlan)

            cumRewards[i] = cumReward
            evaluations += [(cumReward,evaluation)]

        #Sort to Cumulative Reward
        evaluations.sort(key=lambda tup:tup[0])

        #Mark this Sequence
        sortedCumRewards = []
        for ix,a in enumerate(evaluations):
            sortedCumRewards += [(ix,a[0],a[1])]

        #Sort to true Evaluation
        #An ideal reward system will have a perfectly sorted marked sequence
        sortedCumRewards.sort(key=lambda tup: tup[2])

        rewardSystemEval = np.array([i[0] for i in sortedCumRewards])

        analyser = Analysor()
        #print("RewardSystem " +str(systemid+1))
        #print(system)
        obs = np.append(system,np.array(analyser.calculateInversionNumber(rewardSystemEval)))
        #print(len(rewardSystemEval))
        #print(pd.DataFrame(obs).T)
        df = df.append(pd.DataFrame(obs).T)
    #print(min(df[4]))
    df.plot()
    plt.show()
