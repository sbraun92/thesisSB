import numpy as np
from evaluator.evaluator import Evaluator
from env.roroDeck import RoRoDeck
from algorithms.analyser import Analysor
import pandas as pd
import tqdm




def calculateRewardsystems():

    reward_simpleLoading = np.arange(0, 1, 2)
    reward_shifts = np.arange(-10, -2,1)
    reward_terminalSpace = np.arange(-10, -2, 1)
    reward_terminalMand = np.arange(-40, -10,2)

    systems = np.array(np.meshgrid(reward_simpleLoading,reward_shifts, reward_terminalSpace, reward_terminalMand)).T.reshape(-1, 4)

    return systems


if __name__ == '__main__':
    np.random.seed(0)
    systems = calculateRewardsystems()
    df = pd.DataFrame()
    print(len(systems))
    allobservations = []

    for systemid, system in tqdm.tqdm(enumerate(systems)):
        numberOfStowagePlans = 8
        cumRewards = np.zeros(numberOfStowagePlans)
        evaluations = []

        env = RoRoDeck(True)
        env.rewardSystem = system
        evaluator = Evaluator(env.vehicleData,env.grid)

        unique_CumRrewards = set()
        unique_eval = set()
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
            if cumReward not in unique_CumRrewards and evaluation not in unique_eval:
                evaluations += [(cumReward,evaluation)]


            unique_eval.add(evaluation)
            unique_CumRrewards.add(cumReward)

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

        invNo,degree = analyser.calculateInversionNumber(rewardSystemEval)

        obs = np.append(system,np.array(degree))

        allobservations += [obs]

    df = pd.DataFrame(np.array(allobservations))
    print(df)
    print(df[4].idxmax())
    print(df.iloc[df[4].idxmax()])
    print(df[df[4]==1])

