import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import logging

sns.set(style="darkgrid")

class Plotter(object):
    def __init__(self, path, it, rewardPlot = True, stateExpPlot = True, stepPlot=True, epsHistory=True):
        logging.getLogger('log1').info("Initialise Plotting Unit")
        self.rewardPlot = rewardPlot
        self.stateExpPlot = stateExpPlot
        self.stepPlot = stepPlot
        self.epsHistory = epsHistory
        self.it = it
        self.smoothingWindow = int(it/200)
        #self.smoothingWindow = 200
        #if self.smoothingWindow == 0:
        #    self.smoothingWindow = 1
        logging.getLogger('log1').info("Setting path for plots to:"+ path)
        self.path = path

    def plot(self, rewards, states, steps, eps_history):
        if self.rewardPlot == True:
            self.plotRewardPlot(rewards)
        if self.stateExpPlot is True:
            self.plotStateExp(states)
        if self.stepPlot is True:
            self.plotStepPlot(steps)
        if self.epsHistory is True:
            self.plotEPSHistory(eps_history)

    def plotRewardPlot(self,totalRewards):
        logging.getLogger('log1').info("prepare reward plot...")
        fig2 = plt.figure(figsize=(10, 5))
        rewards_smoothed = pd.Series(totalRewards).rolling(self.smoothingWindow, min_periods=self.smoothingWindow).mean()
        rewards_q75 = pd.Series(totalRewards).rolling(self.smoothingWindow, min_periods=self.smoothingWindow).quantile(0.75)
        rewards_q25 = pd.Series(totalRewards).rolling(self.smoothingWindow, min_periods=self.smoothingWindow).quantile(0.25)
        rewardsDf = pd.DataFrame(totalRewards)
        ax = sns.lineplot(data=rewards_smoothed, linewidth=2.5, dashes=False)

        #ax = sns.lineplot(data=rewards_smoothed+rewards_std, linewidth=2.5, dashes=False,ax=ax)

        plt.fill_between(rewards_smoothed.index,rewards_smoothed,  rewards_q75, color='gray', alpha=0.2)
        plt.fill_between(rewards_smoothed.index, rewards_smoothed, rewards_q25, color='gray',alpha=0.2)
        #plt.savefig(self.path + '_Rewards.png')
        #plt.show()
        #plt.plot(rewards_smoothed)
        plt.xlabel("Episode")
        plt.ylabel("Episode Reward (Smoothed)")
        plt.title("Episode Reward over Time (Smoothed over window size {} and IQR)".format(self.smoothingWindow))
        plt.savefig(self.path + '_Rewards.png')
        #plt.close(fig2)
        logging.getLogger('log1').info("finished plot")

    # Plot StateExpantion
    def plotStateExp(self, stateExpantion):
        logging.getLogger('log1').info("prepare state Expansion plot...")
        fi3 = plt.figure(figsize=(10, 5))
        ax = sns.lineplot(data=pd.Series(stateExpantion), linewidth=2.5, dashes=False, color="black")
        plt.xlabel("Episode")
        plt.ylabel("States Explored")
        plt.title("State Expansion over time")
        # plt.show()
        fi3.savefig(self.path + '_StateExpansion.png')
        logging.getLogger('log1').info("finished plot")

    # Plot smoothed Steps to Exit
    def plotStepPlot(self,stepsToExit):
        logging.getLogger('log1').info("prepare step to finish plot...")
        steps_smoothed = pd.Series(stepsToExit).rolling(int(self.smoothingWindow / 2+1),
                                                        min_periods=int(self.smoothingWindow / 2+1)).mean()
        fi4 = plt.figure(figsize=(10, 5))
        ax = sns.lineplot(data=steps_smoothed, linewidth=2.5, dashes=False, color="green")
        plt.xlabel("Episode")
        plt.ylabel("Steps to Finish (Smoothed)")
        plt.title("Steps to Finish over Time (Smoothed over window size {})".format(int(self.smoothingWindow / 2)))
        fi4.savefig(self.path + '_StepsToFinish.png')
        logging.getLogger('log1').info("finished plot")

    def plotEPSHistory(self,eps_history):
        logging.getLogger('log1').info("prepare state Epsilon plot...")
        fi3 = plt.figure(figsize=(10, 5))
        ax = sns.lineplot(data=pd.Series(eps_history), linewidth=2.5, dashes=False, color="blue")
        plt.xlabel("Episode")
        plt.ylabel("Epsilon development")
        plt.title("Epsilon Development per episode")
        # plt.show()
        fi3.savefig(self.path + '_EPSDevelopment.png')
        logging.getLogger('log1').info("finished plot")