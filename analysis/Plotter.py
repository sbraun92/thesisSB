import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import logging

sns.set(style="whitegrid")
#sns.set(font_scale=1, rc={'text.usetex' : True})
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'text.color' : "black",
                     'axes.labelcolor' : "black"})
plt.tight_layout()



class Plotter(object):
    def __init__(self, path, it, rewardPlot = True, stateExpPlot = True, stepPlot=True, epsHistory=True, algorithm=None, smoothing_window = None):
        logging.getLogger('log1').info("Initialise Plotting Unit")
        self.rewardPlot = rewardPlot
        self.stateExpPlot = stateExpPlot
        self.stepPlot = stepPlot
        self.epsHistory = epsHistory
        self.it = it
        if smoothing_window is None:
            self.smoothing_window = max(int(it / 100), 1)
        else:
            self.smoothing_window = smoothing_window
        #self.smoothingWindow = 200
        #if self.smoothingWindow == 0:
        #    self.smoothingWindow = 1
        logging.getLogger('log1').info("Setting path for plots to:"+ path)
        self.path = path
        self.algorithm = algorithm


    def plot(self, rewards, states, steps, eps_history):
        if self.rewardPlot is True:
            try:
                self.plotRewardPlot(rewards)
            except:
                logging.getLogger(__name__).warning('Could not plot Reward development')
        if self.stateExpPlot is True:
            try:
                self.plotStateExp(states)
            except:
                logging.getLogger(__name__).warning('Could not plot state expansion')
        if self.stepPlot is True:
            try:
                self.plotStepPlot(steps)
            except:
                logging.getLogger(__name__).warning('Could not plot "steps to exit"')
        if self.epsHistory is True:
            try:
                self.plotEPSHistory(eps_history)
            except:
                logging.getLogger(__name__).warning('Could not plot Epsilon-history')

    def plotRewardPlot(self, totalRewards):
        logging.getLogger('log1').info("prepare reward plot...")
        fig2 = plt.figure(figsize=(5.9, 3.8))
        rewards_smoothed = pd.Series(totalRewards).rolling(self.smoothing_window, min_periods=self.smoothing_window).mean()
        #rewards_smoothed_high_res = pd.Series(totalRewards).rolling(self.smoothingWindow,
        #                                                   min_periods=self.smoothingWindow/100).mean()
        rewards_q75 = rewards_smoothed+pd.Series(totalRewards).rolling(self.smoothing_window, min_periods=self.smoothing_window).std()
        rewards_q25 = rewards_smoothed-pd.Series(totalRewards).rolling(self.smoothing_window, min_periods=self.smoothing_window).std()
        rewardsDf = pd.DataFrame(totalRewards)
        ax = sns.lineplot(data=rewards_smoothed, linewidth=2, dashes=False)

        #ax = sns.lineplot(data=rewards_smoothed+rewards_std, linewidth=2.5, dashes=False,ax=ax)

        plt.fill_between(rewards_smoothed.index, rewards_smoothed,  rewards_q75, color='gray', alpha=0.2)
        plt.fill_between(rewards_smoothed.index, rewards_smoothed, rewards_q25, color='gray', alpha=0.2)

        #plt.fill_between(rewards_smoothed.index, rewards_smoothed, totalRewards, color='gray', alpha=0.2)

        #plt.savefig(self.path + '_Rewards.png')
        #plt.show()
        #plt.plot(rewards_smoothed)
        plt.xlabel("Episode")
        plt.ylabel("episode reward")
        ax.legend(["episode reward (smoothed over window size {})".format(self.smoothing_window),
                   "standard deviation of last {} episodes".format(self.smoothing_window)],
                  loc='lower right')

        xlabels = ['{:,.1f}'.format(x) + 'K' for x in ax.get_xticks() / 1000]
        ax.set_xticklabels(xlabels)

        if self.algorithm is not None:
            plt.title(self.algorithm+ ": Rewards over time")
            plt.savefig(self.path + "_" + self.algorithm + '_Rewards.pdf', dpi=600, bbox_inches="tight")
        else:
            plt.savefig(self.path + '_Rewards.pdf',dpi=600, bbox_inches = "tight")
        #plt.close(fig2)
        logging.getLogger('log1').info("finished plot")

    # Plot StateExpantion
    def plotStateExp(self, stateExpantion):
        logging.getLogger('log1').info("prepare state Expansion plot...")
        fi3 = plt.figure(figsize=(5.9, 3.5))
        ax = sns.lineplot(data=pd.Series(stateExpantion), linewidth=2.5, dashes=False, color="black")
        plt.xlabel("Episode")
        plt.ylabel("States Explored")
        xlabels = ['{:,.1f}'.format(x) + 'K' for x in ax.get_xticks() / 1000]
        ax.set_xticklabels(xlabels)

        ylabels = ['{:,.1f}'.format(y) + 'K' for y in ax.get_yticks() / 1000]
        ax.set_yticklabels(ylabels)

        if self.algorithm is not None:
            plt.title(self.algorithm+ ": State Expansion over time")
            fi3.savefig(self.path +"_" + self.algorithm +  '_StateExpansion.pdf', dpi=600, bbox_inches="tight")
        
        else:
            fi3.savefig(self.path + '_StateExpansion.pdf', dpi=600, bbox_inches="tight")
            plt.title("State Expansion over time")
        # plt.show()



        logging.getLogger('log1').info("finished plot")

    # Plot smoothed Steps to Exit
    def plotStepPlot(self,stepsToExit):
        logging.getLogger('log1').info("prepare step to finish plot...")
        steps_smoothed = pd.Series(stepsToExit).rolling(int(self.smoothing_window / 2 + 1),
                                                        min_periods=int(self.smoothing_window / 2 + 1)).mean()
        fi4 = plt.figure(figsize=(5.9, 3.5))
        ax = sns.lineplot(data=steps_smoothed, linewidth=2.5, dashes=False, color="green")
        plt.xlabel("Episode")
        plt.ylabel("Steps to Finish (Smoothed)")
        if self.algorithm is not None:
            plt.title(self.algorithm + ": Steps to Finish over Time (Smoothed over window size {})".format(int(self.smoothing_window / 2)))
            fi4.savefig(self.path +"_"+ self.algorithm+'_StepsToFinish.pdf', dpi=600, bbox_inches="tight")
        else:
            plt.title("Steps to Finish over Time (Smoothed over window size {})".format(int(self.smoothing_window / 2)))
            fi4.savefig(self.path + '_StepsToFinish.pdf',dpi=600, bbox_inches = "tight")
        logging.getLogger('log1').info("finished plot")

    def plotEPSHistory(self,eps_history):
        logging.getLogger('log1').info("prepare state Epsilon plot...")
        fi3 = plt.figure(figsize=(5.9, 3.5))
        ax = sns.lineplot(data=pd.Series(eps_history), linewidth=2.5, dashes=False, color="blue")
        plt.xlabel("Episode")
        plt.ylabel("Epsilon development")
        if self.algorithm is not None:
            plt.title(self.algorithm+ ": Epsilon Development per episode")
            fi3.savefig(self.path +"_"+ self.algorithm+'_EPSDevelopment.pdf', dpi=600, bbox_inches="tight")
        else:
            plt.title("Epsilon Development per episode")
            fi3.savefig(self.path + '_EPSDevelopment.pdf',dpi=600, bbox_inches = "tight")
        logging.getLogger('log1').info("finished plot")