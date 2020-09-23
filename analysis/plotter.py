import pandas as pd
import seaborn as sns;

sns.set()
import matplotlib.pyplot as plt
import logging
import os

sns.set(style="whitegrid")
# sns.set(font_scale=1, rc={'text.usetex' : True})
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 11})
plt.rcParams.update({'text.color': "black",
                     'axes.labelcolor': "black"})
plt.tight_layout()


class Plotter(object):
    def __init__(self, path, it, rewardPlot=True, stateExpPlot=True, stepPlot=True, epsHistory=True, algorithm=None,
                 smoothing_window=None, plot_standard_dev=True, show_plot=False, show_title=True):
        logging.getLogger(__name__).info("Initialise Plotting Unit")
        self.rewardPlot = rewardPlot
        self.stateExpPlot = stateExpPlot
        self.stepPlot = stepPlot
        self.epsHistory = epsHistory
        self.it = it
        # Boolean if the Plots should be shown or closed in execution
        self.show_plot = show_plot
        # Boolean if Title should be plotted ("False" for publication)
        self.show_title = show_title

        if smoothing_window is None:
            self.smoothing_window = max(int(it / 100), 1)
        else:
            self.smoothing_window = smoothing_window

        self.plot_standard_dev = plot_standard_dev

        self.path = path + '_Plots\\'
        logging.getLogger(__name__).info("Setting path for plots to:" + path)

        os.makedirs(self.path, exist_ok=True)
        self.algorithm = algorithm

    # Parent method for Training Plots
    def plot(self, rewards, states, steps, eps_history):
        if self.rewardPlot is True and rewards is not None:
            self.plotRewardPlot(rewards)
            try:
                self.plotRewardPlot(rewards)
            except:
                logging.getLogger(__name__).warning('Could not plot Reward development')
        if self.stateExpPlot is True and states is not None:
            try:
                self.plotStateExp(states)
            except:
                logging.getLogger(__name__).warning('Could not plot state expansion')
        if self.stepPlot is True and steps is not None:
            try:
                self.plot_cargo_units_loaded(steps)
            except:
                logging.getLogger(__name__).warning('Could not plot "Cargo Units Loaded"')
        if self.epsHistory is True and eps_history is not None:
            try:
                self.plotEPSHistory(eps_history)
            except:
                logging.getLogger(__name__).warning('Could not plot Epsilon-history')

    def plotRewardPlot(self, total_rewards):
        logging.getLogger(__name__).info("prepare reward plot...")
        fig2 = plt.figure(figsize=(5.9, 3.8))

        rewards_smoothed = pd.Series(total_rewards).rolling(self.smoothing_window,
                                                            min_periods=self.smoothing_window)
        mean = rewards_smoothed.mean()
        std = rewards_smoothed.std()

        if self.plot_standard_dev:
            rewards_q75 = mean + std
            rewards_q25 = mean - std
        ax = sns.lineplot(data=mean, linewidth=2, dashes=False)

        if self.plot_standard_dev:
            plt.fill_between(mean.index, mean, rewards_q75, color='gray', alpha=0.2)
            plt.fill_between(mean.index, mean, rewards_q25, color='gray', alpha=0.2)

        plt.xlabel("Episode")
        plt.ylabel("episode reward")
        if self.plot_standard_dev:
            ax.legend(["episode reward",
                       "Std. of smoothing window"],
                      loc='lower right')
        else:
            ax.legend(["episode reward)".format(self.smoothing_window)],
                      loc='lower right')

        xlabels = ['{:,.1f}'.format(x) + 'K' for x in ax.get_xticks() / 1000]
        ax.set_xticklabels(xlabels)

        if self.algorithm is not None:
            if self.show_title:
                plt.title(self.algorithm + ": Rewards over time\n(smoothed over {} it.)".format(self.smoothing_window))
            plt.savefig(self.path + self.algorithm + '_Rewards.pdf', dpi=600, bbox_inches="tight")
        else:
            if self.show_title:
                plt.title("Rewards over time\n(smoothed over {} it.)".format(self.smoothing_window))
            plt.savefig(self.path + 'Rewards.pdf', dpi=600, bbox_inches="tight")
        logging.getLogger(__name__).info("finished plot")
        plt.show() if self.show_plot else plt.close()

    # Plot StateExpantion
    def plotStateExp(self, state_expansion):
        logging.getLogger(__name__).info("Prepare state Expansion plot...")
        fi3 = plt.figure(figsize=(5.9, 3.5))
        ax = sns.lineplot(data=pd.Series(state_expansion), linewidth=2.5, dashes=False, color="black")
        plt.xlabel("Episode")
        plt.ylabel("States Explored")
        xlabels = ['{:,.1f}'.format(x) + 'K' for x in ax.get_xticks() / 1000]
        ax.set_xticklabels(xlabels)

        ylabels = ['{:,.1f}'.format(y) + 'K' for y in ax.get_yticks() / 1000]
        ax.set_yticklabels(ylabels)

        if self.algorithm is not None:
            if self.show_title:
                plt.title(self.algorithm + ": State Expansion over time")
            fi3.savefig(self.path + self.algorithm + '_StateExpansion.pdf', dpi=600, bbox_inches="tight")

        else:
            if self.show_title:
                plt.title("State Expansion over time")
            fi3.savefig(self.path + '_StateExpansion.pdf', dpi=600, bbox_inches="tight")
        # plt.show()

        logging.getLogger(__name__).info("Finished plot")
        plt.show() if self.show_plot else plt.close()

    # Plot smoothed Steps to Exit
    def plot_cargo_units_loaded(self, units_loaded):
        logging.getLogger(__name__).info("Prepare step to finish plot...")
        steps_smoothed = pd.Series(units_loaded).rolling(int(self.smoothing_window),
                                                         min_periods=int(self.smoothing_window)).mean()
        fi4 = plt.figure(figsize=(5.9, 3.5))
        ax = sns.lineplot(data=steps_smoothed, linewidth=2.5, dashes=False, color="green")
        plt.xlabel("Episode")
        plt.ylabel("Cargo Units")
        if self.algorithm is not None:
            if self.show_title:
                plt.title(self.algorithm + ": Cargo Units loaded over Time\n(Smoothed over {} it.)".format(
                int(self.smoothing_window / 2)))
            fi4.savefig(self.path + self.algorithm + '_CargoUnitsLoaded.pdf', dpi=600, bbox_inches="tight")
        else:
            if self.show_title:
                plt.title(" Cargo Units loaded over Time\n(smoothed over{} it.)".format(int(self.smoothing_window)))
            fi4.savefig(self.path + '_CargoUnitsLoaded.pdf', dpi=600, bbox_inches="tight")
        logging.getLogger(__name__).info("Finished plot")
        plt.show() if self.show_plot else plt.close()

    def plotEPSHistory(self, eps_history):
        logging.getLogger(__name__).info("Prepare state Epsilon plot...")
        fi3 = plt.figure(figsize=(5.9, 3.5))
        ax = sns.lineplot(data=pd.Series(eps_history), linewidth=2.5, dashes=False, color="blue")
        plt.xlabel("Episode")
        plt.ylabel("Epsilon development")
        if self.algorithm is not None:
            if self.show_title:
                plt.title(self.algorithm + ": Epsilon Development per episode")
            fi3.savefig(self.path + self.algorithm + '_EPSDevelopment.pdf', dpi=600, bbox_inches="tight")
        else:
            if self.show_title:
                plt.title("Epsilon Development per episode")
            fi3.savefig(self.path + 'EPSDevelopment.pdf', dpi=600, bbox_inches="tight")
        logging.getLogger(__name__).info("finished plot")
        plt.show() if self.show_plot else plt.close()
