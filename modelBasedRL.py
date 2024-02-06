import numpy as np
import matplotlib.pyplot as plt
from valueIter import valueIter

class ModelBasedRL:

    def __init__(self, env):
        self.noise_P = None
        self.env = env
        self.nS = env.observation_space.n
        self.nA = env.action_space.n
        self.mean_rewards = []
        self.std_rewards = []
        self.alphal = []


    def generate_noisy_transitions(self, alpha):
        self.noise_P = []
        for s in range(self.nS):
            for a in range(self.nA):
                s_neighbours = [sp for p, sp, _, _ in self.env.unwrapped.P[s][a] if p > 0]
                k = len(s_neighbours)
                p = np.asarray([p for p, _, _, _ in self.env.unwrapped.P[s][a]])
                q = np.random.dirichlet(np.ones(k))
                noise_probs = alpha * q + float(1 - alpha) * p
                noise_t = self.env.unwrapped.P.copy()
                # print(noise_t[s][a])
                for i, sp in enumerate(s_neighbours):
                    noise_t[s][a][i] = (noise_probs[i], sp, self.env.unwrapped.P[s][a][i][2], self.env.unwrapped.P[s][a][i][3])
            self.noise_P.append(noise_t[s])
        return self.noise_P

    def noise_prob_value_iteration(self):
        for alpha in np.arange(0.0, 0.9, 0.1):
            self.policy = np.zeros(self.nS)
            self.noise_P = self.generate_noisy_transitions(alpha)
            mbrlvi = valueIter(self.env, self.noise_P, False)
            mbrlvi.value_iteration()
            self.policy = mbrlvi.extract_policy()
            self.evaluate_policy(alpha)

    def evaluate_policy(self, alpha):
        total_rewards = []
        for _ in range(500):  # Run policy for 500 episodes
            state = self.env.reset()[0]
            episode_reward = 0
            done = False

            while not done:
                action = self.policy[int(state)]
                next_state, reward, done, _, info = self.env.step(int(action))
                episode_reward += reward
                state = next_state

            total_rewards.append(episode_reward)

        mean_r = np.mean(total_rewards)
        std_r = np.std(total_rewards)
        self.mean_rewards.append(mean_r)
        self.std_rewards.append(std_r)
        self.alphal.append(alpha)
        print(f"For alpha: {alpha}: Mean discounted rewards: {mean_r}")
        print(f"For alpha: {alpha}: Std Deviation rewards: {std_r}")


    def create_plots(self):

        plt.plot(self.alphal, self.mean_rewards, '-o', color='b', label='Mean')
        plt.errorbar(self.alphal, self.mean_rewards, yerr=self.std_rewards, ecolor='orange', label='Standard deviation',capsize=3)

        plt.title('Mean and Standard Deviation of Expected Discounted Rewards')
        plt.xlabel('Alpha')
        plt.ylabel('Mean & Std Discounted Reward')
        plt.legend()  # Show legend with labels
        plt.show()



