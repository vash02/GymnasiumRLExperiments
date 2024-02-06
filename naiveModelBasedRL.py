import time
import numpy as np
import matplotlib.pyplot as plt
from valueIter import valueIter

#Used ChatGPT for figuring out a BUG in this code

class NaiveModelBasedRL:
    def __init__(self, env):
        self.env = env
        self.nS = env.observation_space.n
        self.nA = env.action_space.n
        self.estP = {}
        for s in range(self.nS):
            self.estP[s] = {}
            for a in range(self.nA):
                self.estP[s][a] = []
        self.policy = np.zeros(self.nS)
        self.mean_rewards = []
        self.std_rewards = []
        self.episodes = []


    def naive_iteration(self,N):
        self.initP = {}
        for s in range(self.nS):
            self.initP[s] = {}
            for a in range(self.nA):
                self.initP[s][a] = []
        init_state, info = self.env.reset()
        terminated=False
        truncated=False
        for _i in range(N):
            action = self.env.action_space.sample()
            observation, reward, terminated, truncated, info = self.env.step(action)
            self.initP[init_state][action].append((observation, reward, terminated))
            init_state = observation

            if (terminated or truncated):
                init_state, info = self.env.reset()
                # time.sleep(1)

    def update_rewards_and_prob(self):
        for s in range(self.nS):
            for a in range(self.nA):
                updated_entries = []
                next_state_rewards = {}
                total_entries = len(self.initP[s][a])

                if total_entries == 0:
                    updated_entries.append((1.0, s, 0.0, True))
                    updated_entries.extend([(0.0, sp, 0.0, False) for _, sp, _, _ in self.env.unwrapped.P[s][a] if sp != s])
                    self.estP[s][a] = updated_entries
                    continue
                for entry in self.initP[s][a]:
                    next_state = entry[0]
                    reward = entry[1]

                    if next_state not in next_state_rewards:
                        next_state_rewards[next_state] = {'rewards': [], 'count': 0}
                    next_state_rewards[next_state]['rewards'].append(reward)
                    next_state_rewards[next_state]['count'] += 1

                # Calculate mean of rewards and probability for each next_state
                all_possible_next_states_count_sum = sum(next_state_rewards[sp]['count'] for sp, _, _ in self.initP[s][a])
                for sp, _, _ in self.initP[s][a]:
                    # if sp not in next_state_rewards.keys() and sp != s:
                    #     updated_entry = (0.0, sp, 0, False)
                    # elif sp not in next_state_rewards.keys() and sp == s:
                    #     updated_entry = (1.0, sp, 0, True)
                    # else:
                    rewards_info = next_state_rewards[sp]
                    mean_reward = np.mean(rewards_info['rewards'])
                    probability = rewards_info['count'] / all_possible_next_states_count_sum
                    updated_entry = (probability, sp, mean_reward, False)
                    updated_entries.append(updated_entry)
                self.estP[s][a] = updated_entries


    def evaluate_policy(self, N):
        total_rewards = []
        for _ in range(500):
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
        self.episodes.append(N)
        print(f"For {N} episodes: Mean discounted rewards: {mean_r}")
        print(f"For {N} episodes: Std Deviation rewards: {std_r}")

    def naive_vi_with_est_prob(self):
        for N in np.arange(2500, 51000, 2500):
            self.naive_iteration(N)
            self.update_rewards_and_prob()
            nviesp = valueIter(self.env, self.estP, False)
            nviesp.value_iteration()
            self.policy = nviesp.extract_policy()
            self.evaluate_policy(N)
            self.env.close()

    def create_plots(self):

        plt.plot(self.episodes, self.mean_rewards, '-o', color='b', label='Mean')
        plt.errorbar(self.episodes, self.mean_rewards, yerr=self.std_rewards, ecolor='orange', label='Standard deviation',capsize=3)
        plt.title('Mean and Standard Deviation of Discounted Rewards')
        plt.xlabel('Number of Episodes')
        plt.ylabel('Mean & Std Discounted Reward')
        plt.legend()
        plt.show()


