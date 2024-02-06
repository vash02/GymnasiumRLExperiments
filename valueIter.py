import numpy as np
import matplotlib.pyplot as plt

#Took ChatGPT's help for resolving a bug

class valueIter:
    def __init__(self, env, P, eval_policy = False, gamma=0.999, c = 0.001):
        self.env = env
        self.nS = env.observation_space.n
        self.nA = env.action_space.n
        self.P = P
        self.gamma = gamma
        self.c = c
        self.values = np.zeros(self.nS)
        self.policy = np.zeros(self.nS)
        self.eval_policy = eval_policy
        self.n = 0
        self.sab = 0
        self.mean_rewards = []
        self.sabl = []
        self.iterations = []

    def value_iteration(self):
        while self.n < 500:
            self.n += 1
            delta = 0
            for s in range(self.nS):
                v = self.values[s]
                self.values[s] = max([sum([p * (r + self.gamma * self.values[sp]) for p, sp, r, _ in self.P[s][a]]) for a in range(self.nA)])
                self.sab += 1 * self.nA
                delta = max(delta, abs(v - self.values[s]))

            if self.eval_policy and (self.n % 10 == 0):
                self.extract_policy()

            if delta < self.c:
                break

    def extract_policy(self):
        self.policy = np.zeros(self.nS)
        for s in range(self.nS):
            values = [sum([p * (r + self.gamma * self.values[sp]) for p, sp, r, _ in self.P[s][a]]) for a in range(self.nA)]
            self.policy[s] = np.argmax(values)
        if self.eval_policy:
            self.evaluate_policy(self.policy, self.n, self.sab)
        return self.policy

    def evaluate_policy(self, policy, iteration, sab):
        total_rewards = []
        for _ in range(500):  # Run policy for 500 episodes
            state = self.env.reset()[0]
            episode_reward = 0
            done = False

            while not done:
                action = policy[int(state)]
                next_state, reward, done, _, info = self.env.step(int(action))
                episode_reward += reward
                state = next_state

            total_rewards.append(episode_reward)

        mean_r = np.mean(total_rewards)
        self.iterations.append(iteration)
        self.mean_rewards.append(mean_r)
        self.sabl.append(sab)
        print(self.iterations)
        print(self.mean_rewards)
        print(f"Iteration {iteration}: Quality: {mean_r}")
        print(f"After Iteration {iteration}: Total sab: {sab}")

    def create_plots(self):
        plt.subplot(2, 1, 1)
        plt.plot(self.iterations, self.mean_rewards,'-o')
        plt.title('Mean Reward vs Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Quality')

        # Plot SAB (Maximum Absolute Bellman Error)
        plt.subplot(2, 1, 2)
        plt.plot(self.iterations, self.sabl,'-o', color='orange')
        plt.title('SAB vs Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Efforts')

        plt.tight_layout()
        plt.show()

