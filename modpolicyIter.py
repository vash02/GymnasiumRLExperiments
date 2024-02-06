import numpy as np
import matplotlib.pyplot as plt

class modPolicyIter:
    def __init__(self, env, eval_policy = False, m=3, c=0.001, gamma = 0.999):
        self.env = env
        self.nS = env.observation_space.n
        self.nA = env.action_space.n
        self.c = c
        self.V = np.zeros(self.nS)
        self.policy = np.zeros(self.nS)
        self.n = 0
        self.m = m
        self.gamma = gamma
        self.eval_policy = eval_policy
        self.sab = 0
        self.mean_rewards = []
        self.sabl = []
        self.iterations = []

    def modified_policy_iteration(self):
        while True:
            k = 0
            while k < self.m:
                k += 1
                self.V = self.bellman_operator(self.policy, self.V)
            self.n += 1
            V_n = np.copy(self.V)
            self.policy = self.greedy_policy(V_n)
            U_0 = self.bellman_operator(self.policy, V_n)

            # For empirical policy evaluation
            if self.eval_policy:
                self.evaluate_policy(self.policy, self.n, self.sab)

            if np.max(np.abs(U_0 - V_n)) <= self.c:
                break
        return self.policy

    def bellman_operator(self, policy, V):
        U_new = np.zeros(self.nS)

        for state in range(self.nS):
            action = policy[state]
            U_new[state] = sum([p * (r + self.gamma * V[sp])
                               for p, sp, r, _ in self.env.unwrapped.P[state][action]])
            self.sab += 1

        return U_new

    def greedy_policy(self, V):
        policy = np.zeros(self.nS, dtype=int)

        for state in range(self.nS):
            action_values = [sum([p * (r + self.gamma * V[sp])
                                  for p, sp, r, _ in self.env.unwrapped.P[state][action]])
                             for action in range(self.nA)]
            self.sab += 1 * self.nA
            policy[state] = np.argmax(action_values)

        return policy

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

        mean_reward = np.mean(total_rewards)
        self.mean_rewards.append(mean_reward)
        self.iterations.append(iteration)
        self.sabl.append(sab)
        print(f"Iteration {iteration}: Mean Total Reward: {mean_reward}")
        print(f"After Iteration {iteration}: Total sab: {sab}")

    def create_plots(self):
        plt.subplot(2, 1, 1)
        plt.plot(self.iterations, self.mean_rewards)
        plt.title('Mean Reward vs Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Quality')

        # Plot SAB (Maximum Absolute Bellman Error)
        plt.subplot(2, 1, 2)
        plt.plot(self.iterations, self.sabl, color='orange')
        plt.title('SAB vs Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Efforts')

        plt.tight_layout()
        plt.show()
