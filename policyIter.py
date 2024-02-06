import numpy as np
import matplotlib.pyplot as plt

#Referred to Medium article for explainatiton and pseudo code of the algorithm
# https://towardsdatascience.com/implement-policy-iteration-in-python-a-minimal-working-example-6bf6cc156ca9


class policyIter:
    def __init__(self, env, eval_policy = False, gamma = 0.999, c = 0.001):
        self.nS = env.observation_space.n
        self.nA = env.action_space.n
        self.P = env.unwrapped.P
        self.env = env
        self.gamma = gamma
        self.c = c
        self.policy = np.zeros(self.nS)  # Initialize with a default policy
        self.n = 0
        self.eval_policy = eval_policy
        self.sab = 0
        self.mean_rewards = []
        self.sabl = []
        self.iterations = []

    def policy_evaluation(self, policy, V):
        while self.n < 500:
            self.n +=1
            delta = 0
            for s in range(self.nS):
                v = V[s]
                a = policy[s]
                V[s] = sum([p * (r + self.gamma * V[sp]) for p, sp, r, _ in self.P[s][a]])
                self.sab += 1
                delta = max(delta, abs(v - V[s]))

            if delta < self.c:
                break

        return V

    def policy_improvement(self, V):
        policy_stable = True

        for s in range(self.nS):
            old_action = self.policy[s]
            action_values = [sum([p * (r + self.gamma * V[sp]) for p, sp, r, _ in self.P[s][a]]) for a in range(self.nA)]
            self.sab += 1 * self.nA
            self.policy[s] = np.argmax(action_values)

            if old_action != self.policy[s]:
                policy_stable = False
        return policy_stable

    def policy_iteration(self):
        i = 0
        while True:
            i += 1
            V = np.zeros(self.nS)
            V = self.policy_evaluation(self.policy, V)
            stable = self.policy_improvement(V)
            if self.eval_policy:
               self.evaluate_policy(self.policy, i, self.sab)
            if stable:
                break

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

        mean_reward = np.mean(total_rewards)
        self.mean_rewards.append(mean_reward)
        self.iterations.append(iteration)
        self.sabl.append(sab)
        print(f"Iteration {iteration}: Quality: {mean_reward}")
        print(f"After Iteration {iteration}: Total sab: {sab}")

    def create_plots(self):
        plt.subplot(2, 1, 1)
        plt.plot(self.iterations, self.mean_rewards, '-o')
        plt.title('Mean Reward vs Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Quality')

        # Plot SAB (Maximum Absolute Bellman Error)
        plt.subplot(2, 1, 2)
        plt.plot(self.iterations, self.sabl, '-o', color='orange')
        plt.title('SAB vs Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Efforts')

        plt.tight_layout()
        plt.show()
