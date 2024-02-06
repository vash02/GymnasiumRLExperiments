import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

from valueIter import valueIter
from policyIter import policyIter
from  modpolicyIter import modPolicyIter
from modelBasedRL import ModelBasedRL
from naiveModelBasedRL import NaiveModelBasedRL

# B659 RL 20224 - gymnasium setup for assignment 1


def prepFrozen(render_mode=None):
    env = gym.make('FrozenLake-v1',desc=generate_random_map(size = 8,seed = 3),render_mode=render_mode)
    dname="Frozen8"
    env._max_episode_steps = 50000
    P=env.unwrapped.P
    nA=env.action_space.n
    nS=env.observation_space.n
    _a,_b = env.reset(seed=3)
    return(nS,nA,P,env,dname)


def prepFrozenSmall(render_mode):
    env = gym.make('FrozenLake-v1',map_name='4x4',render_mode=render_mode)
    dname="Frozen4"
    env._max_episode_steps = 250
    P=env.unwrapped.P
    nA=env.action_space.n
    nS=env.observation_space.n
    _a,_b = env.reset(seed=5)
    return(nS,nA,P,env,dname)


#Task 1: Value Iteration Function

def valIterDemo():
    nS,nA,P,env,envName = prepFrozen(None)
    vi = valueIter(env, env.unwrapped.P, True)

    print("Value Iteration.....")
    vi.value_iteration()

    optimal_policy = vi.extract_policy()

    print("Optimal Policy:")
    print(optimal_policy)

    vi.create_plots()
    env.close()

#Task 1: Policy Iteration Function

def policyIterDemo():
    nS,nA,P,env,envName = prepFrozen(None)
    pi = policyIter(env, True)

    optimal_policy = pi.policy_iteration()
    print("Optimal Policy: ", optimal_policy)

    pi.create_plots()
    env.close()

#Task 1: Modiified Policy Iteration Function

def modPolicyIterDemo():
    nS,nA,P,env,envName = prepFrozen(None)
    mpi = modPolicyIter(env, True)

    optimal_policy = mpi.modified_policy_iteration()
    print("Optimal Policy: ", optimal_policy)

    mpi.create_plots()
    env.close()

#Task 2: Model Based RL with noise introduced transition probabilities VI Funtion

def modelBasedRLDemo():
    nS,nA,P,env,envName = prepFrozen(None)
    mbrl = ModelBasedRL(env)
    mbrl.noise_prob_value_iteration()
    mbrl.create_plots()
    env.close()

#Task 3: Naive Model Based RL VI Function

def naiveModelBasedRLDemo():
    nS,nA,P,env,envName = prepFrozen(None)
    nmbrl = NaiveModelBasedRL(env)
    nmbrl.naive_vi_with_est_prob()
    nmbrl.create_plots()


# CALLING POLICY COMPUTATION FUNCTIONS FOR ALL TASKS

# valIterDemo()
# policyIterDemo()
# modPolicyIterDemo()
# modelBasedRLDemo()
# naiveModelBasedRLDemo()



