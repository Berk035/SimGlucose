import gym
import numpy as np
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C
from gym.envs.registration import register
from stable_baselines.common.evaluation import evaluate_policy
import pickle as pkl

MODE = 1 # 0: test, 1: train
demo_timesteps = 1000
trajectories = 0
save_path = '/home/berk/PycharmProjects/simglucose/examples/results/trajectory.pickle'
memory = {'states': [], 'actions': [], 'rewards': [], 'dones': []}

try:
    with open(save_path, 'rb') as handle:
        memory = pkl.load(handle)
except:
    print('No saved trajectories found')
    pass


def custom_reward(BG_last_hour):
    BG = BG_last_hour[-1]
    upper_bound = 180
    lower_bound = 70
    insulin = np.arange(60.0, 190.0, .5)
    target = 120
    g = 0.7
    
    rew = 1.0 - np.tanh(np.abs((BG - target) / g) * 0.1) ** 2
    
    if BG > upper_bound:
        rew=-1
    elif BG < lower_bound:
        rew=-2
    else:
        rew

    return rew


register(
    id='simglucose-adult-v0',
    entry_point='simglucose.envs:T1DSimEnv',
    kwargs={'patient_name': 'adult#001',
            'reward_fun': custom_reward}
)

env = gym.make('simglucose-adult-v0')
model = A2C(MlpPolicy, env, verbose=1)

if MODE==1:
    model.learn(total_timesteps=int(1e7))
    model.save("/home/berk/PycharmProjects/simglucose/examples/results/a2c_adult")
elif MODE==0:
    model.load("/home/berk/PycharmProjects/simglucose/examples/results/a2c_adult")
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(f"Mean reward: {mean_reward}, std: {std_reward}")

if MODE is not 1:
    obs_record = []
    rew_record = []
    action_record = []

    # Enjoy trained agent
    obs = env.reset()
    for i in range(demo_timesteps):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)

        obs_record.append(obs)
        rew_record.append(rewards)
        action_record.append(action)

        memory['states'].append(obs)
        memory['actions'].append(action)
        memory['rewards'].append(rewards)
        memory['dones'].append(dones)

        if dones:
            obs = env.reset()
            print(f"Episode finished after {i+1} timesteps")
            trajectories += 1
            

        env.render(mode='human')
    env.close()

    print('trajectories:', trajectories)
    print('states collected:', len(memory['states']))

    with open(save_path, 'wb') as handle:
        f = pkl.dump(memory, handle, protocol=pkl.HIGHEST_PROTOCOL)
        f.close()
        print("trajectories saved to", save_path)

