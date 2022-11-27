import stable_baselines3 as sb3
from stable_baselines3 import PPO
import gym
import simglucose
import wandb
from wandb.integration.sb3 import WandbCallback
from gym.envs.registration import register
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

SAVE_PATH = "/home/berk/VS_Project/simglucose/sb_test_2.zip"

def custom_reward(BG_last_hour):
    if BG_last_hour[-1] > 180:
        return -0.5
    elif BG_last_hour[-1] < 70:
        return -1
    else:
        return 0.5

register(
    id='simglucose-adult-v1',
    entry_point='simglucose.envs:T1DSimEnv',
    kwargs={'patient_name': 'adult#001',
            'reward_fun': custom_reward}
)

env = gym.make('simglucose-adult-v1')

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": int(1e+6),
    "env_name": "PPO-simBG-v1",
}

# run = wandb.init(
#     project="SB3_SimBG",
#     name="sb_ppo_v1",
#     config=config,
#     sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
#     save_code=True,  # optional
# )

model = PPO("MlpPolicy", env, verbose=1)
#model.learn(total_timesteps=int(1e+6))
#model.save(SAVE_PATH)

# artifact = wandb.Artifact('model', type='dataset')
# artifact.add_dir('./sb_test_2')
# wandb.log_artifact(artifact)

# run.finish()


test=1
if test==1:
    model = PPO.load(SAVE_PATH)
    obs = env.reset()
    for i in range(1440):
        env.render(mode="human")
        action, _state = model.predict(obs)
        print(action)
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()