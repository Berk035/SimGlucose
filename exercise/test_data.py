import pandas as pd
import numpy as np

memory = {'observations': [], 'next_observations':[], 'actions':[], 'rewards':[], 'terminals':[]}

obs_record = []
next_obs_record= []
action_record= []
rew_record= []
dones= []


for idx in range(10):
    obs_record.append(idx)
    next_obs_record.append(idx+1)
    action_record.append(1)
    rew_record.append(-1)
    dones.append(False)

memory['observations'].append(np.asarray(obs_record))
memory['next_observations'].append(np.asarray(next_obs_record))
memory['actions'].append(np.asarray(action_record))
memory['rewards'].append(np.asarray(rew_record))
memory['terminals'].append(np.asarray(dones))

df = pd.DataFrame(data=memory)
df.to_csv('/home/berk/VS_Project/simglucose/examples/trajectories/deneme.csv')

df = pd.read_csv('/home/berk/VS_Project/simglucose/examples/trajectories/deneme.csv')
print(df.columns)
print(df.__len__)