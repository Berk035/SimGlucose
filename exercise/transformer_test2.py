import pickle as pkl
import numpy as np
from torch.utils.data import Dataset, DataLoader

with open ('/home/berk/VS_Project/simglucose/examples/trajectories/DATA_eps_11-2022-11-21 20:20:00.pkl', 'rb') as handle:
    df = pkl.load(handle)

class CustomDataset(Dataset):
    def __init__(self, data):
        self.obs = data["observations"][0]
        self.n_obs = data["next_observations"][0]
        self.actions = data["actions"][0]
        self.rewards = data["rewards"][0]
        self.dones = data["terminals"][0]

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        obs = self.obs[idx]
        next_obs = self.n_obs[idx]
        actions = self.actions[idx]
        reward = self.rewards[idx]
        dones = self.dones[idx]
        
        sample = {"observation": obs, "next_observations": next_obs, "actions": actions, "reward": reward, "dones": dones}
        return sample

def collate_batch(batch):
    #print("Type of batch: ", type(batch))
    
    #Return without defult tensor type
    return batch 

df2 = CustomDataset(df)
dataset = DataLoader(df2, batch_size=10, shuffle=False, collate_fn=collate_batch)
#dataset = DataLoader(df2, batch_size=100, shuffle=False)

# Display image and label.
print('\nFirst iteration of data set: ', next(iter(dataset)), '\n')

# Print how many items are in the data set
print('Length of data set: ', len(dataset), '\n')

# Print entire data set
print('Entire data set: ', list(dataset), '\n')