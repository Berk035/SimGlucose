import pickle as pkl
import numpy as np
from torch.utils.data import Dataset, DataLoader

with open ('/home/berk/VS_Project/simglucose/examples/trajectories/hopper-medium-replay-v2.pkl', 'rb') as handle:
    orj_data = pkl.load(handle)

with open ('/home/berk/VS_Project/simglucose/examples/trajectories/DATA_eps_11-2022-11-21 20:20:00.pkl', 'rb') as handle:
    own_data = pkl.load(handle)

print(type(orj_data))
print(type(own_data))

# print(type(orj_data[0]))
# row=len(orj_data)
# column=len(orj_data[0])
# print(f'Rows:{row}, Column:{column}')
# print("Shape of a list:",len(orj_data))

# for key, values in orj_data[0].items():
#     print(type(values))

row=len(own_data)
column=len(own_data[0])
print(f'Rows:{row}, Column:{column}')
print("Shape of a list:",len(own_data))