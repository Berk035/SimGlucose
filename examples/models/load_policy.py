import pickle
import torch
import numpy as np
from torch.autograd import Variable

filename = '/home/berk/VS_Project/simglucose/examples/trajectories/data.pickle'

def load_policy(filename):
    with open(filename,'rb') as f:
        data = pickle.loads(f.read())
    return data

def get_batch():
    data  = load_policy('/home/berk/VS_Project/simglucose/examples/trajectories/data.pickle')
    batchnum = 1
    #states = list(map(lambda x:x[0],data[batchnum]))
    #actions = list(map(lambda x:x[1],data[batchnum]))
    
    states = data['states'][:][0]
    actions = data['actions']

    return Variable(torch.cat(states)),Variable(torch.cat(actions))

load_policy