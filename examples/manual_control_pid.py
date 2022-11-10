#!/home/berk/VS_Project/simglucose/SIMBG/bin/python

from datetime import date
import gym
from numpy import array
import pygame
from gym.envs.registration import register
import sys, keyboard
from pygame.locals import *
import pickle as pkl
import argparse
from simglucose.controller.base import Action
import pandas as pd
import glob
from datetime import datetime
import numpy as np
from simglucose.analysis.risk import risk_index


class PIDAction:
    def __init__(self, P=1, I=0, D=0, target=1):
        self.P = P
        self.I = I
        self.D = D
        self.target = target
        self.integrated_state = 0
        self.prev_state = 0

    def policy(self, current_act):
        sample_time = 3

        # BG is the only state for this PID controller
        control = current_act
        control_input = self.P * (control - self.target) + \
            self.I * self.integrated_state + \
            self.D * (control - self.prev_state) / sample_time

        # update the states
        self.prev_state = control
        self.integrated_state += (control - self.target) * sample_time

        # # return the actionq
        action = control_input
        if action <=0:
            #self.target=0
            action=0
        # #print(f"Target: \t {(self.target)} \n Action: \t {(action)}")
    
        #print("Target:%.3f"%self.target)
        return action
        
    def reset(self):
        self.prev_state = 0
        self.integrated_state = 0
        
    def key_callback(self,act):
        if keyboard.is_pressed('up'):
            act +=0.5
            #self.target+=0.5
        if keyboard.is_pressed('down'):
            act -=0.5
        return act


def main():
    n_trajectory = 0

    # Register gym environment. By specifying kwargs,
    # you are able to choose which patient to simulate.
    # patient_name must be 'adolescent#001' to 'adolescent#010',
    # or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
    register(
        id='simglucose-adult-v1',
        entry_point='simglucose.envs:T1DSimEnv',
        kwargs={'patient_name': 'adult#001'}
    )

    env = gym.make('simglucose-adult-v1')
    observation = env.reset()
    action_ref = 0
    act_obj = PIDAction(P=args.pid_tune[0], 
                        I=args.pid_tune[1], 
                        D=args.pid_tune[2], target=action_ref)

    # 0: read, 1: collect
    if args.collect:

        try:
            with open(args.save_path, 'rb') as handle:
                memory = pkl.load(handle)
        except:
            print('No saved trajectories found')
            pass
        
        memory = {'observations': [], 'next_observations':[], 'actions':[], 'rewards':[], 'terminals':[]}
        for e in range(args.episodes):
            obs_record = []
            next_obs_record = []
            rew_record = []
            action_record = []
            dones = []
            timestamps = []

            for t in range(args.timesteps):
                env.render(mode='human')

                if args.collect: action_ref = act_obj.key_callback(action_ref)
                # if keyboard.is_pressed('up'):
                #     action_ref +=0.5
                # if keyboard.is_pressed('down'):
                #     action_ref -=0.5
                action = act_obj.policy(action_ref)

                # Action in the gym environment is a scalar
                # representing the basal insulin, which differs from
                # the regular controller action outside the gym
                # environment (a tuple (basal, bolus)).qq
                # In the perfect situation, the agent should be able
                # to control the glucose only through basal instead
                # of asking patient to take bolus

                next_observation, reward, done, info = env.step(action)
                insulin_value = env.env.insulin_hist[-1]
                _, _, risk = risk_index([observation.CGM], 1)
                _, _, next_risk = risk_index([next_observation.CGM], 1)

                obs_record.append([np.array([[observation.CGM], [risk]])])
                next_obs_record.append(np.array([[next_observation.CGM], [next_risk]]))
                action_record.append(np.array([insulin_value]))
                rew_record.append(np.array([reward]))
                dones.append(np.array([done]))
                timestamps.append(t)

                observation = next_observation

                if done or t==(args.timesteps-1):
                    n_trajectory+=1
                    #GAIL uses fixed timestep for all records
                    if t==(args.timesteps-1):

                        memory['observations'].append(obs_record)
                        memory['next_observations'].append(next_obs_record)
                        memory['actions'].append(action_record)
                        memory['rewards'].append(rew_record)
                        memory['terminals'].append(dones)

                        concat_fnc = lambda a,b: [a,b]
                        df = {'observations': list(map(concat_fnc,env.env.BG_hist[:-1],env.env.risk_hist[:-1])), 
                                     'next_observations': next_obs_record,'actions': env.env.insulin_hist, 
                                     'rewards': rew_record, 'terminals': dones}
                    
                        # df['observations']=obs_record
                        # df['next_observations']=next_obs_record
                        # df['actions']=action_record
                        # df['rewards']=rew_record
                        # df['terminals']=dones
                        # df.index = env.env.time_hist[:-1]
                        # df.index.name = 'Date'

                        #Save as pickle file
                        with open(args.save_path + f"_eps_{n_trajectory}" + "-%s.pkl"%datetime.now().replace(second=0, microsecond=0), 'wb') as handle:
                            pkl.dump(df, handle, protocol=pkl.HIGHEST_PROTOCOL)
                        
                        with open(args.save_path + "_Memory" + f"_eps_{n_trajectory}" + "-%s.pkl"%datetime.now().replace(second=0, microsecond=0), 'wb') as handle:
                            pkl.dump(memory, handle, protocol=pkl.HIGHEST_PROTOCOL)

                        #Save as csv file
                        df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in df.items() ]))
                        df.to_csv(args.save_path + f"_eps_{n_trajectory}" + "-%s.csv"%datetime.now().replace(second=0, microsecond=0))

                        df2 = pd.DataFrame(data=memory)
                        df2.to_csv(args.save_path + "_Memory" + f"_eps_{n_trajectory}" + "-%s.csv"%datetime.now().replace(second=0, microsecond=0))


                    print("Episode finished after {} timesteps".format(t + 1))
                    observation = env.reset()
                    act_obj.reset()
                    env.close()
                    action_ref = 0

        print('trajectories:', n_trajectory)
        print('states collected:', len(memory['observations']))

    else:
        try:
            with open(args.save_path, 'rb') as handle:
                memory = pkl.load(handle)
        except:
            print('No saved trajectories found')
            pass

        print(f"Length Memory Trajectories: {len(memory['dones'])}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--collect', type=bool, default= 1)
    parser.add_argument('--save_path', type=str, default= '/home/berk/VS_Project/simglucose/examples/trajectories/DATA')
    parser.add_argument('--pid_tune', nargs="+", default= [0.5, 0, 0.1])
    parser.add_argument('--episodes', type=int, default= 1)
    parser.add_argument('--timesteps', type=int, default= 100)

    args = parser.parse_args()
    main()