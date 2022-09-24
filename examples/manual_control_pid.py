#!/home/berk/VS_Project/simglucose/SIMBG/bin/python

import gym
import pygame
from gym.envs.registration import register
import sys, keyboard
from pygame.locals import *
import pickle as pkl

MODE = 1 # 0: read, 1: collect
save_path = '/home/berk/VS_Project/simglucose/examples/trajectories/data.pickle'
memory = {'states': [], 'actions': [], 'rewards': [], 'dones': []}

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

        # return the actionq
        action = control_input
        if action<0:
            action=0
        #print(f"Target: \t {(self.target)} \n Action: \t {(action)}")

        return action


def main():
    trajectories = 0
    episodes = 10
    timesteps = 100

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
    act_obj = PIDAction(P=0.1, I=0.001, D=1, target=action_ref)

    if MODE:
        for e in range(episodes):
            obs_record = []
            rew_record = []
            action_record = []

            try:
                with open(save_path, 'rb') as handle:
                    memory = pkl.load(handle)
            except:
                print('No saved trajectories found')
                pass

            for t in range(timesteps):
                env.render(mode='human')


                if keyboard.is_pressed('up'):
                    action_ref += 0.5
                if keyboard.is_pressed('down'):
                    action_ref -= 0.5

                action = act_obj.policy(action_ref)

                # Action in the gym environment is a scalar
                # representing the basal insulin, which differs from
                # the regular controller action outside the gym
                # environment (a tuple (basal, bolus)).qq
                # In the perfect situation, the agent should be able
                # to control the glucose only through basal instead
                # of asking patient to take bolus
                print(f"Action: {action}")
                print(f"Observation: {observation}")

                observation, reward, done, info = env.step(action)

                obs_record.append(observation)
                rew_record.append(reward)
                action_record.append(action)

                memory['states'].append(observation)
                memory['actions'].append(action)
                memory['rewards'].append(reward)
                memory['dones'].append(done)
                
                if done or t==timesteps:
                    observation = env.reset()
                    print("Episode finished after {} timesteps".format(t + 1))
                    trajectories +=1
                    action_ref, action = 0,0
                    act_obj = PIDAction(P=0.1, I=0.001, D=1, target=action_ref)
                    env.close()
                    break

        print('trajectories:', trajectories)
        print('states collected:', len(memory['states']))

        with open(save_path, 'wb') as handle:
            f = pkl.dump(memory, handle, protocol=pkl.HIGHEST_PROTOCOL)
            #f.close()
            print("trajectories saved to", save_path)
    else:
        try:
            with open(save_path, 'rb') as handle:
                memory = pkl.load(handle)
        except:
            print('No saved trajectories found')
            pass

        print(f"Length Memory States: {len(memory['states'])}")
        #print(f"Memory: {memory['states']}")

if __name__ == '__main__':
    main()