import gym
import numpy as np
from gym import wrappers
import random
import envs
import traci
from collections import deque


if __name__ == '__main__':

	n_games = 20
	load_checkpoint = False
	Train = False

	ENV_NAME = 'SumoEnv-v3'
	env = gym.make(ENV_NAME)

	# Here you can add your RL algorithm:

	for i in range(n_games):
		i+=1
		observation = env.reset()
		print (observation)
		action = 0.6
		observation_, step_reward, done, info = env.step(action)
		print (observation_)