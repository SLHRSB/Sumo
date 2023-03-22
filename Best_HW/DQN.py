import gym
import envs
from DQN_classes import *
import numpy as np
import matplotlib.pyplot as plt


def main(train, gamma, epsilon):
       
    num_epochs = 2000

    env = gym.make('SumoEnv-v3')
    agent = Agent(gamma, epsilon, batch_size=64, n_actions=2, eps_end=0.01,input_dims=[4], lr=0.001)
    
    Ave_Rewards=[]
    totalRewards=[]
    avg_speeds =  []
    eps_history = []

    for i in range(num_epochs):
        ep_reward = 0
        done = False
        observation = env.reset()
        speed_history =[]
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            speed = observation_[0] * 30
            if speed <-100 :
                 speed =0
            speed_history.append(speed)
            ep_reward += reward
            agent.store_transition(observation, action, reward, observation_, done)
            if train:
                agent.learn()
            observation = observation_

        totalRewards.append(ep_reward)
        Ave_Rewards.append(np.mean(totalRewards[-10:]))
        avg_speed = np.mean(speed_history)
        avg_speeds.append(avg_speed)
        eps_history.append(agent.epsilon)
        print('episode ', i, 'ep_reward %.2f' % ep_reward,'epsilon %.2f' % agent.epsilon)
       
    
    print("End of Learning. the total number of collision: ", env.collisions)
    return (Ave_Rewards, avg_speeds)



if __name__ == '__main__':

    train = 1
    gamma = 0.99    
    epsilon = 1.0
    epsilon_1, avg_speeds = main(train, gamma, epsilon)

    plt.plot(epsilon_1, label = "epsilon: 1")
    plt.ylabel('Average reward')
    plt.xlabel('epoch')
    plt.legend(loc='upper left')
    plt.show()

    plt.plot(avg_speeds, label = "avg_speeds")
    plt.ylabel('Average speed')
    plt.xlabel('epoch')
    plt.legend(loc='upper left')
    plt.show()
