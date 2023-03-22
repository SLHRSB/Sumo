import numpy as np
import matplotlib.pyplot as plt
import gym
import envs
import traci
import pickle


def get_disc_state(observation, pos_space,speed_space,angle_space,angular_vel_space):
    position, speed, angle, angular_vel = observation
    position = int(np.digitize(position, pos_space))
    speed = int(np.digitize(speed, speed_space))
    angle = int(np.digitize(angle, angle_space))
    angular_vel = int(np.digitize(angular_vel, angular_vel_space))

    return (position, speed, angle, angular_vel)
    

def select_action(Q, state, epsilon, env):    
    values = np.array([Q[state,a] for a in range(2)])
    max_action = np.argmax(values)
    rand = np.random.random()
    a = max_action if rand < (1-epsilon) else env.action_space.sample()

    return a


def main(train, buckets,alpha,gamma,epsilon):
    buckets = buckets
    alpha = alpha
    gamma =gamma    
    epsilon = epsilon
    num_epochs = 1000
    
    #discretize the spaces    
    pos_space = np.linspace(0, 1, buckets)
    speed_space = np.linspace(0, 1, buckets)
    angle_space = np.linspace(0, 1, buckets)
    angular_vel_space = np.linspace(0, 1, buckets)
    
    
    env = gym.make('SumoEnv-v3')

    states = [(i,j,k,l) for i in range(len(pos_space)+1) for j in range(len(speed_space)+1)
          for k in range(len(angle_space)+1) for l in range(len(angular_vel_space)+1)]

    if train:
        Q = {(s, a): 0 for s in states for a in range(2)}
    else:
        print("test")
        with open('q_table.pkl', 'rb') as f:
            Q = pickle.load(f)
   
    
    Rewards=[]
    totalRewards=[]
    avg_speeds =  []

    for i in range(num_epochs):
        # print('epoch ', i)
        if i == 0 : 
          print("Start!")

        observation = env.reset()        
        s = get_disc_state(observation, pos_space,speed_space,angle_space,angular_vel_space)
        a = select_action(Q, s, epsilon, env) 
        done = False
        epRewards = 0
        speed_history =[]

        while not done:
            # env.render()
            observation_, reward, done, info = env.step(a)  
            speed = observation_[0] * 30
            if speed <-100 :
                 speed =0 

            speed_history.append(speed)
            s_ = get_disc_state(observation_, pos_space,speed_space,angle_space,angular_vel_space)
            a_ = select_action(Q, s_, epsilon, env)
            epRewards += reward
            Q[s,a] = Q[s,a] + alpha*(reward + gamma*Q[s_,a_] - Q[s,a])
            s, a = s_, a_    

        epsilon -= 1/(num_epochs) if epsilon > 0 else 0

        totalRewards.append(epRewards)
        Rewards.append(np.mean(totalRewards[-50:]))

        avg_speed = np.mean(speed_history)
        avg_speeds.append(avg_speed)
        
    env.closeSimulator()
    if train:
        with open('q_table.pkl', 'wb') as f:
            pickle.dump(Q, f)
    print("End of Learning. the total number of collision: ", env.collisions)
    return (Rewards, avg_speeds)



train = 0
buckets = 10
alpha = 0.1
gamma = 0.9    
epsilon = 1.0
epsilon_1, avg_speeds = main(train, buckets,alpha,gamma,epsilon)

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