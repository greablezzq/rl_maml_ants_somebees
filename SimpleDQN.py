import numpy as np
import random
import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd

def  actionEncode(position, antIndex, dimensions):
        return antIndex*dimensions[0]*dimensions[1]+ position

def actionDecode(action, dimensions):
        antIndex = int(action/(dimensions[0]*dimensions[1]))
        position = action % (dimensions[0]*dimensions[1])
        x = int(position/dimensions[1])
        y = position % dimensions[1]
        return (x, y, antIndex)

def actionToStrategy(colony, action):
    # colony.deploy_ant('tunnel_0_0', 'Thrower')
        ants = colony.ant_types_list
        position_action = actionDecode(action, colony.dimensions)
        x = position_action[0]
        y = position_action[1]
        antIndex = position_action[2]
        if(colony.antsPlace[x][y] == 0):
                placeName = 'tunnel_{0}_{1}'.format(x, y)
                ant = ants[antIndex]
                antName = ant["name"]
                if(colony.food >= (ant["cost"]+colony.ants_random_cost[antName])):
                        colony.deploy_ant(placeName, antName)
        return

class DQNReplayer:
    def __init__(self, capacity):
        self.memory = pd.DataFrame(index=range(capacity),
                columns=['observation', 'action', 'reward',
                'next_observation', 'done'])
        self.i = 0
        self.count = 0
        self.capacity = capacity
    
    def store(self, *args):
        self.memory.loc[self.i] = args
        self.i = (self.i + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)
        
    def sample(self, size):
        indices = np.random.choice(self.count, size=size)
        return (np.stack(self.memory.loc[indices, field]) for field in self.memory.columns)


class DQNAgent:
    def __init__(self, colony, gamma=0.99, epsilon=0.4,
             replayer_capacity=10000, batch_size=64):
        self.action_n = colony.dimensions[0] * colony.dimensions[1] * len(colony.ant_types_list)
        self.gamma = gamma
        self.epsilon = epsilon
        self.observation_dim = (colony.dimensions[0], colony.dimensions[1], 2)
        self.batch_size = batch_size
        self.replayer = DQNReplayer(replayer_capacity) 
         
        self.evaluate_net = self.build_network(input_size=self.observation_dim,
                output_size=self.action_n) 
        self.target_net = self.build_network(input_size=self.observation_dim,
                output_size=self.action_n) 

        self.target_net.set_weights(self.evaluate_net.get_weights())

    def build_network(self, input_size, output_size,
                activation=tf.nn.relu, output_activation=None,
                learning_rate=0.01): 
        model = keras.models.Sequential([
                keras.layers.Conv2D(input_shape=input_size,  filters = 3, kernel_size=(2,2), activation=activation, padding="valid"),
                keras.layers.MaxPool2D(pool_size=2, padding="same"),
                keras.layers.Flatten(),
                # keras.layers.Flatten(input_shape=input_size),
                keras.layers.Dense(units=100, activation=activation),
                keras.layers.Dense(units=80, activation=activation),
                keras.layers.Dense(units=output_size, activation=activation),
        ])
        optimizer = tf.optimizers.Adam(lr=learning_rate)
        model.compile(loss='mse', optimizer=optimizer)
        return model
        
    def learn(self, observation, action, reward, next_observation, done, update = False):
        self.replayer.store(observation, action, reward, next_observation, done) 

        observations, actions, rewards, next_observations, dones = \
                self.replayer.sample(self.batch_size) 

        next_qs = self.target_net.predict(next_observations)
        next_max_qs = next_qs.max(axis=-1)
        us = rewards + self.gamma * (1. - dones) * next_max_qs
        targets = self.evaluate_net.predict(observations)
        targets[np.arange(us.shape[0]), actions] = us
        self.evaluate_net.fit(observations, targets, verbose=0)

        if update: 
            self.target_net.set_weights(self.evaluate_net.get_weights())

        return 0

    def checkValid(self, action, colony):
        position_action = actionDecode(action, colony.dimensions)
        x = position_action[0]
        y = position_action[1]
        return colony.antsPlace[x][y] == 0

    def decide(self, observation, colony): 
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_n)
        observation = np.array(observation)
        qs = self.evaluate_net.predict(observation[np.newaxis])
        idx = np.argsort(qs[0])[::-1]
        for i in range(len(idx)):
            if(self.checkValid(idx[i], colony)):
                return idx[i]
        return np.argmax(qs)

def play_qlearning(colony, agent, count, isprint, train=False, update=False):
    episode_reward = 0
    loss = 0
    observation = [colony.beesPlace, colony.antsPlace]
    observation = np.array(observation).transpose(1,2,0)
    while True:
        action = agent.decide(observation, colony)
        if(isprint):
            print('action: ', action)
        next_observation, reward, done= colony.simulateOnce(action, count, isprint)
        next_observation = np.array(next_observation).transpose(1,2,0)
        episode_reward += reward
        if train:
            loss += agent.learn(observation, action, reward, next_observation, done, update)
        if done:
            break
        observation = next_observation
    return episode_reward, loss