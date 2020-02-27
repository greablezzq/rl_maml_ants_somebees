import numpy as np
import random
import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
from SimpleDQN import actionDecode, actionEncode, actionToStrategy

class TwoStepDQNReplayer:
    def __init__(self, capacity, antTypes):
        self.memory = [pd.DataFrame(index=range(capacity), columns=['observation', 'action', 'reward', 'next_observation', 'done']) for _ in range(antTypes)]
        self.i = [0 for _ in range(antTypes)]
        self.count = [0 for _ in range(antTypes)]
        self.capacity = capacity
        self.antTypes = antTypes
    
    def store(self, antType, *args):
        self.memory[antType].loc[self.i[antType]] = args
        self.i[antType] = (self.i[antType] + 1) % self.capacity
        self.count[antType] = min(self.count[antType] + 1, self.capacity)
        
    def sample(self, size, antType):
        if(antType is None):
            indices = np.random.choice(sum(self.count), size=size)
            memoryAll = pd.concat([self.memory[i].head(self.count[i]) for i in range(self.antTypes)], axis=0, ignore_index=True)
            return (np.stack(memoryAll.loc[indices, field]) for field in memoryAll.columns)
        else:
            indices = np.random.choice(self.count[antType], size=size)
            return (np.stack(self.memory[antType].loc[indices, field]) for field in self.memory[antType].columns)

class TwoStepDQNAgent:
    def __init__(self, colony, gamma=0.99, epsilon=0.4,
             replayer_capacity=10000, batch_size=64, lr=0.0001):
        self.antTypes = len(colony.ant_types_list)
        self.positions = colony.dimensions[0] * colony.dimensions[1]
        self.action_n = self.positions * self.antTypes
        self.gamma = gamma
        self.epsilon = epsilon
        self.observation_dim = (colony.dimensions[0], colony.dimensions[1], 2)
        self.batch_size = batch_size
        self.replayer = TwoStepDQNReplayer(replayer_capacity, self.antTypes) 
        self.evaluate_net_type = self.build_network_type(input_size=self.observation_dim, output_size=self.antTypes, learning_rate=lr) # 评估网络
        self.target_net_type = self.build_network_type(input_size=self.observation_dim, output_size=self.antTypes, learning_rate=lr) # 目标网络
        self.evaluate_net_position = [self.build_network_position(input_size=self.observation_dim, output_size=self.positions, learning_rate=lr) for _  in range(self.antTypes)]
        self.target_net_position = [self.build_network_position(input_size=self.observation_dim, output_size=self.positions, learning_rate=lr) for _  in range(self.antTypes)]

        self.target_net_type.set_weights(self.evaluate_net_type.get_weights())
        for i, net in enumerate(self.target_net_position):
            net.set_weights(self.evaluate_net_position[i].get_weights())
        
        self.trainState = 0 #(0: only train net_type; 1:only train net_position; 2:train together)

    def naiveSelectPosition(self, colony):
        x = colony.dimensions[0]
        y = colony.dimensions[1]
        for i in range(y):
            for j in range(x):
                if colony.antsPlace[j][i] == 0:
                    return j*y+i
        return 0

    def build_network_type(self, input_size, output_size,
                activation=tf.nn.relu, output_activation=None,
                learning_rate=0.01): 
        model = keras.models.Sequential([
                keras.layers.Conv2D(input_shape=input_size,  filters = 3, kernel_size=(2,2), activation=activation, padding="valid"),
                keras.layers.MaxPool2D(pool_size=2, padding="same"),
                keras.layers.Flatten(),
                # keras.layers.Flatten(input_shape=input_size),
                keras.layers.Dense(units=50, activation=activation),
                keras.layers.Dense(units=30, activation=activation),
                keras.layers.Dense(units=output_size, activation=activation),
        ])
        optimizer = tf.optimizers.Adam(lr=learning_rate)
        model.compile(loss='mse', optimizer=optimizer)
        return model

    def build_network_position(self, input_size, output_size,
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
        antIndex = int(action/self.positions)
        self.replayer.store(antIndex, observation, action, reward, next_observation, done) 

        if(self.trainState != 0):
        # position_net
            observations, actions, rewards, next_observations, dones = \
                self.replayer.sample(self.batch_size, antIndex) 
            actions = actions % self.positions
            next_qs = self.target_net_position[antIndex].predict(next_observations)
            next_max_qs = next_qs.max(axis=-1)
            us = rewards + self.gamma * (1. - dones) * next_max_qs
            targets = self.evaluate_net_position[antIndex].predict(observations)
            targets[np.arange(us.shape[0]), actions] = us
            history = self.evaluate_net_position[antIndex].fit(observations, targets, verbose=0)
            loss += history.history['loss'][0]

        if(self.trainState != 1):
        # type_net
            observations, actions, rewards, next_observations, dones = \
                self.replayer.sample(self.batch_size, None) 
            actions = actions/self.positions
            actions = actions.astype(int)
            next_qs = self.target_net_type.predict(next_observations)
            next_max_qs = next_qs.max(axis=-1)
            us = rewards + self.gamma * (1. - dones) * next_max_qs
            targets = self.evaluate_net_type.predict(observations)
            targets[np.arange(us.shape[0]), actions] = us
            history = self.evaluate_net_type.fit(observations, targets, verbose=0)
            loss += history.history['loss'][0]

        if update: # Update target net
            if(self.trainState != 1):
                self.target_net_type.set_weights(self.evaluate_net_type.get_weights())
            if(self.trainState != 0):
                for i, net in enumerate(self.target_net_position):
                    net.set_weights(self.evaluate_net_position[i].get_weights())
        
        return loss

    def checkValid(self, position, colony):
        x = int(position/colony.dimensions[1])
        y = position % colony.dimensions[1]
        return colony.antsPlace[x][y] == 0

    def decide(self, observation, colony): # epsilon greedy
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_n)
        observation = np.array(observation)
        antIndex = np.argmax(self.evaluate_net_type.predict(observation[np.newaxis])[0])
        if(self.trainState == 0):
            position = self.naiveSelectPosition(colony)
            return actionEncode(position, antIndex,colony.dimensions)
        qs = self.evaluate_net_position[antIndex].predict(observation[np.newaxis])
        idx = np.argsort(qs[0])[::-1]
        for i in range(len(idx)):
            if(self.checkValid(idx[i], colony)):
                return actionEncode(idx[i], antIndex, colony.dimensions)
        return actionEncode(np.argmax(qs), antIndex, colony.dimensions)

        

