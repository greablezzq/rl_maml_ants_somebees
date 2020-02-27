import numpy as np
import random
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as keras_backend
import pandas as pd
import copy
from SimpleDQN import actionDecode, actionEncode, actionToStrategy

def loss_function(pred_y, y):
  return keras_backend.mean(keras.losses.mean_squared_error(y, pred_y))

def compute_loss(model, x, y, loss_fn=loss_function):
    logits = model.forward(x)
    mse = loss_fn(y, logits)
    return mse, logits

def copy_model(model, copied_model, x):
    copied_model.forward(x)
    copied_model.set_weights(model.get_weights())
    return copied_model

class DQNTypeModel(keras.Model):
    def __init__(self, input_size, output_size, activation=tf.nn.relu, output_activation=None):
        super().__init__()
        self.layer1= keras.layers.Conv2D(input_shape=input_size,  filters = 3, kernel_size=(2,2), activation=activation, padding="valid")
        self.layer2 = keras.layers.MaxPool2D(pool_size=2, padding="same")
        self.layer3 = keras.layers.Flatten()
        self.layer4 = keras.layers.Dense(units=50, activation=activation)
        self.layer5 = keras.layers.Dense(units=30, activation=activation)
        self.layer6 = keras.layers.Dense(units=output_size, activation=activation)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x

class DQNPositionModel(keras.Model):
    def __init__(self, input_size, output_size, activation=tf.nn.relu, output_activation=None):
        super().__init__()
        self.layer1= keras.layers.Conv2D(input_shape=input_size,  filters = 3, kernel_size=(2,2), activation=activation, padding="valid")
        self.layer2 = keras.layers.MaxPool2D(pool_size=2, padding="same")
        self.layer3 = keras.layers.Flatten()
        self.layer4 = keras.layers.Dense(units=100, activation=activation)
        self.layer5 = keras.layers.Dense(units=80, activation=activation)
        self.layer6 = keras.layers.Dense(units=output_size, activation=activation)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x

class TwoStepDQNReplayer:
    def __init__(self, antTypes):
        self.memory = [pd.DataFrame(columns=['observation', 'action', 'reward', 'next_observation', 'done']) for _ in range(antTypes)]
        self.count = [0 for _ in range(antTypes)]
        self.antTypes = antTypes
    
    def store(self, antType, *args):
        self.memory[antType].loc[self.count[antType]] = args
        self.count[antType] = self.count[antType] + 1
        
    def sample(self, size, antType):
        if(antType is None):
            indices = np.random.choice(sum(self.count), size=size)
            memoryAll = pd.concat([self.memory[i].head(self.count[i]) for i in range(self.antTypes)], axis=0, ignore_index=True)
            return (np.stack(memoryAll.loc[indices, field]) for field in memoryAll.columns)
        else:
            indices = np.random.choice(self.count[antType], size=size)
            return (np.stack(self.memory[antType].loc[indices, field]) for field in self.memory[antType].columns)

class TwoStepDQNAgent:
    def __init__(self, colony, count, args, actionToStrategy, sampleTask, K=5, gamma=0.99, epsilon=0.4, batch_size=64, batch_size_for_MAML=5, alpha=0.0001):
        self.antTypes = len(colony.ant_types_list)
        self.positions = colony.dimensions[0] * colony.dimensions[1]
        self.action_n = self.positions * self.antTypes
        self.gamma = gamma
        self.epsilon = epsilon
        self.observation_dim = (colony.dimensions[0], colony.dimensions[1], 2)
        self.batch_size = batch_size 
        self.evaluate_net_type = self.build_network_type(input_size=self.observation_dim, output_size=self.antTypes) # 评估网络
        self.copied_net_type = self.build_network_type(input_size=self.observation_dim, output_size=self.antTypes) # 目标网络
        self.evaluate_net_position = [self.build_network_position(input_size=self.observation_dim, output_size=self.positions) for _  in range(self.antTypes)]
        self.copied_net_position = [self.build_network_position(input_size=self.observation_dim, output_size=self.positions) for _  in range(self.antTypes)]
        self.validation_net_type = self.build_network_type(input_size=self.observation_dim, output_size=self.antTypes)
        self.validation_net_position = [self.build_network_position(input_size=self.observation_dim, output_size=self.positions) for _  in range(self.antTypes)]

        self.copied_net_type.set_weights(self.evaluate_net_type.get_weights())
        for i, net in enumerate(self.copied_net_position):
            net.set_weights(self.evaluate_net_position[i].get_weights())
        
        self.trainState = 0 #(0: only train net_type; 1:only train net_position; 2:train together)
        self.count = count
        self.isprint = False
        self.K = K
        self.sampleTask = sampleTask
        self.batch_size_for_MAML = batch_size_for_MAML
        self.alpha = alpha
        self.args = args
        self.actionToStrategy = actionToStrategy

        self.episodes_reward=[]

    def naiveSelectPosition(self, colony):
        x = colony.dimensions[0]
        y = colony.dimensions[1]
        for i in range(y):
            for j in range(x):
                if colony.antsPlace[j][i] == 0:
                    return j*y+i
        return 0

    def build_network_type(self, input_size, output_size,
                activation=tf.nn.relu, output_activation=None): 
        model = DQNTypeModel(input_size, output_size, activation, output_activation)
        return model

    def build_network_position(self, input_size, output_size,
                activation=tf.nn.relu, output_activation=None):
        model = DQNPositionModel(input_size, output_size, activation, output_activation)
        return model
        
    def sampleTrajectories(self, type_model, position_model, colony):
        replayer = TwoStepDQNReplayer(self.antTypes)
        for _ in range(self.K):
            newcolony = copy.deepcopy(colony)
            replayer, _ = self.play_qlearning(newcolony, self.count, type_model, position_model, replayer, self.isprint)
        return replayer

    def play_qlearning(self, colony, count, type_model, position_model, replayer, isprint):
        episode_reward = 0
        observation = [colony.beesPlace, colony.antsPlace]
        observation = np.array(observation).transpose(1,2,0)
        if (replayer is None):
            replayer = TwoStepDQNReplayer(self.antTypes)
        while True:
            action = self.decide(observation, colony, type_model, position_model)
            if(isprint):
                print('action: ', action)
            next_observation, reward, done= colony.simulateOnce(action, count, isprint)
            next_observation = np.array(next_observation).transpose(1,2,0)
            antIndex = int(action/self.positions)
            episode_reward += reward
            replayer.store(antIndex, observation, action, reward, next_observation, done)
            if done:
                self.episodes_reward.append(episode_reward)
                break
            observation = next_observation
        return replayer, episode_reward

    def generateTrainingDataType(self, replayer, model):
        observations, actions, rewards, next_observations, dones = replayer.sample(self.batch_size, None)
        actions = actions/self.positions
        actions = actions.astype(int)
        next_qs = model.forward(next_observations).numpy()
        next_max_qs = next_qs.max(axis=-1)
        us = rewards + self.gamma * (1. - dones) * next_max_qs
        targets = model.forward(observations).numpy()
        targets[np.arange(us.shape[0]), actions] = us
        return observations, targets

    def generateTrainingDataPosition(self, replayer, model, antIndex):
        observations, actions, rewards, next_observations, dones = replayer.sample(self.batch_size, antIndex) # 经验回放
        actions = actions % self.positions
        next_qs = model.forward(next_observations).numpy()
        next_max_qs = next_qs.max(axis=-1)
        us = rewards + self.gamma * (1. - dones) * next_max_qs
        targets = model.forward(observations).numpy()
        targets[np.arange(us.shape[0]), actions] = us
        return observations, targets

    def  trainOnce(self):
        optimizer = keras.optimizers.Adam()
        with tf.GradientTape(persistent=True) as test_tape:
            overallLossType = 0
            overallLossPosition = [0 for _ in range(self.antTypes)]
            for i  in range(self.batch_size_for_MAML):
                colony = self.sampleTask(self.args, self.actionToStrategy, True)
                replayer = self.sampleTrajectories(self.evaluate_net_type, self.evaluate_net_position, colony)
                if(self.trainState != 1):
                    observations, targets = self.generateTrainingDataType(replayer, self.evaluate_net_type)
                    with tf.GradientTape() as train_tape:
                        train_loss, _ = compute_loss(self.evaluate_net_type, observations, targets)
                    gradients = train_tape.gradient(train_loss, self.evaluate_net_type.trainable_variables)
                    k = 0
                    self.copied_net_type = copy_model(self.evaluate_net_type, self.build_network_type(input_size=self.observation_dim, output_size=self.antTypes), observations)
                    for j in [0, 3, 4, 5]:
                        self.copied_net_type.layers[j].kernel = tf.subtract(self.evaluate_net_type.layers[j].kernel,
                                tf.multiply(self.alpha, gradients[k]))
                        self.copied_net_type.layers[j].bias = tf.subtract(self.evaluate_net_type.layers[j].bias,
                                tf.multiply(self.alpha, gradients[k+1]))
                        k += 2
                if(self.trainState != 0):
                    for t in range(self.antTypes):
                        observations, targets = self.generateTrainingDataPosition(replayer, self.evaluate_net_position[t], t)
                        with tf.GradientTape() as train_tape:
                            train_loss, _ = compute_loss(self.evaluate_net_position[t], observations, targets)
                        gradients = train_tape.gradient(train_loss, self.evaluate_net_position[t].trainable_variables)
                        k = 0
                        self.copied_net_position[t] = copy_model(self.evaluate_net_position[t], self.build_network_position(input_size=self.observation_dim, output_size=self.positions), observations)
                        for j in [0, 3, 4, 5]:
                            self.copied_net_position[t].layers[j].kernel = tf.subtract(self.evaluate_net_position[t].layers[j].kernel,
                                    tf.multiply(self.alpha, gradients[k]))
                            self.copied_net_position[t].layers[j].bias = tf.subtract(self.evaluate_net_position[t].layers[j].bias,
                                    tf.multiply(self.alpha, gradients[k+1]))
                            k += 2
                replayer_test = self.sampleTrajectories(self.copied_net_type, self.copied_net_position, colony)
                if(self.trainState != 1):
                    observation_test, targets_test = self.generateTrainingDataType(replayer_test, self.copied_net_type)
                    test_loss, _ = compute_loss(self.copied_net_type, observation_test, targets_test)
                    overallLossType += test_loss
                if(self.trainState != 0):
                    for t in range(self.antTypes):
                        observation_test, targets_test = self.generateTrainingDataPosition(replayer_test, self.copied_net_position[t], t)
                        test_loss, _ = compute_loss(self.copied_net_position[t], observation_test, targets_test)
                        overallLossPosition[t] += test_loss
        if(self.trainState != 1):             
            gradients = test_tape.gradient(overallLossType, self.evaluate_net_type.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.evaluate_net_type.trainable_variables))
        if(self.trainState != 0):
            for t in range(self.antTypes):
                gradients = test_tape.gradient(overallLossPosition[t], self.evaluate_net_position[t].trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.evaluate_net_position[t].trainable_variables))


    def checkValid(self, position, colony):
        x = int(position/colony.dimensions[1])
        y = position % colony.dimensions[1]
        return colony.antsPlace[x][y] == 0

    def decide(self, observation, colony, type_model, position_model): # epsilon greedy
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_n)
        observation = np.array(observation)
        antIndex = np.argmax(type_model.forward(observation[np.newaxis])[0])
        if(position_model is None or self.trainState == 0):
            position = self.naiveSelectPosition(colony)
            return actionEncode(position, antIndex,colony.dimensions)
        qs = position_model[antIndex].forward(observation[np.newaxis])
        idx = np.argsort(qs[0])[::-1]
        for i in range(len(idx)):
            if(self.checkValid(idx[i], colony)):
                return actionEncode(idx[i], antIndex, colony.dimensions)
        return actionEncode(np.argmax(qs), antIndex, colony.dimensions)

    def validation(self, episodes, colony):
        optimizer = keras.optimizers.Adam()
        for i in range(episodes):
            replayer = self.sampleTrajectories(self.validation_net_type, self.validation_net_position, colony)
            if(self.trainState != 1):
                observations, targets = self.generateTrainingDataType(replayer, self.validation_net_type)
                with tf.GradientTape() as valid_tape:
                    train_loss, _ = compute_loss(self.validation_net_type, observations, targets)
                gradients = valid_tape.gradient(train_loss, self.validation_net_type.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.validation_net_type.trainable_variables))
            if(self.trainState != 0):
                for t in range(self.antTypes):
                    observations, targets = self.generateTrainingDataPosition(replayer, self.validation_net_position[t], t)
                    with tf.GradientTape() as valid_tape:
                        train_loss, _ = compute_loss(self.validation_net_position[t], observations, targets)
                    gradients = valid_tape.gradient(train_loss, self.validation_net_position[t].trainable_variables)
                    optimizer.apply_gradients(zip(gradients, self.validation_net_position[t].trainable_variables))
        epsilon = self.epsilon
        self.epsilon = 0
        _, episode_reward = self.play_qlearning(colony, self.count, self.validation_net_type, self.validation_net_position, None, False)
        self.epsilon = epsilon
        return episode_reward


        

