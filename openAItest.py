import gym
import random 
import numpy as np
from collections import Counter
import tflearn
from tflearn.layers.core import input_data,fully_connected,dropout
from tflearn.layers.estimator import regression

training_len=10000
scoreRequired=50

env = gym.make('CartPole-v0')
env._max_episode_steps = 500

env.reset()

observationArray=[]

def random_test():
    for test in range(5):
        env.reset()
        for _ in range(10):
            env.render()
            action = env.action_space.sample()
            print(action)
            observation, reward, done, info = env.step(action)
            observationArray.append(observation)
            if done:
                break
    
training_data = []
score_append = []

def generate_training_data():
    for _ in range(10000):
        score = 0
        game_memory = []
        prev_observation = []
        for _ in range(500):
            action = random.randrange(0,2)
            observation, reward, done, info = env.step(action)
            score += reward

            if len(prev_observation):
                game_memory.append([prev_observation,action])

            prev_observation = observation

            if done:
                break
        
        #print(score)
        if(score>=scoreRequired):
            score_append.append(score)
            for data in game_memory:
                if(data[1]==1):
                   output = [0,1]
                elif(data[1]==0):
                    output = [1,0]
                training_data.append([data[0],output])
        env.reset()

    training_data_save = np.array(training_data,dtype=object)
    #training_data_save = np.random.shuffle(training_data_save)
    #np.save('save_training_data',training_data_save)
    print("Average: ", sum(score_append)/len(score_append))
    print(Counter(score_append))
    return training_data_save



def neural_network_model(input_size):
    network = input_data(shape = [None,input_size,1],name='input')

    network = fully_connected(network, 128, activation = 'relu' )
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation = 'relu' )
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation = 'relu' )
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation = 'relu' )
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation = 'relu' )
    network = dropout(network, 0.8)    

    network = fully_connected(network, 2, activation = 'softmax' )
    
    network = regression(network, optimizer='adam',learning_rate=0.001,loss='categorical_crossentropy',name='targets')

    model = tflearn.DNN(network)

    return model



def train_model(train_data, model = False):
    X=np.array([i[0] for i in train_data]).reshape(-1, len(train_data[0][0]), 1)
    Y=[i[1] for i in train_data]

    if not model:
        model = neural_network_model(input_size=len(X[0]))
    
    model.fit({'input':X},{'targets':Y},n_epoch=3,show_metric=True,run_id='openAItestPole')
    return model

test_score = []
test_choice = []

def test_model(model):

    for test in range(5):
        prev_observation = []
        env.reset()
        score = 0
        for _ in range(700):
            env.render()
            if(len(prev_observation)==0):
                action = random.randrange(0,2)
            else:
                action=np.argmax(model.predict(prev_observation.reshape(-1,len(prev_observation),1))[0])
            
            # print(action)
            test_choice.append(action)
            observation, reward, done, info = env.step(action)
            
            score += reward
            prev_observation = np.array(observation)

            if done:
                break
        
        test_score.append(score)




# mainTrainData = generate_training_data()
# # model = train_model(mainTrainData)
# # model.save('cartPoleModel2.tflearn')
model = neural_network_model(4)
model.load('./cartPoleModel2.tflearn')
test_model(model)

print('Average test score:',sum(test_score)/len(test_score))




