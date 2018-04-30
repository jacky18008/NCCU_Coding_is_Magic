
# coding: utf-8

# In[1]:


#numpy
import numpy as np

#記錄用的小工具
import logging

#keras
import keras.backend as K
from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Permute
from keras.optimizers import Adam
from keras.layers import Input, Dense, concatenate, Lambda, Conv2D, Reshape

#OpenAI gym
import gym
from gym import error, spaces, utils
from gym.utils import seeding

# 數學函式庫
import math

#keras-rl
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.agents.dqn import DQNAgent
from rl.core import Env
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

#讀資料用
import os, sys, csv
import numpy as np
import pickle

#繪圖
import snake




# In[3]:

#重頭戲來了，我們需要定義一個完整的RL模型，讓keras-rl跟OpenAI gym可以幫我們跑

#裡面有些東西是一定要填的，是OpenAI環境模板的規定

class Snake_Game(Env):
    
    #環境的初始化（毫不猶豫，一定要填）
    def __init__(self, mode="test"):
        
        #狀態空間、動作空間，以及reward的定義必須依照gym的資料結構
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low = 0, high = 3, shape = (8, 8))
        self.reward_range = (-1, 1000)
        
        self.mode = mode 
        #障礙物的數目
        self.num_of_blocks = 5
        
        #隨便取個名字，方便我們存資料
        self.name = "Snake"
        
        #設定隨機的seed
        self.seed()
        
        #重設遊戲
        self.reset()
        
        
    #盤面的重設，一定要填    
    def reset(self):
        
        #建立空白遊戲盤面
        self.board = np.zeros(shape=(8, 8))
        
        
        #回合數統計
        self.term = 0
        
        #檯面上有沒有糖果
        self.candy = 0
        
        #初始化盤面--加入蛇、糖果、障礙物
        self.initial_each_step()
        self.term += 1
        
        #初始分數
        self.score = 0

        #表示遊戲結束與否        
        self.DONE = False
        
        #輸掉時畫面全黑
        #self.lose_board = np.ones(shape=(8,8))

        #回傳初始盤面（gym的規定）
        return self.get_observation()
    
    def initial_each_step(self):
        #(Blank, Snake, Candy, Snake_head, Block) = (0, 1, 2, 3, -1)
        
        # first time, determine the snake location
        if self.term == 0:
            #print("first time, determine the snake location")
            
            location_row = np.random.randint(low=0, high=8)
            location_column = np.random.randint(low=0, high=6)
            self.board[location_row, location_column:location_column+3] = 1
            
            #define a list to tract the whole snake here, by the way, we put tail in first.
            self.snake_list = []
            self.snake_list.append((location_row, location_column))
            self.snake_list.append((location_row, location_column+1))
            self.snake_list.append((location_row, location_column+2))
            
            #we also need to have to tract head cooperation here.
            self.head = (location_row, location_column+2)
            self.board[self.head] = 3
            #print("initial_head: ", self.head)

            
            # default direction = 3(right)
            self.direction = 3
            
        #put candy if just eaten
        if self.candy == 0:
            blank_list = np.where(self.board == 0)
            candy_index = np.random.randint(blank_list[0].shape[0])
            self.board[blank_list[0][candy_index], blank_list[1][candy_index]] = 2
            self.candy = 1
        
        #list all the blank tile on the board
        blank_list = np.where(self.board == 0)

        if self.term%5 == 0:        
            #delete blocks generated last time.
            self.board = np.where(self.board == -1, 0, self.board)
        
            #generate blocks from all the blank tiles
            num_of_blank_tiles = blank_list[0].shape[0]
            block_list = np.random.randint(num_of_blank_tiles, size=(self.num_of_blocks))
            for i in block_list:
                self.board[blank_list[0][i], blank_list[1][i]] = -1
            
        
    #每一回合的執行（包括選擇動作、更新現有資產情況、計算reward等等）（一定要填）       
    def step(self, action):
        
        #繪圖用
        if self.mode == "test":
            snake.init()
        
        #這裡必須回傳特定資料作為紀錄（格式是字典檔），因為我們目前沒有需要，所以隨便設個空的字典檔。
        info = dict()
        
        #reward for this time
        reward = 0
    
        #回合數+1
        self.term += 1
        
        #動作會介在0~3之間，分別set as 0: up, 1: left, 2: down, 3: right。
    
        # action
        #print("action:", action)
        if action == 0:
            logging.debug("Up")
        elif action == 1:
            logging.debug("Left")
        elif action == 2:
            logging.debug("Down")
        else: #action == 3
            logging.debug("Right")
        
        #if the action has the same axis with current direction, go that direction.
        if action%2 == self.direction%2:
            action = self.direction
            
        #change direction    
        self.direction = action
        
        #vertical axis
        #change row value
        if action%2 == 0:
            #check if exceed board first:
            row = self.head[0]
            row += action-1
            
            if row < 8 and row >= 0:
                self.head = (row, self.head[1])
                #print(self.head)
            else:
                self.DONE = True
                if self.mode == "test": 
                    snake.draw( mapArray=self.board, score=self.score, isOver=self.DONE)
                return self.get_observation(), -10, self.DONE, info
                   
        #horizontal axis
        #change column value
        else:
            #check if exceed board first:
            column = self.head[1]
            column += action-2
            
            if column < 8 and column >= 0:
                self.head = (self.head[0], column)  
            else:
                self.DONE = True
                if self.mode == "test":
                    snake.draw( mapArray=self.board, score=self.score, isOver=self.DONE)
                return self.get_observation(), -10, self.DONE, info 
        
            
        #blank on target point    
        if self.board[self.head] == 0:
            #put the new head in the list, pop the tail.
            self.snake_list.append(self.head)
            pop = self.snake_list.pop(0)
            
            #update board
            
            #set all snake -> 1
            for grid in self.snake_list:
                self.board[grid] = 1
                
            self.board[self.head] = 3
            self.board[pop] = 0
            
            reward = 0
        
        #candy on target point
        elif self.board[self.head] == 2:
            
            #put head in queue without pop
            self.snake_list.append(self.head)
            
            #eat candy(update board)
            #set all snake -> 1
            for grid in self.snake_list:
                self.board[grid] = 1
                
            self.board[self.head] = 3
            self.candy = 0
            
            #reward = 1
            reward = 1

        
        #crash with blocks or its own body    
        else:
            self.DONE = True
            
            #put the new head in the list, pop the tail.
            self.snake_list.append(self.head)
            pop = self.snake_list.pop(0)
            
            #update board
            
            #set all snake -> 1
            for grid in self.snake_list:
                self.board[grid] = 1
                
            self.board[self.head] = 3
            self.board[pop] = 0
            if self.mode == "test":
                snake.draw( mapArray=self.board, score=self.score, isOver=self.DONE)
            return self.get_observation(), -10, self.DONE, info
        
        
        #because we will add new bloacks here, we should put it at the end of fcn.
        self.initial_each_step()
        
        #show
        #print(self.term, " step")
        #print(self.get_observation())
            
        #這裡要回傳什麼，回傳的順序，都是gym規定的
        self.score += reward
        if self.mode == "test":
            snake.draw( mapArray=self.board, score=self.score, isOver=self.DONE)
        return self.get_observation(), reward, self.DONE, info
            
            
    #定義一個方式，讓環境可以roll出隨機的數字（一定要填）    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    
    #取得當前狀態
    def get_observation(self):
        #get board for last 20 days
        return self.board
    
    def get_snake_length(self):
        return len(self.snake_list)

    #這裡是拿來做test時候的顯示（也是一定要填）        
    def render(self, mode='human', close=False):
        if close:
            return
        outfile = None
        
        if self.DONE == True:
            outfile = StringIO() if mode == 'ansi' else sys.stdout
            s = ""
            
            for i in range(self.get_observation().shape[0]):
                for j in range(self.get_observation().shape[1]):
                    s += str(int(self.board[i][j]))
                    s += "  "
                s += "\n"
            
            outfile.write(s)
        return outfile
    
    def close(self):
        pass
        #self.reset()
        
        
        






# In[2]:


# In[4]:

#要開始建模了

#這裡的window_length 是指當我需要傳入包括前幾次畫面作為資料時的東西，
#他是把它當作CNN的channel數一樣的東西
#本來這裡是不需要加的，只是keras-rl寫死了所以我只好傳進去。
BOARD_INPUT_SHAPE = (8, 8)
WINDOW_LENGTH = 1


#另外，由於資料最後一步是keras-rl處理的，他的變數順序這樣寫，
#我也只好這樣寫
input_shape = (WINDOW_LENGTH,) + BOARD_INPUT_SHAPE 

#設定輸入層的形狀
model_input = Input(shape = input_shape)

#視不同的backend要排一下順序
if K.image_dim_ordering() == 'tf':
    # (width, height, channels)
    permute = Permute((2, 3, 1), input_shape=input_shape)
elif K.image_dim_ordering() == 'th':
    # (channels, width, height)
    permute = Permute((1, 2, 3), input_shape=input_shape)
    
#把排列的結果套用上去，喬一下我們的原始input
preprocessed_input = permute(model_input)



# In[3]:


# 隨便弄一個model
# 雖然很抱歉，不過只有這裡可以改
Layer_1 = Dense(16, activation = "relu")(preprocessed_input)
Layer_2 = Dense(32, activation = "relu")(Layer_1)
Layer_3 = Dense(64, activation = "relu")(Layer_2)

#conv_1 = Conv2D(filters = 16, kernel_size = (3, 3), padding='same', activation = "relu")(preprocessed_input)
#conv_2 = Conv2D(filters = 32, kernel_size = (3, 3), padding='same', activation = "relu")(conv_1)
#conv_3 = Conv2D(filters = 64, kernel_size = (3, 3), padding='same', activation = "relu")(conv_2)



# In[4]:


#拉直
flatten = Flatten()(Layer_3)

#soft landing
soft_landing = Dense(64, activation="relu")(flatten)

#動作有4種，所以最後輸出是4維
action = Dense(4, activation="linear")(soft_landing)


#把整個model包起來
model = Model(model_input, action)

json_string = model.to_json()

#看看我們包出來的結果
model.summary()



# In[5]:


#準備要實地測試了：

#你可以自己決定模式跟步數
mode = input("Mode? (train or test)")
step = int(input("Step? (如果可以，請大於1000。"))

#把我們辛苦架好的遊戲環境作為測試環境
Snake_env = Snake_Game(mode)
nb_actions = Snake_env.action_space.n


#設定記憶體
memory = SequentialMemory(limit=1000000, window_length=1)

#設定策略
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=.6, value_min=.1, value_test=.00, nb_steps=step)

#DQN設定
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory = memory, policy = policy,
               nb_steps_warmup=1000, gamma=.90, target_model_update=1000)

dqn.compile(Adam(lr=.01), metrics=['mae'])


#實際跑看看
if mode == 'train':

    #儲存權重的一些設定：
    weights_filename = 'dqn_{}_weights.h5f'.format(Snake_env.name)
    checkpoint_weights_filename = 'dqn_' + Snake_env.name + '_weights_{step}.h5f'
    log_filename = 'dqn_{}_log.json'.format(Snake_env.name)
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=1000)]
    callbacks += [FileLogger(log_filename, interval=1000)]


    weights = "dqn_"+Snake_env.name+"_weights_" + str(step) + ".h5f"
    #if weights:
    #    weights_filename_1 = weights
    #dqn.load_weights(weights_filename_1)


    #訓練開始
    dqn.fit(Snake_env, callbacks=callbacks, nb_steps=step, log_interval=1000, verbose=1)

    #把權重存起來
    dqn.save_weights(weights_filename, overwrite=True)


    
elif mode == 'test':
    
    #讀取權重
    weights = "dqn_"+Snake_env.name+"_weights_" + str(step) + ".h5f"
    if weights:
        weights_filename = weights
    dqn.load_weights(weights_filename)
    dqn.test(Snake_env, nb_episodes=10, visualize=True)

