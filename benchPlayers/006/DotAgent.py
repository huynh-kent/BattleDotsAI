from unicodedata import name
import torch
import random
import numpy as np
from mymodel import Linear_QNET, Q_Trainer
from collections import deque
import os
import time
import math

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.1
score = 0
num_games = 0

class DotAgent:
    def __init__(self, cursor, num_games):
        self.num_games = num_games
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNET(8, 256, 256,  9)
        self.trainer = Q_Trainer(self.model, lr=LR, gamma=self.gamma)
        self.cur = cursor
        self.distance_to_food = None
        self.x = None
        self.y = None
        self.wallup = None
        self.walldown = None
        self.wallleft = None
        self.wallright = None

    def set_xy(self, x, y):
    #    self.x = x
    #    self.y = y
        self.check_wall(x,y)

    def check_wall(self,x,y):
        if x == 1:
            self.wallleft = True
        if y == 1:
            self.wallup = True
        if x == 29:
            self.wallright = True
        if y == 14:
            self.walldown = True

    def get_new_dot_id(self):
        for count in list(self.cur.execute("SELECT COUNT(*) from main_game_field WHERE owner_id=(select owner_id from owner where name='players.006.DotAgent') and is_flag = FALSE LIMIT 1")):
            id = count[0]
        return id

        

    def get_flagPOS(self):
        flag = self.cur.execute("SELECT x,y from main_game_field WHERE owner_id=(select owner_id from owner where name='players.006.DotAgent') and is_flag = TRUE LIMIT 1")

        for flag_pos in flag:
            #print (flag_pos)
            return flag_pos


    def update_score(self):
        global score
        new_dots = self.cur.execute(f"SELECT COUNT(*) from main_game_field WHERE X = {self.get_flagPOS()[0]} and Y = {self.get_flagPOS()[1]} and owner_id=(select owner_id from owner where name='players.003.kenty1_0') and is_flag = FALSE")

        for dot in new_dots.fetchone():
            score += dot
        # print (score)


    def get_food(self, row):
        # selects ALL food
        allfood = self.cur.execute("SELECT * from main_game_field WHERE owner_id=(select owner_id from owner where name='Food') LIMIT 1")


        for food in (allfood):
            print(food)

        # selects closest food
        nearestfood = self.cur.execute(f"SELECT * from main_game_field WHERE owner_id=(select owner_id from owner where name='Food') ORDER BY SQRT(abs({row[0]}-X)*({row[0]}-X) + abs({row[1]}-Y)*({row[1]}-Y)) LIMIT 1")

        for food in nearestfood: 
            print (food)

    def get_nearestFood(self, pos_x, pos_y):
        # selects closest food
        for food in list(self.cur.execute(f"SELECT *, min(SQRT(({pos_x}-X)*({pos_x}-X) + abs({pos_y}-Y)*({pos_y}-Y))) from main_game_field WHERE owner_id=(select owner_id from owner where name='Food')")):
            nearestfood = food
        #print (nearestfood)
        #self.distance_to_food = math.sqrt((nearestfood[0]-pos_x)**2 + (nearestfood[1]-pos_y)**2)
        return nearestfood
    
    def get_gamestate(self, pos_x, pos_y): # game table  [X , Y, owner_id, is_flag]
                            # output table [src_x int, src_y int, dst_x int, dst_y int , action text]
        nearest_food = self.get_nearestFood(pos_x, pos_y)
        nearest_food_x = nearest_food[0]
        nearest_food_y = nearest_food[1]


        state = [
            # Nearest Food location
            nearest_food_x < pos_x, # food left
            nearest_food_x > pos_x, # food right
            nearest_food_y < pos_y, # food up
            nearest_food_y > pos_y, # food down

            # Next to Wall
            pos_x < 2,
            pos_x > 28,
            pos_y < 2,
            pos_y > 13,

            ]
        return np.array(state, dtype=int)

    def get_action(self, state):
    # random moves: tradeoff exploration / exploitation
        self.epsilon = 100 - self.num_games
        move = [0,0,0,0,0,0,0,0,0]
        if random.randint(0, 500) < self.epsilon: #self.epsilon: # exploration
            move_index = random.randint(0,8)
        else: # predict with model / exploitation
            state_initial = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_initial)
            move_index = torch.argmax(prediction).item()
        move[move_index] = 1
        return move

    ### agent training
    def remember(self, state, action, reward, next_state, done): # state, action, reward, next_state, done
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            sample = random.sample(self.memory, BATCH_SIZE) # random sample
        else:
            sample = self.memory

        # get states,actions,rewards,dones from sample
        states, actions, rewards, next_states, dones = zip(*sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)



    def train_short_memory(self, state, action, reward, next_state, done):  # state, action, reward, next_state, done
        self.trainer.train_step(state, action, reward, next_state, done)

    def move_step(self, action):
        # [up, upright, right, downright, down, downleft, left, upleft, none]
        if np.array_equal(action, [1, 0, 0, 0, 0, 0, 0, 0, 0]): # and not self.wallup:
            x = 0
            y = -1
        elif np.array_equal(action, [0, 1, 0, 0, 0, 0, 0, 0, 0]): # and not self.wallright and not self.wallup:
            x = 1
            y = -1
        elif np.array_equal(action, [0, 0, 1, 0, 0, 0, 0, 0, 0]): # and not self.wallright:
            x = 1
            y = 0
        elif np.array_equal(action, [0, 0, 0, 1, 0, 0, 0, 0, 0]): # and not self.wallright and not self.walldown:
            x = 1
            y = 1
        elif np.array_equal(action, [0, 0, 0, 0, 1, 0, 0, 0, 0]): # and not self.walldown:
            x = 0
            y = 1
        elif np.array_equal(action, [0, 0, 0, 0, 0, 1, 0, 0, 0]): # and not self.walldown and not self.wallleft:
            x = -1
            y = 1
        elif np.array_equal(action, [0, 0, 0, 0, 0, 0, 1, 0, 0]): # and not self.wallleft:
            x = -1
            y = 0
        elif np.array_equal(action, [0, 0, 0, 0, 0, 0, 0, 1, 0]): # and not self.wallup and not self.wallleft:
            x = -1
            y = -1
        else:                       # [0, 0, 0, 0, 0, 0, 0, 0, 1]
            x = 0
            y = 0
        
        return x, y

    def play_step(self, newpos_x, newpos_y, state):
        reward = 0
        game_over = False

        for count in list(self.cur.execute(f"SELECT count(*) from main_game_field WHERE x = {1000} and y = {1000} ")): #owner_id=(select owner_id from owner where name='{state['NAME']}'
            if count[0] > 0:
                game_over = True

        # check if dot is on team dot
        #for count in list(self.cur.execute(f"select count(*) from main_game_field as gf, owner  where x = {newpos_x} and y = {newpos_y} and gf.owner_id = owner.owner_id and owner.name='{state['NAME']}'")):
        #    if count[0] > 0:
        #        reward = -100

        if newpos_x < 1 or newpos_x > 29 or newpos_y > 14 or newpos_y < 1:
            reward = -100

        # check if dot is on food
        for count in list(self.cur.execute(f"select count(*) from main_game_field WHERE x = {newpos_x} and y = {newpos_y} and owner_id=(select owner_id from owner where name='Food')")):
            if count[0] > 0:
                reward = 100


        # return reward, gameover, score
        return reward, game_over #, score


def init():
    return("6") #🦍

def run(db_cursor, state):
    train(db_cursor, state)


trained_agents = []

def train(db_cursor, state):
    global num_games
    global trained_agents
    record = 10000000
    # get all my dots
    rows = db_cursor.execute(f"select X,y,dot_id from main_game_field as gf, owner  where is_flag = FALSE and gf.owner_id = owner.owner_id and owner.name = '{state['NAME']}'")
    # for each dot
    for row in rows.fetchall():
        #if trained_agents:
        #    dotAgent = trained_agents.pop()
        #else:
        dotAgent = DotAgent(db_cursor, num_games)

        dotAgent.model.load(row[2]+600)
        #dotAgent.set_xy(row[0], row[1])

        # get old state
        state_old = dotAgent.get_gamestate(row[0], row[1])

        # get move
        action = dotAgent.get_action(state_old)
        
        # perform move
        new_x, new_y = dotAgent.move_step(action)

        newpos_x = row[0] + new_x
        newpos_y = row[1] + new_y

        reward, done = dotAgent.play_step(newpos_x, newpos_y, state)
        dotAgent.cur.execute(f"insert into engine_orders values( {row[0]}, {row[1]}, {newpos_x}, {newpos_y}, 'MOVE')")

        # get new state
        state_new = dotAgent.get_gamestate(newpos_x, newpos_y)

        # train short memory
        dotAgent.train_short_memory(state_old, action, reward, state_new, done)

        # remember
        dotAgent.remember(state_old, action, reward, state_new, done)
        
        #trained_agents.append(dotAgent)
        dotAgent.model.save(row[2]+600)
        if done:
            # train long memory
            #print("done")
            dotAgent.train_long_memory()
            dotAgent.model.save(row[2]+600)
            #trained_agents.pop()
            #trained_agents.append(dotAgent)
            #dotAgent.model.save()

            # TEST
            #for i in range(len(trained_agents)):
            #    agent = trained_agents[i]
            #    agent.model.save(model_file=f'Model_{i}.pth')

            ### timer TODO
            #start = time.perf_counter()
            #stop = time.perf_counter()
            #time_to_finish = start - stop
            
            #update model if high score
            #if time_to_finish < record:
                #record = time_to_finish
                #print (record)
                


            
            ##print(f"Game #{dotAgent.num_games} - Score:{score} - Record:{record}")

    num_games += 1
