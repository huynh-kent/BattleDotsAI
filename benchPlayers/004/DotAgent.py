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
highest_eaten = 3
best_model = Linear_QNET(4, 256, 9)
model = Linear_QNET(4, 256, 9)
class DotAgent:
    def __init__(self, cursor, num_games):
        self.num_games = num_games
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNET(4, 256, 9)
        self.trainer = Q_Trainer(self.model, lr=LR, gamma=self.gamma)
        self.cur = cursor
        self.amount_eaten = 0
        self.distance_to_food = None

    def get_flagPOS(self):
        flag = self.cur.execute("SELECT x,y from main_game_field WHERE owner_id=(select owner_id from owner where name='players.004.DotAgent') and is_flag = TRUE LIMIT 1")

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
        for food in list(self.cur.execute(f"SELECT * from main_game_field WHERE owner_id=(select owner_id from owner where name='Food') ORDER BY SQRT(abs({pos_x}-X)*({pos_x}-X) + abs({pos_y}-Y)*({pos_y}-Y)) LIMIT 1")):
            nearestfood = food
        #print (nearestfood)
        self.distance_to_food = math.sqrt((nearestfood[0]-pos_x)**2 + (nearestfood[1]-pos_y)**2)
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
            nearest_food_y > pos_y # food down

            # Enemy location

            # Friend location
            ]
        return np.array(state, dtype=int)

    def get_action(self, state):
    # random moves: tradeoff exploration / exploitation
        self.epsilon = 800 - self.num_games
        move = [0,0,0,0,0,0,0,0,0]
        if random.randint(0, 2000) < self.epsilon: #self.epsilon: # exploration
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
        if np.array_equal(action, [1, 0, 0, 0, 0, 0, 0, 0, 0]):
            x = 0
            y = -1
        elif np.array_equal(action, [0, 1, 0, 0, 0, 0, 0, 0, 0]):
            x = 1
            y = -1
        elif np.array_equal(action, [0, 0, 1, 0, 0, 0, 0, 0, 0]):
            x = 1
            y = 0
        elif np.array_equal(action, [0, 0, 0, 1, 0, 0, 0, 0, 0]):
            x = 1
            y = 1
        elif np.array_equal(action, [0, 0, 0, 0, 1, 0, 0, 0, 0]):
            x = 0
            y = 1
        elif np.array_equal(action, [0, 0, 0, 0, 0, 1, 0, 0, 0]):
            x = -1
            y = 1
        elif np.array_equal(action, [0, 0, 0, 0, 0, 0, 1, 0, 0]):
            x = -1
            y = 0
        elif np.array_equal(action, [0, 0, 0, 0, 0, 0, 0, 1, 0]):
            x = -1
            y = -1
        else:                       # [0, 0, 0, 0, 0, 0, 0, 0, 1]
            x = 0
            y = 0
        
        return x, y

    def play_step(self, new_x, new_y, state):
        global highest_eaten
        global best_model
        reward = 0
        game_over = False

        # check if game over
        for count in list(self.cur.execute("SELECT count(*) from main_game_field WHERE owner_id=(select owner_id from owner where name='Food')")):
            if count[0] < 19:
                game_over = True
        
        # check if dot is on team dot
        if list(self.cur.execute(f"select * from main_game_field as gf, owner  where x = {new_x} and y = {new_y} and gf.owner_id = owner.owner_id and owner.name='{state['NAME']}'")) is not None:
            reward -= 50
        
        # check if dot is on food
        if list(self.cur.execute(f"select * from main_game_field WHERE x = {new_x} and y = {new_y} and owner_id=(select owner_id from owner where name='Food')")) is not None:
            reward += 100
        else:
            reward -=1
      
        # check if dot moved closer to food
        for nearest_food in list(self.cur.execute(f"SELECT x,y from main_game_field WHERE owner_id=(select owner_id from owner where name='Food') ORDER BY SQRT(abs({new_x}-X)*({new_x}-X) + abs({new_y}-Y)*({new_y}-Y)) LIMIT 1")):
            if math.sqrt((nearest_food[0]-new_x)**2 + (nearest_food[1]-new_y)**2) < self.distance_to_food:
                reward += 50



        # return reward, gameover, score
        return reward, game_over, score


def init():
    return("🦍") #🦍

def run(db_cursor, state):
    train(db_cursor, state)


def train(db_cursor, state):
    global num_games
    global model
    record = 10000000
    # get all my dots
    rows = db_cursor.execute(f"select x,y from main_game_field as gf, owner  where is_flag = FALSE and gf.owner_id = owner.owner_id and owner.name = '{state['NAME']}'")
    # for each dot
    for row in rows.fetchall():
        dotAgent = DotAgent(db_cursor, num_games)
        #dotAgent.model = model
        #if model_list:
        #    dotAgent.model = model_list.pop()
        # get old state
        state_old = dotAgent.get_gamestate(row[0], row[1])

        # get move
        action = dotAgent.get_action(state_old)
        
        # perform move
        new_x, new_y = dotAgent.move_step(action)
        dotAgent.cur.execute(f"insert into engine_orders values( {row[0]}, {row[1]}, {row[0] + new_x}, {row[1] + new_y}, 'MOVE')")
        reward, done, score = dotAgent.play_step(new_x, new_y, state)

        # get new state
        newpos_x = row[0] + new_x
        newpos_y = row[1] + new_y
        state_new = dotAgent.get_gamestate(newpos_x, newpos_y)

        # train short memory
        dotAgent.train_short_memory(state_old, action, reward, state_new, done)

        # remember
        dotAgent.remember(state_old, action, reward, state_new, done)
        
        #model_list.append(dotAgent.model)
        model = dotAgent.model
        if done:
            # train long memory
            dotAgent.train_long_memory()

            best_model.save()

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
