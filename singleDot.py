import torch
import random
import numpy as np
from mymodel import Linear_QNET, Q_Trainer
from collections import deque
import yaml

# get dimensions
configs = None
with open('battleDots6_0.yml', 'r') as file:
    configs  = yaml.safe_load(file)
GAME_WIDTH = configs['env']['width']
GAME_HEIGHT = configs['env']['height']



MAX_MEMORY = 10_000
BATCH_SIZE = 100
LR = 0.005

class DotAgent:
    def __init__(self, cursor):
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNET(9, 256, 9)
        self.trainer = Q_Trainer(self.model, lr=LR, gamma=self.gamma)
        self.cur = cursor
        self.name = 'players.050698203.singleDot'

    def get_score(self):
        for count in list(self.cur.execute(f"select count(*) from main_game_field where is_flag=FALSE and owner_id=(select owner_id from owner where name='{self.name}')")):
            score = count[0]
        # returns amount of our dots aka score
        return score

    def get_nearestFood(self, pos_x, pos_y):
        # selects closest food
        for food in list(self.cur.execute(f"SELECT x,y, min(SQRT(({pos_x}-X)*({pos_x}-X) + ({pos_y}-Y)*({pos_y}-Y))) from main_game_field WHERE owner_id=(select owner_id from owner where name='Food')")):
            nearestfood = food
        # returns its x, y
        return nearestfood[0], nearestfood[1]
    
    def check_if_food(self):
        for food in list(self.cur.execute("SELECT * from main_game_field WHERE owner_id=(select owner_id from owner where name='Food')")):
            if food is not None:
                return True

    def get_gamestate(self, pos_x, pos_y): 
        # game table  [X , Y, owner_id, is_flag]
        # output table [src_x int, src_y int, dst_x int, dst_y int , action text]
        nearest_food_x_1, nearest_food_y_1 = self.get_nearestFood(pos_x, pos_y)

        state = [
            # Nearest Foods location
            nearest_food_x_1 < pos_x, # food left
            nearest_food_x_1 > pos_x, # food right
            nearest_food_y_1 < pos_y, # food up
            nearest_food_y_1 > pos_y, # food down

            # Next to wall
            pos_x <= 0, #wallleft
            pos_x >= GAME_WIDTH, #wallright
            pos_y <= 0,  #wallup
            pos_y >= GAME_HEIGHT, #walldown

            1, # BIAS

            ]
        return np.array(state, dtype=int)

    def get_action(self, state):
    # random moves: tradeoff exploration / exploitation
        # if ai is stuck
        if num_moves > 100:
            #randomness = 0
            randomness = num_games
        else:
            randomness = num_games
        self.epsilon = 50 - randomness # lower for loading better models
        move = [0,0,0,0,0,0,0,0,0]
        if random.randint(0, 200) < self.epsilon: #self.epsilon: # exploration
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

    def play_step(self, newpos_x, newpos_y, state):
        global reward, total_reward
        game_over = False

        # if dot goes out of bounds
        for count in list(self.cur.execute(f"SELECT count(*) from main_game_field WHERE ({newpos_x} < 0 or {newpos_x} > {GAME_WIDTH} or {newpos_y} < 0 or {newpos_y} > {GAME_HEIGHT}) and is_flag=FALSE and owner_id=(select owner_id from owner where name='{self.name}')")):
            if count[0] > 0:
                game_over = True
                reward = -100

        # check if dot is on food
        for count in list(self.cur.execute(f"select count(*) from main_game_field WHERE x = {newpos_x} and y = {newpos_y} and owner_id=(select owner_id from owner where name='Food')")):
            if count[0] > 0:
                reward += 25
                for count in list(self.cur.execute("SELECT count(*) from main_game_field WHERE owner_id=(select owner_id from owner where name='Food')")):
                    if count[0] <= 1:
                        game_over = True
                        reward += 100

        score = self.get_score()

        total_reward = (float(reward)/(float(num_moves+1))) # food per move
        
        # return reward, gameover, score
        return total_reward, game_over, score

####### plot graph
import matplotlib.pyplot as plt
from IPython import display

def plot(scores, avg_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Blue: Single game score - Orange: Total Avg Score')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(avg_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(avg_scores)-1, avg_scores[-1], str(avg_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)


def init():
    return("🐬") #🐬

trained_agents = []
high_score = 0
plot_scores = []
plot_avg_scores = []
total_score = 0
num_games = 0
num_moves = 0
total_reward = 0.0
reward = 0

def run(db_cursor, state):
    train(db_cursor, state)

def train(db_cursor, state):
    global trained_agents, high_score, plot_scores, plot_avg_scores, total_score, score, num_games, num_moves, total_reward, reward
    # get all my dots
    rows = db_cursor.execute(f"select * from main_game_field as gf, owner  where is_flag = FALSE and gf.owner_id = owner.owner_id and owner.name = '{state['NAME']}'")
    all_rows = rows.fetchall()

    for row in all_rows:
        if trained_agents:  # use existing agent
            dotAgent = trained_agents.pop()
        else:   # create agent
            dotAgent = DotAgent(db_cursor)
            dotAgent.model.load(601)



        if dotAgent.check_if_food():
            # get old state
            state_old = dotAgent.get_gamestate(row[0], row[1])

            # get move
            action = dotAgent.get_action(state_old)
            
            # perform move
            new_x, new_y = dotAgent.move_step(action)
            newpos_x = row[0] + new_x
            newpos_y = row[1] + new_y
            dotAgent.cur.execute(f"insert into engine_orders values( {row[0]}, {row[1]}, {newpos_x}, {newpos_y}, 'MOVE')")

            # get consquences of move
            new_reward, done, score = dotAgent.play_step(newpos_x, newpos_y, state)

            # get new state
            state_new = dotAgent.get_gamestate(newpos_x, newpos_y)

            # train short memory
            dotAgent.train_short_memory(state_old, action, new_reward, state_new, done)

            # remember
            dotAgent.remember(state_old, action, new_reward, state_new, done)

            # store agent in array
            trained_agents.append(dotAgent)
        else:
            done = True

        # if done train long memory
        if done:
            num_games += 1
            #dotAgent.train_long_memory()

            # if new highscore update score and save model
            if score > high_score or not dotAgent.check_if_food():
                high_score = score
                score = 0
                dotAgent.model.save(601)

            reward = 0

            break

            # plot
            #plot_scores.append(score)
            #total_score += score
            #avg_score = total_score / dotAgent.num_games
            #plot_avg_scores.append(avg_score)
            #plot(plot_scores, plot_avg_scores)
        
    num_moves += 1
    print(f"\nGame#{num_games} - DOLPHINS - Score:{score} - Record:{high_score} - MOVES#:{num_moves} - REWARD:{reward} - RPM: {total_reward}")


