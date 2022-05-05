import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNET(nn.Module):
    def __init__(self, input_size, layer1,layer2,output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, layer1)
        self.linear2 = nn.Linear(layer1, layer2)
        self.linear3 = nn.Linear(layer1, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
    
    def save(self, model_num):
        model_file= f'Model3.0_{model_num}.pth'
        folder_path = './NNetModel'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        model_file = os.path.join(folder_path, model_file)
        torch.save(self.state_dict(), model_file)
    
    def load(self, model_num):
        model_file= f'Model3.0_{model_num}.pth'
        folder_path = './NNetModel'
        model_file = os.path.join(folder_path, model_file)

        if os.path.isfile(model_file):
            self.load_state_dict(torch.load(model_file), strict=False)
            self.eval()
            #print ('Loading Model')
            #return True
        
        #print ('No model found')
        #return False


class Q_Trainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

    # 1: predicted Q values with current state
        prediction = self.model(state)
        target = prediction.clone()
        for index in range(len(done)):
            Q_new = reward[index]

            if not done[index]:
                Q_new = reward[index] + self.gamma * torch.max(self.model(next_state[index]))
            
            target[index][torch.argmax(action).item()] = Q_new
            # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
                # pred.clone()
                # preds[argmax(action)] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, prediction)
        loss.backward()

        self.optimizer.step()


