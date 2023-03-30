import random
import numpy as np
from draw import DrawGrid

class GridWorld:
    def __init__(self, alpha, gamma, size_w,size_h, T):
        # Define the locations of the obstacles and rewards, q-table, actions
        self.size_w = size_w
        self.size_h = size_h
        self.obstacles = [(2, 4), (2, 5), (2, 6), (2, 7), (2, 9),
                          (6, 2), (7, 2), (8, 2), (9, 2), (7, 3),
                          (7, 6), (7, 6), (7, 8),
                          (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8)] 
        self.rewards = [(9, 9)]
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']   

        self.Q_table = np.zeros((self.size_w, self.size_h)) 
        self.Q_table_adjustment()
    
        self.learningRate = alpha
        self.discountFactor = gamma
        self.temperature = T

    def Q_table_adjustment(self):
        print(self.size_w)
        print(self.size_h)
        for row in range(self.size_h):
            for col in range(self.size_w):
                if (row, col) in self.obstacles:
                    self.Q_table[row,col]= -100
                elif (row, col) in self.rewards:
                    self.Q_table[row,col]= 100

    def decrement_temperature(self, decrement):
        if (self.temperature-decrement) >0.4:
            self.temperature = self.temperature-decrement

    # Define a function to get the next state and reward given the current state and action
    def get_next_state_and_reward(self, state, action):
        # Get the coordinates of the current state
        i, j = state
        # Move in the specified direction
        if action == 'UP':
            i = max(i - 1, 0)
        elif action == 'DOWN':
            i = min(i + 1, self.size_h - 1)
        elif action == 'LEFT':
            j = max(j - 1, 0)
        elif action == 'RIGHT':
            j = min(j + 1, self.size_w - 1)

        # Check if the new state is an obstacle
        if (i, j) in self.obstacles:
            # Move to next state and get a negative reward
            reward = -100
        else:
            # Check if the new state is a reward
            if (i, j) in self.rewards:
                reward = 100
            else:
                reward = -1
        # Update the next state
        next_state = (i, j)

        return next_state, reward

    # Define a function to select an action using Boltzmann exploration
    def boltzmann_policy(self,state, Q):  
        exp_sum = 0 
        exp_current=[]

        # calculates exp_sum and stores values of current exp
        for action in self.actions:
            next_state,reward  = self.get_next_state_and_reward(state, action)
            # print(next_state)
            
            if list(next_state) == state:
                exp_current.append(0)
            else:
                exp_sum += np.exp(Q[next_state] / self.temperature)
                exp_current.append(np.exp(Q[next_state] / self.temperature))

        #normalized current exp values
        scaled_exps =[]
        for value in exp_current:
            scaled_exps.append(value/exp_sum)

        #Defines ranges for fitness proportional
        ranges =[]
        pointer = 0
        for i in range(len(scaled_exps)):
            limits = [pointer, pointer+scaled_exps[i]]
            ranges.append(limits)
            pointer += scaled_exps[i]

        # Generates a random float between 0 and 1  
        p1Index=9999
        randomIndex = random.uniform(0,1)
        for index in range(len(ranges)):
            if randomIndex >= ranges[index][0] and randomIndex <= ranges[index][1]:
                p1Index = index
                break

        #Outputs action
        return(self.actions[p1Index])

    def verifyState(self, state):
        if state in self.obstacles:
            return False
        return True

    #Update Q_table
    def update(self, state, reward, next_state):
        self.Q_table[state[0],state[1]] += self.learningRate * (reward + (self.discountFactor * self.Q_table[next_state[0],next_state[1]]) - self.Q_table[state[0],state[1]])

    #Prepare info from q-table in necessary format to pass to Drawing file
    def visual_representation(self):
        arrows=[]
        for row in range(self.size_h):
            for col in range(self.size_w):
                future_rewards=[]
                for action in self.actions:
                    new_state, new_reward=self.get_next_state_and_reward([row,col], action)
                    if new_state == (row,col):
                        future_rewards.append(-100000)
                    else:
                        future_rewards.append(self.Q_table[new_state[0],new_state[1]])
                
                max_value = max(future_rewards)
                max_index = future_rewards.index(max_value)
                direction=self.actions[max_index]
                arrows.append(direction)

        return (arrows)


# Define the agent's parameters
NUM_EPISODES = 1000
MAX_STEPS= 70

# Define the  Grid's parameters (size, learning rate and discount factor, temperature)
SIZE_W =11
SIZE_H =11
ALPHA = 0.8
GAMMA = 0.8
TEMPERATURE = 1

# Initialize Grid and Define the main function for episodic learning
G1 = GridWorld(ALPHA, GAMMA, SIZE_W, SIZE_H, TEMPERATURE)
def run_episodes(num_episodes, max_steps):
    for episode in range(num_episodes):
        # Initialize the starting state
        state = (np.random.randint(G1.size_w), np.random.randint(G1.size_h))
        while (G1.verifyState(state) == False):
            state = (np.random.randint(G1.size_w), np.random.randint(G1.size_h))

        # Adjust temperature for every episode
        G1.decrement_temperature(TEMPERATURE/num_episodes)

        for step in range(max_steps):
            # Select an action using Boltzmann exploration
            action= G1.boltzmann_policy(state,G1.Q_table)

            # Get the next state and reward
            next_state, reward = G1.get_next_state_and_reward(state, action)

            # Update the value of the current state
            G1.update(state, reward, next_state)
        
            # # Move to the next state
            state = next_state

            # # Check if the episode has ended
            if state in G1.obstacles or state in G1.rewards or step == max_steps - 1:
                break

    # Visually represent arrows
    arrows_direction = G1.visual_representation()
    D1 = DrawGrid(G1.obstacles, G1.rewards, SIZE_W, SIZE_H , arrows_direction)

run_episodes(NUM_EPISODES, MAX_STEPS)