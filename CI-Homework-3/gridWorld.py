import random
import numpy as np
from agent import DrawGrid

class GridWorld:
    def __init__(self, alpha, gamma, size, T):
        # Define the locations of the obstacles and rewards
        # Define the initial value of all states
        self.size = size
        self.obstacles = [(0, 2),
                          (2, 4), (2, 5), (2, 6), (2, 7), (2, 9),
                          (6, 2), (7, 2), (8, 2), (9, 2), (7, 3),
                          (7, 6), (7, 6), (7, 8),
                          (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8)] 
        self.rewards = [(9, 9)]
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        
        self.Q_table = np.zeros((self.size, self.size)) 
        # self.Q_table = np.full((self.size, self.size), -1)
        self.learningRate = alpha
        self.discountFactor = gamma
        self.temperature = T
        

    # Define a function to get the next state and reward given the current state and action
    def get_next_state_and_reward(self, state, action):
        # Get the coordinates of the current state
        i, j = state
        # Move in the specified direction
        if action == 'UP':
            i = max(i - 1, 0)
        elif action == 'DOWN':
            i = min(i + 1, self.size - 1)
        elif action == 'LEFT':
            j = max(j - 1, 0)
        elif action == 'RIGHT':
            j = min(j + 1, self.size - 1)

        # Check if the new state is an obstacle
        if (i, j) in self.obstacles:
            # Stay in the current state and get a negative reward
            reward = -100
            next_state = state
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
        # # Compute the Boltzmann probabilities
        exp_sum = 0 
        exp_current=[]
        # count=0
        
        # calculates exp_sum and stores values of current exp
        for action in self.actions:
            next_state,reward  = self.get_next_state_and_reward(state, action)
            if list(next_state) == state:
                exp_current.append(0)
            else:
                exp_sum += np.exp(Q[next_state] / self.temperature)
                exp_current.append(np.exp(Q[next_state] / self.temperature))
        
        # print(exp_current)
        # print(exp_sum)

        #normalized current exp values
        scaled_exps =[]
        for value in exp_current:
            scaled_exps.append(value/exp_sum)

        print(scaled_exps)
        #Defines ranges for fitness proportional
        ranges =[]
        pointer = 0
        for i in range(len(scaled_exps)):
            limits = [pointer, pointer+scaled_exps[i]]
            ranges.append(limits)
            pointer += scaled_exps[i]

        print("BOLTZMANN RANGES", ranges)
        # Generates a random float between 0 and 1  
        p1Index=9999
        randomIndex = random.uniform(0,1)
        for index in range(len(ranges)):
            # print(index)
            if randomIndex >= ranges[index][0] and randomIndex <= ranges[index][1]:
                p1Index = index
                break
        #Outputs action
        print(self.actions[p1Index])
        return(self.actions[p1Index])

        # for action in self.actions:
        #     next_state,reward  = self.get_next_state_and_reward(state, action)
        #     if list(next_state) == state:
        #         boltzmann_probs.append(0.01)
        #     else:
        #         boltzmann_probs.append(exp_current[count]/exp_sum)
        #     count+=1

    def verifyState(self, state):
        if state in self.obstacles:
            return False
        return True

    def update(self, state, reward, next_state):
        future_rewards=[]
        print(next_state)
        for action in self.actions:
            new_state, new_reward=self.get_next_state_and_reward(next_state, action)
            print("ACTION: ", action, "REWARD: ", new_reward)
            future_rewards.append(new_reward)
        print(future_rewards)
        max_value = max(future_rewards)
        print("MAX VLAUE: ",max_value)
        print("CURRENT STATE: ",state)
        print("CUREENT VALUE: ",self.Q_table[state[0],state[1]])
        print("REWARD: ", reward)
        print((reward + (self.discountFactor * max_value) - self.Q_table[state[0],state[1]]))
        print()
        value=self.Q_table[state[0],state[1]] +  self.learningRate * (reward + (self.discountFactor * max_value) - self.Q_table[state[0],state[1]])
        print(value)
        self.Q_table[state[0],state[1]] += self.learningRate * (reward + (self.discountFactor * max_value) - self.Q_table[state[0],state[1]])


# Q[current_state, current_action] = Q[current_state, current_action] + alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[current_state, current_action])


# Define the agent's parameters
NUM_EPISODES = 1000
MAX_STEPS= 70


# Define the  Grid's parameters (size, learning rate and discount factor, temperature)
SIZE =10
ALPHA = 0.5
GAMMA = 0.9
TEMPERATURE = 0.5

#TESTING
G1 = GridWorld(ALPHA, GAMMA, SIZE, TEMPERATURE)
# D1 = DrawGrid(G1.obstacles, G1.rewards, SIZE, SIZE )

# action = G1.boltzmann_policy([0,0],G1.Q_table)
# next_state, reward = G1.get_next_state_and_reward([0,0], "RIGHT")
# print("NEXT STATE: ", next_state, "   REWARD: ", reward)
# G1.update([0,0], reward, next_state)

# next_state, reward = G1.get_next_state_and_reward([0,1], "RIGHT")
# print("NEXT STATE: ", next_state, "   REWARD: ", reward)
# G1.update([0,1], reward, next_state)


# Define the main function for episodic learning
def run_episodes(num_episodes, max_steps):
    for episode in range(num_episodes):
        # Initialize the starting state
        
        state = (np.random.randint(G1.size), np.random.randint(G1.size))
        while (G1.verifyState(state) == False):
            state = (np.random.randint(G1.size), np.random.randint(G1.size))

        print("CURRENT STATE: ", state)

        # Initialize the episode history
        history = []

        for step in range(max_steps):
            # Select an action using Boltzmann exploration
            action= G1.boltzmann_policy(state,G1.Q_table)
            print("ACTION: ", action)
            # action = np.random.choice(G1.actions)

            # Get the next state and reward
            next_state, reward = G1.get_next_state_and_reward(state, action)
            print("NEXT STATE: ", next_state, "   REWARD: ", reward)

            # Update the value of the current state
            G1.update(state, reward, next_state)
            # G1.Q_table[state] += G1 * (reward + GAMMA * G1.Q_table[next_state])
            # G1.Q_table[state] += G1.learningRate * (reward + G1.discountFactor * G1.Q_table[next_state])

            # Add the current state, action, and reward to the episode history
            history.append([state, action, reward])
        
            # # Move to the next state
            state = next_state

            # # Check if the episode has ended
            if state in G1.obstacles or state in G1.rewards or step == max_steps - 1:
                # Update the value of all states encountered in the episode
                # for state, action, reward in history:
                #     G1.update(state, action, reward, next_state)
                break
    print()
    print("FINAL Q_TABLE")
    for row in G1.Q_table:
        print(row)
        print()
    # print(G1.Q_table)

run_episodes(NUM_EPISODES, MAX_STEPS)
