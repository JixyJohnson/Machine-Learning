import numpy as np
import random

states = 6
actions = 2
Q = np.zeros((states, actions))

alpha = 0.8
gamma = 0.9

for episode in range(500):
    state = random.randint(0,4)
    while state != 5:
        action = random.randint(0,1)
        next_state = min(5,state+1) if action==1 else max(0,state-1)
        reward = 100 if next_state==5 else -1
        Q[state][action] = Q[state][action] + alpha * (
            reward + gamma*np.max(Q[next_state]) - Q[state][action])
        state = next_state

print(Q)
