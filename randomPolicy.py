import blackjack
from pylab import *
import numpy as np 
import random

def equiprobable_policy(state):
    if state == 0:
        return 0
    else:
        equiprobable = random.random()
        if equiprobable > 0.5:
            return 0
        else:
            return 1
        
def run():
    numEpisodes = 2000
    
    returnSum = 0.0
    for episodeNum in range(numEpisodes):
        G = 0
        
        state = blackjack.init()
        
        while(state != -1):                                      # keep playing until state = -1 (arrival in terminal)
            action = equiprobable_policy(state)                  # (equiprobable policy) pick action: 0 or 1
            reward, next_state = blackjack.sample(state,action)  # get Return and Next state
            state = next_state                                   # update previous state <- next state (present)
            G = G + reward
            
        print "Episode: ", episodeNum, "Return: ", G
        returnSum = returnSum + G    
    print "Average return: ", returnSum/numEpisodes
    
run()