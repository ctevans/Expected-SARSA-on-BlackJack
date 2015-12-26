import blackjack
from pylab import *
import random
import numpy

numEpisodes = 10000000
num_states = 182
num_actions = 2
returnSum = 0.0

runningCount = 0 #Used in order to tell us how many episodes have come and gone by for print statements!
alpha = 0.001
epsilonpi = 0.01
epsilonmu = 1
gamma = 1
 
 
def expectedQUnderPi(state2):
    x = (1-epsilonpi) * max(Q[state2,1], Q[state2,0])  + (epsilonpi/2)*( Q[state2,1] + Q[state2,0]) 
    return x

def behaviourPolicy(state):
    if state == 0:
        return 0; 
    else:
        epsilonRand = random.random() #random floating point b/t 0-1 auto before nop args
        if epsilonRand > epsilonmu: #GREEDY!
            if Q[state, 1] > Q[state, 0]: 
                return 1
            else: 
                return 0 
        else: #If it is NOT greedy then it is random, and thus "equiprobable" 
            equiprobable = random.random()
            if equiprobable > 0.5:
                return 0
            if equiprobable <= 0.5:
                return 1

Q = 0.00001*numpy.random.random((num_states, num_actions)) #pass as tuple, need double parenthesis. 

for episodeNum in range(numEpisodes):
    runningCount = runningCount + 1

    G = 0

    state = blackjack.init()

    while state != -1:
       action = behaviourPolicy(state)
       reward, state2 = blackjack.sample(state, action)

       Q[state][action] = Q[state][action] + alpha * (reward + gamma*expectedQUnderPi(state2) -(Q[state][action]))

       G = G + reward
       state = state2

    returnSum = returnSum + G
    
    # print the average every 10,000 episodes
    if runningCount % 10000 == 0:
        print "Average return at step", runningCount, "is", returnSum/episodeNum 

    
print "Average return: ", returnSum/numEpisodes
# print the final policy
blackjack.printPolicy(behaviourPolicy)

# Deterministic learned policy
runningCount = 0
epsilonmu = 0
returnSum = 0.0
for episodeNum in range(numEpisodes):
    runningCount = runningCount + 1

    G = 0

    state = blackjack.init()

    while state != -1:
       action = behaviourPolicy(state)
       reward, state2 = blackjack.sample(state, action)

       G = G + reward
       state = state2

    returnSum = returnSum + G
    
    # print the average every 10,000 episodes
    if runningCount % 10000 == 0:
        print "Average return at step", runningCount, "is", returnSum/episodeNum
        
print "Average return: ", returnSum/numEpisodes
# print the final policy
blackjack.printPolicy(behaviourPolicy)
