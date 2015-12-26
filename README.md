# Expected-SARSA-on-Blackjack Taking the card game Blackjack, let's solve it with Expected SARSA! 

Language: Python 2.x 

Tools: Canopy, numpy. (The HTML files provided are used to open the data generated by canopy.) 

The card game blackjack allows the players to either ask for another card or to hold, these basic actions can be decided upon by an agent (the program) and learned through experience. This is using Reinforcement Learning in order to solve the problem of when we should or should not hold in blackjack, this also includes data on whether an ace was or was not used (which complicates the problem likewise!)! 


##Expected SARSA (A brief introduction):
Expected SARSA is a TD learning method from Reinforcement Learning, this algorithm is going to modify the set of values associated with each experience that the agent may encounter through the various trails ran through by the program. Expected SARSA will be updating a pool of values representing various states the agent encounters by an update equation that is based upon the current value of the state action pair (commonly referred to as Q) and the bootstrapped estimate of all various state action pairs that the agent may take from it's current state. 

In other words, as the agent explores through the "environment" of blackjack it will be making constant updates to the value of each state-action pair, which will in turn impact how the agent will choose it's future actions. Leading to the agent picking the "best" action in a given situation.

##Results:
The results of running this algorithm can be seen in the results.pdf file! 