import collections, util, math, random
from typing import List

############################################################
# Problem 4.1.1

def computeQ(mdp, V, state, action):
    """
    Return Q(state, action) based on V(state).  Use the properties of the
    provided MDP to access the discount, transition probabilities, etc.
    In particular, MDP.succAndProbReward() will be useful (see util.py for
    documentation).  Note that |V| is a dictionary.  
    """
    # BEGIN_YOUR_CODE (around 2 lines of code expected)
    # successor = (newState, prob, reward)
    return sum([ successor[1] * (successor[2] + mdp.discount() * V[successor[0]]) for successor in mdp.succAndProbReward(state, action)])
    # END_YOUR_CODE

def computeNoisyQ(mdp, V, state, action, alpha):
    sumT = sum([successor[1] + alpha for successor in mdp.succAndProbReward(state, action)])
    return sum([(successor[1] + alpha) * (successor[2] + mdp.discount() * V[successor[0]]) / sumT for successor in mdp.succAndProbReward(state, action)])

############################################################
# Problem 4.1.2

def policyEvaluation(mdp, V, pi, epsilon=0.001):
    """
    Return the value of the policy |pi| up to error tolerance |epsilon|.
    Initialize the computation with |V|.  Note that |V| and |pi| are
    dictionaries.
    """
    # BEGIN_YOUR_CODE (around 7 lines of code expected)
    while True:
        Vps = [computeQ(mdp, V, state, pi[state]) for state in mdp.states]
        count = 0
        i = 0
        for state in mdp.states:
            if abs(Vps[i] - V[state]) <= epsilon:
                count = count + 1
            V[state] = Vps[i]
            i = i + 1
        if count == len(mdp.states):
            return V
    # END_YOUR_CODE

############################################################
# Problem 4.1.3

def computeOptimalPolicy(mdp, V):
    """
    Return the optimal policy based on V(state).
    You might find it handy to call computeQ().  Note that |V| is a
    dictionary.
    """
    # BEGIN_YOUR_CODE (around 4 lines of code expected)
    policy = {}
    for state in V.keys():
        util = - 2^32
        optimalAction = None
        for action in mdp.actions(state):
            q = computeQ(mdp, V, state, action)
            if util < q:
                optimalAction = action
                util = q
        policy[state] = optimalAction
    return policy
    # END_YOUR_CODE

def computeOptimalNoisyPolicy(mdp, V, alpha):
    policy = {}
    for state in V.keys():
        util = - 2^32
        optimalAction = None
        for action in mdp.actions(state):
            q = computeNoisyQ(mdp, V, state, action, alpha)
            if util < q:
                optimalAction = action
                util = q
        policy[state] = optimalAction
    return policy

############################################################
# Problem 4.1.4

class PolicyIteration(util.MDPAlgorithm):
    def solve(self, mdp, epsilon=0.001):
        mdp.computeStates()
        # compute |V| and |pi|, which should both be dicts
        # BEGIN_YOUR_CODE (around 8 lines of code expected)

        # set up
        V = {}
        pi = {}
        for state in mdp.states:
            V[state] = 0
            pi[state] = random.choice(mdp.actions(state))
        # policy interations
        while True:
            V = policyEvaluation(mdp, V, pi, epsilon)
            newPi = computeOptimalPolicy(mdp, V)
            if list(newPi.values()) == list(pi.values()):
                pi = newPi
                break
            pi = newPi
        # END_YOUR_CODE
        self.pi = pi
        self.V = V

############################################################
# Problem 4.1.5

class ValueIteration(util.MDPAlgorithm):
    def solve(self, mdp, epsilon=0.001):
        mdp.computeStates()
        # BEGIN_YOUR_CODE (around 10 lines of code expected)
        V = {}
        pi = {}
        for state in mdp.states:
            V[state] = 0
        states = list(V.keys())
        while True:
            count = 0
            for state in states:
                maxQ = max([computeQ(mdp, V, state, action) for action in mdp.actions(state)])
                if abs(maxQ - V[state]) <= epsilon:
                    count = count + 1
                V[state] = maxQ
            if count == len(V.keys()):
                break
        pi = computeOptimalPolicy(mdp, V)
        # END_YOUR_CODE
        self.pi = pi
        self.V = V

    def solveWithNoise(self, mdp, alpha, epsilon=0.001):
        mdp.computeStates()
        # BEGIN_YOUR_CODE (around 10 lines of code expected)
        V = {}
        pi = {}
        for state in mdp.states:
            V[state] = 0
        states = list(V.keys())
        while True:
            count = 0
            for state in states:
                maxQ = max([computeNoisyQ(mdp, V, state, action, alpha) for action in mdp.actions(state)])
                if abs(maxQ - V[state]) <= epsilon:
                    count = count + 1
                V[state] = maxQ
            if count == len(V.keys()):
                break
        pi = computeOptimalNoisyPolicy(mdp, V, alpha)
        # END_YOUR_CODE
        self.pi = pi
        self.V = V

############################################################
# Problem 4.1.6

# If you decide 1f is true, prove it in writeup.pdf and put "return None" for
# the code blocks below.  If you decide that 1f is false, construct a
# counterexample by filling out this class and returning an alpha value in
# counterexampleAlpha().
class CounterexampleMDP(util.MDP):
    def __init__(self, n=5):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        self.n = n
        # END_YOUR_CODE

    def startState(self):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        return 0
        # END_YOUR_CODE

    # Return set of actions possible from |state|.
    def actions(self, state):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        return [-1, +1]
        # END_YOUR_CODE

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        return [(state, 0.6, 0),
                (min(max(state + action, -self.n), +self.n), 0.4, state)]
        # END_YOUR_CODE

    def discount(self):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        return 0.9
        # END_YOUR_CODE

def counterexampleAlpha():
    # BEGIN_YOUR_CODE (around 1 line of code expected)
    return 1
    '''
    mdp = CounterexampleMDP(2)
    policy = ValueIteration()
    noisyPolicy = ValueIteration()
    policy.solve(mdp)
    noisyPolicy.solveWithNoise(mdp, 1)
    print(policy.V)
    print(noisyPolicy.V)
    '''
    # END_YOUR_CODE

############################################################
# Problem 4.2.1

class BlackjackMDP(util.MDP):
    def __init__(self, cardValues, multiplicity, threshold, peekCost):
        """
        cardValues: list of integers (face values for each card included in the deck)
        multiplicity: single integer representing the number of cards with each face value
        threshold: maximum number of points (i.e. sum of card values in hand) before going bust
        peekCost: how much it costs to peek at the next card
        """
        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.peekCost = peekCost

    # Return the start state.
    # Look closely at this function to see an example of state representation for our Blackjack game.
    # Each state is a tuple with 3 elements:
    #   -- The first element of the tuple is the sum of the cards in the player's hand.
    #   -- If the player's last action was to peek, the second element is the index
    #      (not the face value) of the next card that will be drawn; otherwise, the
    #      second element is None.
    #   -- The third element is a tuple giving counts for each of the cards remaining
    #      in the deck, or None if the deck is empty or the game is over (e.g. when
    #      the user quits or goes bust).
    def startState(self):
        return (0, None, (self.multiplicity,) * len(self.cardValues))  # total, next card (if any), multiplicity for each card

    # Return set of actions possible from |state|.
    # You do not need to modify this function.
    # All logic for dealing with end states should be placed into the succAndProbReward function below.
    def actions(self, state):
        return ['Take', 'Peek', 'Quit']

    def takeCard(self, deckCount, i):
        if i >= len(deckCount) or sum(deckCount) <= 0 or deckCount[i] <= 0:
            return deckCount
        else:
            listDeckCount = list(deckCount)
            listDeckCount[i] = listDeckCount[i] - 1
            return tuple(listDeckCount)

    # Given a |state| and |action|, return a list of (newState, prob, reward) tuples
    # corresponding to the states reachable from |state| when taking |action|.
    # A few reminders:
    # * Indicate a terminal state (after quitting, busting, or running out of cards)
    #   by setting the deck to None.
    # * If |state| is an end state, you should return an empty list [].
    # * When the probability is 0 for a transition to a particular new state,
    #   don't include that state in the list returned by succAndProbReward.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (around 50 lines of code expected)
        # state = (handVal, nextPeekedCardIndex, deckCardCounts)
        successors = []
        # print("At", state, action)
        if state[2] is None or sum(state[2]) == 0:
            print("State[2]: ", state[2])
            # print("Card sum: ", sum(state[2]))
            return successors
        else:
            totalCards = sum(state[2])
            if action == 'Take':
                if state[1] is None: # no peek before
                    for i in range(len(self.cardValues)):
                        if state[2][i] > 0:
                            if totalCards == 1: # draw last card
                                if self.cardValues[i] + state[0] > self.threshold: # last card bust
                                    reward = 0
                                else: # last card normal
                                    reward = self.cardValues[i] + state[0]
                                successors.append(((self.cardValues[i] + state[0], None, None), 1, reward))
                                return successors
                            if self.cardValues[i] + state[0] > self.threshold: # bust
                                successors.append(((self.cardValues[i] + state[0], None, None), state[2][i] / totalCards, 0))
                            else: # normal
                                successors.append(((self.cardValues[i] + state[0], None, self.takeCard(state[2], i)), state[2][i] / totalCards, 0))
                else: # peek before
                    i = state[1]
                    if state[2][i] > 0:
                        if totalCards == 1:
                            if self.cardValues[i] + state[0] > self.threshold: # last card bust
                                reward = 0
                            else: # last card normal
                                reward = self.cardValues[i] + state[0]
                            successors.append(((self.cardValues[i] + state[0], None, None), 1, reward))
                            return successors
                        if self.cardValues[i] + state[0] > self.threshold: # bust
                            successors.append(((self.cardValues[i] + state[0], None, None), 1, 0))
                        else: # normal
                            successors.append(((self.cardValues[i] + state[0], None, self.takeCard(state[2], i)), 1, 0))
            elif action == 'Quit':
                successors.append(((state[0], None, None), 1, state[0]))
            elif action == 'Peek':
                for i in range(len(self.cardValues)):
                    if state[2][i] > 0:
                        successors.append(((state[0], i, state[2]), state[2][i] / totalCards, -self.peekCost))
            return successors
        # END_YOUR_CODE

    def discount(self):
        return 1

############################################################
# Problem 4.2.2

def peekingMDP():
    """
    Return an instance of BlackjackMDP where peeking is the optimal action at
    least 10% of the time.
    """
    # BEGIN_YOUR_CODE (around 2 lines of code expected)
    peekMDP = BlackjackMDP([4,9,11,12,15], 3, 20, 1);
    return peekMDP;
    # END_YOUR_CODE

# counterexampleAlpha()