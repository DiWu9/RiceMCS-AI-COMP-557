# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random
import util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        from util import manhattanDistance
        from game import Directions

        foodList = currentGameState.getFood().asList()
        pacFoodDist = [manhattanDistance(newPos, foodPos)
                       for foodPos in foodList]

        stopPenalty = action == Directions.STOP
        distPenalty = min(pacFoodDist)
        dangerPenalty = 0

        foodDiff = currentGameState.getFood().count() - newFood.count()
        pacGhostDist = [manhattanDistance(
            newPos, ghostState.getPosition()) for ghostState in newGhostStates]
        for i in range(len(pacGhostDist)):
            # may encounter ghost and ghost is not scared
            if pacGhostDist[i] < 4 and newScaredTimes[i] == 0:
                dangerPenalty += 10
            else:
                dangerPenalty -= 1

        score = successorGameState.getScore() + foodDiff - distPenalty - \
            dangerPenalty - stopPenalty
        return score


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        score, action = self.value(gameState, self.index, self.depth)
        return action

    def value(self, gameState, agent, depth):

        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState), None
        else:
            n = gameState.getNumAgents()
            actions = gameState.getLegalActions(agent)
            successors = [gameState.generateSuccessor(
                agent, action) for action in actions]
            if agent == self.index:  # pacman
                values = [self.value(successor, agent+1, depth)[0]
                          for successor in successors]
                maxVal = max(values)
                maxAction = actions[values.index(maxVal)]
                return maxVal, maxAction
            elif agent > self.index and agent < n-1:
                values = [self.value(successor, agent+1, depth)[0]
                          for successor in successors]
                minVal = min(values)
                minAction = actions[values.index(minVal)]
                return minVal, minAction
            else:  # last ghost agent
                depth = depth - 1  # last agent is visited, decrement depth
                values = [self.value(successor, self.index, depth)[0]
                          for successor in successors]
                minVal = min(values)
                minAction = actions[values.index(minVal)]
                return minVal, minAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.value(gameState, self.index, self.depth)

    def value(self, gameState, agent, depth):
        alpha = - 2**31  # alpha = -inf
        beta = 2**31 - 1  # beta = inf
        if agent == self.index:
            score, action = self.maxValue(gameState, depth, alpha, beta)
        else:
            score, action = self.minValue(gameState, agent, depth, alpha, beta)
        return action

    def maxValue(self, gameState, depth, alpha, beta):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState), None
        actions = gameState.getLegalActions(self.index)
        bestAction = None
        for action in actions:
            successor = gameState.generateSuccessor(self.index, action)
            value = self.minValue(successor, self.index +
                                  1, depth, alpha, beta)[0]
            if value > alpha:
                bestAction = action
                alpha = value
            if alpha >= beta:
                return beta, None
        return alpha, bestAction

    def minValue(self, gameState, agent, depth, alpha, beta):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState), None
        n = gameState.getNumAgents()
        actions = gameState.getLegalActions(agent)
        bestAction = None
        for action in actions:
            successor = gameState.generateSuccessor(agent, action)
            if agent < n-1:
                value = self.minValue(successor, agent+1,
                                      depth, alpha, beta)[0]
            else:
                value = self.maxValue(successor, depth-1, alpha, beta)[0]
            if value < beta:
                beta = value
                bestAction = action
            if beta <= alpha:
                return alpha, None
        return beta, bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        score, action = self.value(gameState, self.index, self.depth)
        return action

    def value(self, gameState, agent, depth):

        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState), None
        else:
            n = gameState.getNumAgents()
            actions = gameState.getLegalActions(agent)
            successors = [gameState.generateSuccessor(
                agent, action) for action in actions]
            if agent == self.index:  # pacman
                values = [self.value(successor, agent+1, depth)[0]
                          for successor in successors]
                maxVal = max(values)
                maxAction = actions[values.index(maxVal)]
                return maxVal, maxAction
            elif agent > self.index and agent < n-1:
                values = [self.value(successor, agent+1, depth)[0]
                          for successor in successors]
                expectedVal = sum(values)/len(values)
                return expectedVal, None
            else:  # last ghost agent
                depth = depth - 1  # last agent is visited, decrement depth
                values = [self.value(successor, self.index, depth)[0]
                          for successor in successors]
                expectedVal = sum(values)/len(values)
                return expectedVal, None


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: 
    The evaluation function is a linear polynomial that contains the following 7 variables:
        | variable name      | weight |
        | ------------------ | ------ |
        | isWin              |   1000 |
        | numFood            |    -10 |
        | gameScore          |      1 |
        | minPacFoodDist     |     -2 |
        | minPacGhostDist    |      1 |
        | foodFoodDist       |     -2 |
        | scaredBonus        |      5 |
        | ------------------ | ------ |
    """
    "*** YOUR CODE HERE ***"
    # Useful information extracted from a GameState (pacman.py)
    pos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    ghostPositions = currentGameState.getGhostPositions()
    scaredTimes = [
        ghostState.scaredTimer for ghostState in currentGameState.getGhostStates()]

    # FACTORS:
    # factor 1: win(1) / lose(-1) / neither(0)
    isWin = 0
    if (currentGameState.isWin()):
        isWin = 1
    elif (currentGameState.isLose()):
        isWin = -1

    # factor 2: num foods
    numFood = currentGameState.getFood().count()

    # factor 3: game score
    gameScore = currentGameState.getScore()

    # factor 4: distance to the farest and closest food
    # pacFoodDist = [manhattanDistance(pos, foodPos) for foodPos in foodList]
    pacFoodDist = [len(aStarSearch(pos, foodPos, currentGameState))
                   for foodPos in foodList]
    if len(pacFoodDist) == 0:
        maxPacFoodDist = 0
        minPacFoodDist = 0
    else:
        maxPacFoodDist = max(pacFoodDist)
        minPacFoodDist = min(pacFoodDist)

    # factor 5: distance to ghost
    pacGhostDist = [len(aStarSearch(pos, ghostPos, currentGameState))
                    for ghostPos in ghostPositions]
    # pacGhostDist = [manhattanDistance(pos, ghostPos) for ghostPos in ghostPositions]
    minPacGhostDist = min(pacGhostDist)
    if minPacGhostDist > 6:
        minPacGhostDist = 6

    # factor 6: distance between first food and last food
    if len(foodList) < 2:
        foodFoodDist = 0
    else:
        foodFoodDist = len(aStarSearch(
            foodList[0], foodList[len(foodList)-1], currentGameState))

    # factor 7: scared time
    scaredBonus = 0
    for t in scaredTimes:
        if t > 0:
            scaredBonus += 1

    factors = [isWin, numFood, gameScore, maxPacFoodDist,
               minPacFoodDist, minPacGhostDist, foodFoodDist, scaredBonus]

    # WEIGHTS
    wWin = 1000
    wNumFood = -10  # less food gets higher score
    wGameScore = 1
    wMaxPacFoodDist = 0  # don't consider max distance of food
    wMinPacFoodDist = -2
    wPacGhostDist = 1
    if minPacGhostDist <= 6:
        wPacGhostDist = -wPacGhostDist
    wFoodFoodDist = -2
    wScaredBonus = 5
    weights = [wWin, wNumFood, wGameScore, wMaxPacFoodDist,
               wMinPacFoodDist, wPacGhostDist, wFoodFoodDist, wScaredBonus]

    score = sum([factors[i] * weights[i] for i in range(len(factors))])
    # print(pos, score)
    return score


def aStarSearch(pacPos, targetPos, currentGameState):
    """
    aStarSearch from assignment with h = manhattanDistance
    """
    from util import PriorityQueue
    from util import manhattanDistance
    from game import Directions
    directions = [Directions.NORTH, Directions.SOUTH,
                  Directions.EAST, Directions.WEST]

    visited = []
    actions = []
    frontier = PriorityQueue()
    parent_map = {}
    start = pacPos
    frontier.push(start, 0 + manhattanDistance(start, targetPos))
    # no parent, no prev action, f = h
    parent_map[start] = (None, None, 0 + manhattanDistance(start, targetPos))

    while not frontier.isEmpty():
        currPos = frontier.pop()
        f = parent_map[currPos][2]
        visited.append(currPos)
        if currPos == targetPos:
            while currPos != start:
                parent, action, c = parent_map[currPos]
                actions.append(action)
                currPos = parent
            actions.reverse()
            return actions
        else:
            # positions after [NORTH, SOUTH, EAST, WEST]
            successors = [(currPos[0], currPos[1]+1), (currPos[0], currPos[1]-1),
                          (currPos[0]+1, currPos[1]), (currPos[0]-1, currPos[1])]
            for i in range(len(successors)):
                successor = successors[i]
                # invalid state
                if currentGameState.hasWall(successor[0], successor[1]):
                    continue
                else:
                    hNbr = f + 1 + \
                        manhattanDistance(successor, targetPos) - \
                        manhattanDistance(currPos, targetPos)
                    if successor not in visited and successor not in parent_map.keys():
                        frontier.push(successor, hNbr)
                        parent_map[successor] = (currPos, directions[i], hNbr)
                    elif successor in parent_map.keys() and parent_map[successor][2] > hNbr:
                        frontier.update(successor, hNbr)
                        parent_map[successor] = (currPos, directions[i], hNbr)
    return []


# Abbreviation
better = betterEvaluationFunction
