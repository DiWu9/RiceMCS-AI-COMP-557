# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    from util import Stack

    visited = []
    actions = []
    frontier = Stack()
    parent_map = {}
    start = problem.getStartState()
    frontier.push((start, 0, 0))

    while not frontier.isEmpty():
        curr = frontier.pop()
        visited.append(curr[0])
        if problem.isGoalState(curr[0]):
            # build path and return
            while curr[0] != start:
                actions.append(curr[1])
                curr = parent_map[curr[0]]
            actions.reverse()
            return actions
        else:
            for nbr in problem.getSuccessors(curr[0]):
                if nbr[0] not in visited:
                    frontier.push(nbr)
                    parent_map[nbr[0]] = curr
    return []


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from util import Queue

    visited = []
    actions = []
    frontier = Queue()
    parent_map = {}
    start = problem.getStartState()
    frontier.push((start, 0, 0))

    while not frontier.isEmpty():
        curr = frontier.pop()
        visited.append(curr[0])
        if problem.isGoalState(curr[0]):
            # build path
            while curr[0] != start:
                actions.append(curr[1])
                curr = parent_map[curr[0]]
            actions.reverse()
            return actions
        else:
            for nbr in problem.getSuccessors(curr[0]):
                # check not in visited and not in frontier
                if nbr[0] not in visited and nbr[0] not in parent_map.keys():
                    frontier.push(nbr)
                    parent_map[nbr[0]] = curr
    return []


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue

    visited = []
    actions = []
    frontier = PriorityQueue()
    parent_map = {}
    start = problem.getStartState()
    frontier.push(start, 0)
    parent_map[start] = ((start, 0, 0), (start, 0, 0))

    while not frontier.isEmpty():
        curr = parent_map[frontier.pop()][0]
        visited.append(curr[0])
        if problem.isGoalState(curr[0]):
            while curr[0] != start:
                actions.append(curr[1])
                curr = parent_map[curr[0]][1]
            actions.reverse()
            return actions
        else:
            for nbr in problem.getSuccessors(curr[0]):
                # unexplored state
                if nbr[0] not in visited and nbr[0] not in parent_map.keys():
                    nbr = (nbr[0], nbr[1], nbr[2] + curr[2])
                    frontier.push(nbr[0], nbr[2])
                    parent_map[nbr[0]] = (nbr, curr)
                # in frontier but the current path is shorter
                elif nbr[0] in parent_map.keys() and parent_map[nbr[0]][0][2] > nbr[2] + curr[2]:
                    nbr = (nbr[0], nbr[1], nbr[2] + curr[2])
                    frontier.update(nbr[0], nbr[2])
                    parent_map[nbr[0]] = (nbr, curr)
    return []


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue

    visited = []
    actions = []
    frontier = PriorityQueue()
    parent_map = {}
    start = problem.getStartState()
    frontier.push(start, 0 + heuristic(start, problem))
    parent_map[start] = ((start, 0, 0 + heuristic(start, problem)), (start, 0, 0 + heuristic(start, problem)))

    while not frontier.isEmpty():
        curr = parent_map[frontier.pop()][0]
        visited.append(curr[0])
        if problem.isGoalState(curr[0]):
            while curr[0] != start:
                actions.append(curr[1])
                curr = parent_map[curr[0]][1]
            actions.reverse()
            return actions
        else:
            for nbr in problem.getSuccessors(curr[0]):
                # unexplored state
                if nbr[0] not in visited and nbr[0] not in parent_map.keys():
                    nbr = (nbr[0], nbr[1], nbr[2] + curr[2] + heuristic(nbr[0], problem) - heuristic(curr[0], problem))
                    frontier.push(nbr[0], nbr[2])
                    parent_map[nbr[0]] = (nbr, curr)
                # in frontier but the current path is shorter
                elif nbr[0] in parent_map.keys() and parent_map[nbr[0]][0][2] > nbr[2] + curr[2]:
                    nbr = (nbr[0], nbr[1], nbr[2] + curr[2] + heuristic(nbr[0], problem) - heuristic(curr[0], problem))
                    frontier.update(nbr[0], nbr[2])
                    parent_map[nbr[0]] = (nbr, curr)
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
