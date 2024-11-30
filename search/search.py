# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# # Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in search_agents.py).
"""
import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in obj-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def get_start_state(self):
        """
        Returns the start state for the search problem.
        """
        util.raise_not_defined()

    def is_goal_state(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raise_not_defined()

    def get_successors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raise_not_defined()

    def get_cost_of_actions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raise_not_defined()


def tiny_maze_search(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

# def addSuccessors(problem, addCost=True):

class SearchNode:
    def __init__(self, parent, node_info):
        """
            parent: parent SearchNode.

            node_info: tuple with three elements => (coord, action, cost)

            coord: (x,y) coordinates of the node position

            action: Direction of movement required to reach node from
            parent node. Possible values are defined by class Directions from
            game.py

            cost: cost of reaching this node from the starting node.
        """

        self.__state = node_info[0]
        self.action = node_info[1]
        self.cost = node_info[2] if parent is None else node_info[2] + parent.cost
        self.parent = parent

    # The coordinates of a node cannot be modified, se we just define a getter.
    # This allows the class to be hashable.
    @property
    def state(self):
        return self.__state

    def get_path(self):
        path = []
        current_node = self
        while current_node.parent is not None:
            path.append(current_node.action)
            current_node = current_node.parent
        path.reverse()
        return path
    
    # Consider 2 nodes to be equal if their coordinates are equal (regardless of everything else)
    # def __eq__(self, __o: obj) -> bool:
    #     if (type(__o) is SearchNode):
    #         return self.__state == __o.__state
    #     return False

    # # def __hash__(self) -> int:
    # #     return hash(self.__state)

def depth_first_search(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.get_start_state())
    print("Is the start a goal?", problem.is_goal_state(problem.get_start_state()))
    print("Start's successors:", problem.get_successors(problem.get_start_state()))
    """
    "*** YOUR CODE HERE ***"
    stack = util.Stack()  #We create an empty stack.
    frontier=problem.get_start_state() #We get the initial state of problem.
    stack.push([frontier, 0, []]) #We put into stack the initial state, the initial cost and the empty list of actions.

    expandedNodes = [] #This empty list will be used to store the states that we have visited. 

    while stack:
        [n, cost, action] = stack.pop() #We get the node, the cost and the actions that we can do.

        if problem.is_goal_state(n): #Check if n node is the goal state.
            return action

        if not n in expandedNodes: #Check if n is not in expandedNodes
            expandedNodes.append(n) 
            successors = problem.get_successors(n) #We get the sucessors of the node.

            for n_successor, action_sucessor, cost_sucessor in successors:
                if n_successor not in expandedNodes:
                    new_cost = cost + cost_sucessor #We add the cost_sucessor to next node.
                    new_action = action + [action_sucessor]
                    stack.push([n_successor, new_cost, new_action]) # Push unvisited successor states onto the stack
    
    print("Start:", problem.get_start_state())
    print("Is the start a goal?", problem.is_goal_state(problem.get_start_state()))
    print("Start's successors:", problem.get_successors(problem.get_start_state()))

    util.raiseNotDefined() 



def breadth_first_search(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    Queue = util.Queue() #Instead of stack, we use a queue
    frontier=problem.get_start_state()
    Queue.push([frontier, 0, []]) 

    expandedNodes = []

    while Queue:
        [n, cost, action] = Queue.pop()

        if problem.is_goal_state(n):
            return action

        if not n in expandedNodes:
            expandedNodes.append(n)
            successors = problem.get_successors(n)

            for n_successor, action_sucessor, cost_sucessor in successors:
                if n_successor not in expandedNodes:
                    new_cost = cost + cost_sucessor
                    new_action = action + [action_sucessor]
                    Queue.push([n_successor, new_cost, new_action])

    util.raiseNotDefined() 

def uniform_cost_search(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    PQueue = util.PriorityQueue() #Instead of Queue, we use PQueue
    frontier=problem.get_start_state()
    PQueue.push([frontier, 0, []],0) # Push the initial state into the priority queue with cost 0 and an empty action list

    expandedNodes = []

    while PQueue:
        [n, cost, action] = PQueue.pop()

        if problem.is_goal_state(n):
            return action

        if not n in expandedNodes:
            expandedNodes.append(n)
            successors = problem.get_successors(n)

            for n_successor, action_sucessor, cost_sucessor in successors:
                if n_successor not in expandedNodes:
                    new_cost = cost + cost_sucessor
                    new_action = action + [action_sucessor]
                    # Push the successor node into the priority queue with its new cost as the priority
                    PQueue.push([n_successor, new_cost, new_action],new_cost)
    util.raise_not_defined()

def null_heuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def a_star_search(problem, heuristic=null_heuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    PQueue = util.PriorityQueue()
    frontier=problem.get_start_state()
    PQueue.push([frontier, 0, []], 0) # Push the initial state into the priority queue with cost 0 and an empty action list

    expandedNodes = []

    while PQueue:
        [n, cost, action] = PQueue.pop()

        if problem.is_goal_state(n):
            return action

        if not n in expandedNodes:
            expandedNodes.append(n)
            successors = problem.get_successors(n)

            for n_successor, action_sucessor, cost_sucessor in successors:
                if n_successor not in expandedNodes:
                    new_cost = cost + cost_sucessor
                    new_action = action + [action_sucessor]
                    # Push the successor node into the priority queue with its combined cost as the priority
                    PQueue.update([n_successor, new_cost, new_action],new_cost+heuristic(n_successor,problem))
    util.raise_not_defined()

# Abbreviations
bfs = breadth_first_search
dfs = depth_first_search
astar = a_star_search
ucs = uniform_cost_search
