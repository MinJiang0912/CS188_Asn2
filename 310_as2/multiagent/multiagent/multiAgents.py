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
import random, util
from pacman import Actions

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
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        newghostPos = successorGameState.getGhostPositions()
        newCapsules = successorGameState.getCapsules()

        "*** YOUR CODE HERE ***"
        # 1. Food Evaluation
        foodDistances = [util.manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()]
        minfoodDistance = min(foodDistances) if foodDistances else 0
        foodScore = 1.0 / (1.0 + minfoodDistance)

        # 2. Ghost Evaluation
        ghostDistances = [util.manhattanDistance(newPos, ghostPos) for ghostPos in newghostPos]

        scaredGhost = [ghost for ghost in newGhostStates if ghost.scaredTimer > 0]
        nonscaredGhost = [ghost for ghost in newGhostStates if ghost.scaredTimer == 0]

        nonscared_GhostScore = 0
        if nonscaredGhost:
            min_nonscaredGhostDistance = min(ghostDistances[ghostIndex] for ghostIndex, ghost in enumerate(newGhostStates) if ghost.scaredTimer == 0)
            nonscared_GhostScore = -10 / (1.0 + min_nonscaredGhostDistance)

        scared_GhostScore = 0
        if scaredGhost:
            min_scaredGhostDistance = min(ghostDistances[ghostIndex] for ghostIndex, ghost in enumerate(newGhostStates) if ghost.scaredTimer > 0)
            scared_GhostScore = 5.0 / (1.0 + min_scaredGhostDistance)

        # 3. Capsule Evaluation
        capsuleDistances = [util.manhattanDistance(newPos, capsule) for capsule in newCapsules]
        minCapsuleDistance = min(capsuleDistances) if capsuleDistances else 0
        capsuleScore = 15.0 / (1.0 + minCapsuleDistance)

        # 4. Final Score:
        finalScore = successorGameState.getScore() + foodScore + nonscared_GhostScore + scared_GhostScore +capsuleScore

        return finalScore

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
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
        def minimax(state, depth, agentIndex):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state),""
            
            if agentIndex == 0:
                value = -float('inf')
                function = max 
            
            else:
                value = float('inf')
                function = min

            bestAction = ""

            for action in state.getLegalActions(agentIndex):
                sucessor = state.generateSuccessor(agentIndex, action)

                if agentIndex == 0:                             # pacman?
                    newValue, _ = minimax(sucessor, depth, 1)
                
                elif agentIndex == state.getNumAgents() - 1:    # last ghost？
                    newValue, _ = minimax(sucessor, depth-1, 0)
                
                else:                                           # more ghosts?
                    newValue, _ = minimax(sucessor, depth, agentIndex+1)
                

                if (function == max and newValue > value) or (function == min and newValue < value):
                    value, bestAction = newValue, action

            return value, bestAction
        
        _, bestAction = minimax(gameState, self.depth, 0)
        return bestAction
                


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        def alphabeta(state, depth, agentIndex, alpha, beta):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state), ""

            value = -float('inf') if agentIndex == 0 else float('inf')
            bestAction = ""
            function = max if agentIndex == 0 else min

            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                print(f"Considering action {action}, successor state: {successor}")

                if agentIndex == 0:                              # pacman
                    newValue, _ = alphabeta(successor, depth, 1, alpha, beta)
                    if newValue > value:
                        value, bestAction = newValue, action
                    if value > beta:
                        print("Pruning triggered!")
                        break
                    alpha = max(alpha, value)
                    print(f"Pacman: Updating alpha to {alpha}")

                else:  # ghosts
                    if agentIndex == state.getNumAgents() - 1:  # last ghost
                        newValue, _ = alphabeta(successor, depth-1, 0, alpha, beta)
                    else:                                       # other ghosts
                        newValue, _ = alphabeta(successor, depth, agentIndex+1, alpha, beta)
                    
                    if newValue < value:
                        value, bestAction = newValue, action
                    if value < alpha:
                        print("Pruning triggered!")
                        break
                    beta = min(beta, value)
                    print(f"Ghost: Updating beta to {beta}")

            return value, bestAction

        _, bestAction = alphabeta(gameState, self.depth, 0, -float('inf'), float('inf'))
        return bestAction



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
        self.memo = {}  # Initialize memoization dictionary
        _, action = self.maxValue(gameState, 0)
        return action

    def maxValue(self, gameState, depth):
        if (gameState, depth) in self.memo:  # Check if value is memoized
            return self.memo[(gameState, depth)]

        # Terminal test
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState), ""

        maxVal = float("-inf")
        maxAction = None
        for action in gameState.getLegalActions(0):  # 0 is always Pacman
            successor = gameState.generateSuccessor(0, action)
            val, _ = self.expectValue(successor, depth, 1)  # 1 is the first ghost
            if val > maxVal:
                maxVal, maxAction = val, action

        # Store the result in memo before returning
        self.memo[(gameState, depth)] = (maxVal, maxAction)
        return maxVal, maxAction

    def expectValue(self, gameState, depth, agentIndex):
        if (gameState, depth, agentIndex) in self.memo:  # Check if value is memoized
            return self.memo[(gameState, depth, agentIndex)]

        # Terminal test
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState), ""

        expVal = 0
        actions = gameState.getLegalActions(agentIndex)
        prob = 1.0 / len(actions)  # Uniform probability distribution
        for action in actions:
            successor = gameState.generateSuccessor(agentIndex, action)
            if agentIndex == gameState.getNumAgents() - 1:  # If last ghost
                val, _ = self.maxValue(successor, depth + 1)  # Increase depth for next Pacman move
            else:  # If not last ghost
                val, _ = self.expectValue(successor, depth, agentIndex + 1)  # Move to next ghost
            expVal += val * prob  # Expected value is the sum of all values times their probabilities

        # Store the result in memo before returning
        self.memo[(gameState, depth, agentIndex)] = (expVal, "")
        return expVal, ""



def betterEvaluationFunction(currentGameState):
    position = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghosts = currentGameState.getGhostStates()
    scaredGhost = [ghostState.scaredTimer for ghostState in ghosts]
    
    foodList = food.asList()
    foodDistance = [manhattanDistance(position, f) for f in foodList]
    
    ghostLocation = [ghost.getPosition() for ghost in ghosts]
    ghostDistance = [manhattanDistance(position, l) for l in ghostLocation]
    
    remainingFoodCount = len(foodList)
    remainingCapsuleCount = len(currentGameState.getCapsules())
    
    capsuleDistances = [manhattanDistance(position, capsule) for capsule in currentGameState.getCapsules()]
    nearestCapsuleDist = min(capsuleDistances) if capsuleDistances else 0
    
    # Weights for each feature
    weights = {
        "score": 1.0,
        "avg_all_food_distance": -2.0,
        "nearestGhostDist": 2.0,
        "nearestScaredGhostDist": -1.5,
        "remainingFoodCount": -2.0,
        "remainingCapsuleCount": -10.0,
        "nearestCapsuleDist": -5.0,
        "foodNearScaredGhost": -3.0,
        "corneredCapsule": -20.0  # New weight
    }
    
    avg_all_food_distance = sum(foodDistance) / len(foodDistance) if foodDistance else 0
    nearestGhostDist = min(ghostDistance) if ghostDistance else 0
    nearestScaredGhostDist = min(ghostDistance[i] for i, ghost in enumerate(ghosts) if scaredGhost[i] > 0) if any(scaredGhost) else 0
    
    # Prioritize food near scared ghosts
    foodNearScaredGhost = 0
    if any(scaredGhost):
        for i, ghost in enumerate(ghosts):
            if scaredGhost[i] > 0:  # If the ghost is scared
                for foodPos in foodList:
                    # If food is near the scared ghost, increase the priority
                    if manhattanDistance(ghost.getPosition(), foodPos) < 3:
                        foodNearScaredGhost += 1
    
    # Prioritize capsules that are cornered by walls
    corneredCapsule = 0
    for capsule in currentGameState.getCapsules():
        x, y = capsule
        wallCount = sum([
            currentGameState.hasWall(x+1, y),  # Check right
            currentGameState.hasWall(x-1, y),  # Check left
            currentGameState.hasWall(x, y+1),  # Check up
            currentGameState.hasWall(x, y-1)   # Check down
        ])
        # If capsule has at least two walls around it, increase the priority
        if wallCount >= 2:
            corneredCapsule += 1
    
    evaluationValue = (
        weights["score"] * currentGameState.getScore() +
        weights["avg_all_food_distance"] * avg_all_food_distance +
        weights["nearestGhostDist"] / (nearestGhostDist + 1) +
        weights["nearestScaredGhostDist"] / (nearestScaredGhostDist + 1) +
        weights["remainingFoodCount"] * remainingFoodCount +
        weights["remainingCapsuleCount"] * remainingCapsuleCount +
        weights["nearestCapsuleDist"] / (nearestCapsuleDist + 1) +
        weights["foodNearScaredGhost"] * foodNearScaredGhost +
        weights["corneredCapsule"] * corneredCapsule  # Add the new feature
    )
    
    return evaluationValue


better = betterEvaluationFunction