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

from game import Agent

INF = 9999999

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
        curFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore() - currentGameState.getScore()   #The base of the evaluation function is the difference of the score after taking each action vs before taking the action 

        ghostManhattanDist = []   #Manhattan distances from ghosts
        for ghost in newGhostStates:
            ghostManhattanDist.append(util.manhattanDistance(newPos,ghost.getPosition()))

        foodManhattanDist = []    #Manhattan distances from food
        for foodPosition in newFood.asList():
            foodManhattanDist.append(util.manhattanDistance(newPos,foodPosition))

        
        for distance in ghostManhattanDist:
            score = score + distance     #The further from the ghosts the better
        
        if len(foodManhattanDist) > 0:       
            score = score - min(foodManhattanDist)     #The closer to the closest food the better
        else:
            score = score + 9999      #If in the next action there is no more food, pacman wins so give a huge score boost
    
        if (newPos == currentGameState.getPacmanPosition()):   #Penalty if pacman stops
            score = score - 10
        
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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def maxValue(self,depth,state):
        legalActions = state.getLegalActions(0)
        if (depth > self.depth) or state.isWin() or state.isLose():  #if the required depth is exceeded or the game is over, return the state evaluation
            return self.evaluationFunction(state)
        
        maxEval = -INF

        for action in legalActions:  #find the max value of ghost actions after pacman taking each available action
            successor = state.generateSuccessor(0,action)
            maxEval = max(maxEval , self.minValue(depth,successor,1))
        return maxEval

    def minValue(self,depth,state,agentIndex):
        legalActions = state.getLegalActions(agentIndex)
        if state.isWin() or state.isLose():  #no reason to check the depth since it only changes at when Max plays
            return self.evaluationFunction(state)
        
        minEval = INF
        
        agentCount = state.getNumAgents()
        if agentIndex <= agentCount-2:   #If there are more ghosts after this one
            for action in legalActions:   #find the min value of the actions of the remaining ghosts, after the current ghost takes each available action
                successor = state.generateSuccessor(agentIndex,action)
                minEval = min(minEval, self.minValue(depth,successor,agentIndex+1))
        else:   #If there are no more ghosts after this one then it is pacman's (max) turn
            for action in legalActions:  #find the min value of pacman actions after the ghost takes each available action
                successor = state.generateSuccessor(agentIndex,action)
                minEval = min(minEval, self.maxValue(depth+1,successor))
        
        return minEval
        

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
        bestActionValue = -INF
        bestAction = None
        for action in gameState.getLegalActions(0):  #for every pacman action
            successorState = gameState.generateSuccessor(0,action)   #the next state if pacman takes the current action
            if self.minValue(1,successorState,1) > bestActionValue:   #find the max of the ghost actions
                bestActionValue = self.minValue(1,successorState,1)
                bestAction = action
        return bestAction



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    
    def maxValue(self,depth,state,a,b):
        legalActions = state.getLegalActions(0)
        if (depth > self.depth) or state.isWin() or state.isLose():  #if the required depth is exceeded or the game is over, return the state evaluation
            return self.evaluationFunction(state)
        
        maxEval = -INF

        for action in legalActions:  #find the max value of ghost actions after pacman taking each available action
            successor = state.generateSuccessor(0,action)
            maxEval = max(maxEval , self.minValue(depth,successor,1,a,b))
            if maxEval > b : return maxEval
            a = max(a,maxEval)
        return maxEval

    def minValue(self,depth,state,agentIndex,a,b):
        legalActions = state.getLegalActions(agentIndex)
        if state.isWin() or state.isLose():  #no reason to check the depth since it only changes at when Max plays
            return self.evaluationFunction(state)
        
        minEval = INF
        
        agentCount = state.getNumAgents()
        if agentIndex <= agentCount-2:   #If there are more ghosts after this one
            for action in legalActions:   #find the min value of the actions of the remaining ghosts, after the current ghost takes each available action
                successor = state.generateSuccessor(agentIndex,action)
                minEval = min(minEval, self.minValue(depth,successor,agentIndex+1,a,b))
                if minEval < a : return minEval
                b = min(b,minEval)
        else:   #If there are no more ghosts after this one then it is pacman's (max) turn
            for action in legalActions:  #find the min value of pacman actions after the ghost takes each available action
                successor = state.generateSuccessor(agentIndex,action)
                minEval = min(minEval, self.maxValue(depth+1,successor,a,b))
                if minEval < a : return minEval
                b = min(b,minEval)
        
        return minEval

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        bestActionValue = -INF
        bestAction = None
        a = -INF
        b = INF
        for action in gameState.getLegalActions(0):  #for every pacman action
            successorState = gameState.generateSuccessor(0,action)   #the next state if pacman takes the current action
            if self.minValue(1,successorState,1,a,b) > bestActionValue:   #find the max of the ghost actions
                bestActionValue = self.minValue(1,successorState,1,a,b)
                bestAction = action
            if bestActionValue > b : return bestAction
            a = max(a,bestActionValue) 
        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def maxValue(self,depth,state):
        legalActions = state.getLegalActions(0)
        if (depth > self.depth) or state.isWin() or state.isLose():  #if the required depth is exceeded or the game is over, return the state evaluation
            return self.evaluationFunction(state)
        
        maxEval = -INF

        for action in legalActions:  #find the max value of ghost actions after pacman taking each available action
            successor = state.generateSuccessor(0,action)
            maxEval = max(maxEval , self.chanceValue(depth,successor,1))
        return maxEval

    def chanceValue(self,depth,state,agentIndex):
        legalActions = state.getLegalActions(agentIndex)
        if state.isWin() or state.isLose():  #no reason to check the depth since it only changes at when Max plays
            return self.evaluationFunction(state)
        
        chanceEval = 0
        agentCount = state.getNumAgents()
        if agentIndex <= agentCount-2:   #If there are more ghosts after this one
            for action in legalActions:   #calculate the chance value of the current ghost action plus the next one
                successorState = state.generateSuccessor(agentIndex,action)
                chanceEval = chanceEval + self.chanceValue(depth,successorState,agentIndex+1)
                
        else:   #If there are no more ghosts after this one then it is pacman's (max) turn
            for action in legalActions:  #calculate the chance node's value
                successorState = state.generateSuccessor(agentIndex,action)
                chanceEval = chanceEval + self.maxValue(depth+1,successorState)/len(legalActions)  #probability of a ghost taking each action is 1/len(legalActions)
        
        return chanceEval


    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        bestActionValue = -INF
        bestAction = None
        for action in gameState.getLegalActions(0):  #for every pacman action
            successorState = gameState.generateSuccessor(0,action)   #the next state if pacman takes the current action
            if self.chanceValue(1,successorState,1) > bestActionValue:   #find the max of the ghost actions
                bestActionValue = self.chanceValue(1,successorState,1)
                bestAction = action
        return bestAction

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: 
    Evaluation function that INCREASES the evaluation when pacman:
        -eats food
        -is close to food
        -eats capsule
        -is close to capsule
        -is close to a scared ghost
        -wins (big increase)
    and DECREASES the evaluation when pacman:
        -is close to a non-scared ghost
        -does no progress
        -loses (big decrease)
    """
    "*** YOUR CODE HERE ***"
    score = currentGameState.getScore()  #initial evaluation is the current score so evaluation decreases when pacman does nothing

    pacmanPos = currentGameState.getPacmanPosition()
    foodLeft = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    capsulesLeft = currentGameState.getCapsules()
    
    if len(capsulesLeft) > 0:  #the closer to the closest capsule the better
        capsuleDistances = []
        for capsule in capsulesLeft:
            capsuleDistances.append(util.manhattanDistance(capsule,pacmanPos))
        score = score - min(capsuleDistances)
    
    if len(foodLeft) > 0:  #the closer to the closest food the better
        foodDistances = []
        for food in foodLeft:
            foodDistances.append(util.manhattanDistance(food,pacmanPos))
        score = score - min(foodDistances)
    
    if currentGameState.hasFood(pacmanPos[0],pacmanPos[1]): score = score + 50   #increase score if food is reached
    if currentGameState.isLose(): score = score - 9999    #huge decrease in score if state is loss
    if currentGameState.isWin(): score = score + 9999    #huge increase in score if state is loss
    for capsule in capsulesLeft:    #increase score if capsule is reached
        if pacmanPos == capsule : score = score + 100


    for ghost in ghostStates: 
        if ghost.scaredTimer > 0:   #if ghost is scared then the closer the better
            score = score + util.manhattanDistance(ghost.getPosition(),pacmanPos)
        else:   #if ghost not scared then the closest the worse
            score = score - util.manhattanDistance(ghost.getPosition(),pacmanPos)

    score = score - len(foodLeft)  #the more food left the worse

    return score

# Abbreviation
better = betterEvaluationFunction
