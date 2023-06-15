# All the imports
import copy
import numpy as np 
import random
import matplotlib.pyplot as plt # Graphical library
import time
from sklearn.metrics import mean_squared_error # Mean-squared error function

##################################################### GET CID AND LOGIN
# WARNING: fill in these two functions that will be used by the auto-marking script
# [Action required]
def get_CID():
  return "xxx" # Return your CID (add 0 at the beginning to ensure it is 8 digits long)

def get_login():
  return "xxx" # Return your short imperial login

##################################################### GRAPHICSMAZE
# This class is used ONLY for graphics
# YOU DO NOT NEED to understand it to work on this coursework
class GraphicsMaze(object):

  def __init__(self, shape, locations, default_reward, obstacle_locs, absorbing_locs, absorbing_rewards, absorbing):

    self.shape = shape
    self.locations = locations
    self.absorbing = absorbing

    # Walls
    self.walls = np.zeros(self.shape)
    for ob in obstacle_locs:
      self.walls[ob] = 20

    # Rewards
    self.rewarders = np.ones(self.shape) * default_reward
    for i, rew in enumerate(absorbing_locs):
      self.rewarders[rew] = 10 if absorbing_rewards[i] > 0 else -10

    # Print the map to show it
    self.paint_maps()

  def paint_maps(self):
    """
    Print the Maze topology (obstacles, absorbing states and rewards)
    input: /
    output: /
    """
    plt.figure(figsize=(15,10))
    plt.imshow(self.walls + self.rewarders)
    plt.show()

  def paint_state(self, state):
    """
    Print one state on the Maze topology (obstacles, absorbing states and rewards)
    input: /
    output: /
    """
    states = np.zeros(self.shape)
    states[state] = 30
    plt.figure(figsize=(15,10))
    plt.imshow(self.walls + self.rewarders + states)
    plt.show()

  def draw_deterministic_policy(self, Policy):
    """
    Draw a deterministic policy
    input: Policy {np.array} -- policy to draw (should be an array of values between 0 and 3 (actions))
    output: /
    """
    plt.figure(figsize=(15,10))
    plt.imshow(self.walls + self.rewarders) # Create the graph of the Maze
    for state, action in enumerate(Policy):
      if(self.absorbing[0,state]): # If it is an absorbing state, don't plot any action
        continue
      arrows = [r"$\uparrow$",r"$\rightarrow$", r"$\downarrow$", r"$\leftarrow$"] # List of arrows corresponding to each possible action
      action_arrow = arrows[action] # Take the corresponding action
      location = self.locations[state] # Compute its location on graph
      plt.text(location[1], location[0], action_arrow, ha='center', va='center') # Place it on graph
    plt.show()

  def draw_policy(self, Policy):
    """
    Draw a policy (draw an arrow in the most probable direction)
    input: Policy {np.array} -- policy to draw as probability
    output: /
    """
    deterministic_policy = np.array([np.argmax(Policy[row,:]) for row in range(Policy.shape[0])])
    self.draw_deterministic_policy(deterministic_policy)

  def draw_value(self, Value):
    """
    Draw a policy value
    input: Value {np.array} -- policy values to draw
    output: /
    """
    plt.figure(figsize=(15,10))
    plt.imshow(self.walls + self.rewarders) # Create the graph of the Maze
    for state, value in enumerate(Value):
      if(self.absorbing[0, state]): # If it is an absorbing state, don't plot any value
        continue
      location = self.locations[state] # Compute the value location on graph
      plt.text(location[1], location[0], round(value,2), ha='center', va='center') # Place it on graph
    plt.show()

  def draw_deterministic_policy_grid(self, Policies, title, n_columns, n_lines):
    """
    Draw a grid representing multiple deterministic policies
    input: Policies {np.array of np.array} -- array of policies to draw (each should be an array of values between 0 and 3 (actions))
    output: /
    """
    plt.figure(figsize=(20,8))
    for subplot in range (len(Policies)): # Go through all policies
      ax = plt.subplot(n_columns, n_lines, subplot+1) # Create a subplot for each policy
      ax.imshow(self.walls+self.rewarders) # Create the graph of the Maze
      for state, action in enumerate(Policies[subplot]):
        if(self.absorbing[0,state]): # If it is an absorbing state, don't plot any action
          continue
        arrows = [r"$\uparrow$",r"$\rightarrow$", r"$\downarrow$", r"$\leftarrow$"] # List of arrows corresponding to each possible action
        action_arrow = arrows[action] # Take the corresponding action
        location = self.locations[state] # Compute its location on graph
        plt.text(location[1], location[0], action_arrow, ha='center', va='center') # Place it on graph
      ax.title.set_text(title[subplot]) # Set the title for the graph given as argument
    plt.show()

  def draw_policy_grid(self, Policies, title, n_columns, n_lines):
    """
    Draw a grid representing multiple policies (draw an arrow in the most probable direction)
    input: Policy {np.array} -- array of policies to draw as probability
    output: /
    """
    deterministic_policies = np.array([[np.argmax(Policy[row,:]) for row in range(Policy.shape[0])] for Policy in Policies])
    self.draw_deterministic_policy_grid(deterministic_policies, title, n_columns, n_lines)

  def draw_value_grid(self, Values, title, n_columns, n_lines):
    """
    Draw a grid representing multiple policy values
    input: Values {np.array of np.array} -- array of policy values to draw
    output: /
    """
    plt.figure(figsize=(20,8))
    for subplot in range (len(Values)): # Go through all values
      ax = plt.subplot(n_columns, n_lines, subplot+1) # Create a subplot for each value
      ax.imshow(self.walls+self.rewarders) # Create the graph of the Maze
      for state, value in enumerate(Values[subplot]):
        if(self.absorbing[0,state]): # If it is an absorbing state, don't plot any value
          continue
        location = self.locations[state] # Compute the value location on graph
        plt.text(location[1], location[0], round(value,1), ha='center', va='center') # Place it on graph
      ax.title.set_text(title[subplot]) # Set the title for the graoh given as argument
    plt.show()

##################################################### MAZE
# This class define the Maze environment
class Maze(object):

  # [Action required]
  def __init__(self):
    """
    Maze initialisation.
    input: /
    output: /
    """
    
    # [Action required]
    # Properties set from the CID
    # p = 0.8 + 0.02 * (9-y), where y is 2nd to last digit
    self._prob_success = 0.8 + (0.02 * (9 - int(get_CID()[-2]))) # float
    # gamma = 0.8 + (0.02 * y)
    self._gamma = 0.8 + (0.02 * int(get_CID()[-2])) # float
    # Ri = z mod 4
    self._goal = (int(get_CID()[-1]) % 4) # integer (0 for R0, 1 for R1, 2 for R2, 3 for R3)

    # Build the maze
    self._build_maze()
                              

  # Functions used to build the Maze environment 
  # You DO NOT NEED to modify them
  # Functions used to build the Maze environment 
  # You DO NOT NEED to modify them
  def _build_maze(self):
    """
    Maze initialisation.
    input: /
    output: /
    """

    # Properties of the maze
    self._shape = (13, 10)
    self._obstacle_locs = [
                          (1,0), (1,1), (1,2), (1,3), (1,4), (1,7), (1,8), (1,9), \
                          (2,1), (2,2), (2,3), (2,7), \
                          (3,1), (3,2), (3,3), (3,7), \
                          (4,1), (4,7), \
                          (5,1), (5,7), \
                          (6,5), (6,6), (6,7), \
                          (8,0), \
                          (9,0), (9,1), (9,2), (9,6), (9,7), (9,8), (9,9), \
                          (10,0)
                         ] # Location of obstacles
    self._absorbing_locs = [(2,0), (2,9), (10,1), (12,9)] # Location of absorbing states
    self._absorbing_rewards = [ (500 if (i == self._goal) else -50) for i in range (4) ]
    self._starting_locs = [(0,0), (0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (0,7), (0,8), (0,9)] #Reward of absorbing states
    self._default_reward = -1 # Reward for each action performs in the environment
    self._max_t = 500 # Max number of steps in the environment

    # Actions
    self._action_size = 4
    self._direction_names = ['N','E','S','W'] # Direction 0 is 'N', 1 is 'E' and so on
        
    # States
    self._locations = []
    for i in range (self._shape[0]):
      for j in range (self._shape[1]):
        loc = (i,j) 
        # Adding the state to locations if it is no obstacle
        if self._is_location(loc):
          self._locations.append(loc)
    self._state_size = len(self._locations)

    # Neighbours - each line is a state, ranked by state-number, each column is a direction (N, E, S, W)
    self._neighbours = np.zeros((self._state_size, 4)) 
    
    for state in range(self._state_size):
      loc = self._get_loc_from_state(state)

      # North
      neighbour = (loc[0]-1, loc[1]) # North neighbours location
      if self._is_location(neighbour):
        self._neighbours[state][self._direction_names.index('N')] = self._get_state_from_loc(neighbour)
      else: # If there is no neighbour in this direction, coming back to current state
        self._neighbours[state][self._direction_names.index('N')] = state

      # East
      neighbour = (loc[0], loc[1]+1) # East neighbours location
      if self._is_location(neighbour):
        self._neighbours[state][self._direction_names.index('E')] = self._get_state_from_loc(neighbour)
      else: # If there is no neighbour in this direction, coming back to current state
        self._neighbours[state][self._direction_names.index('E')] = state

      # South
      neighbour = (loc[0]+1, loc[1]) # South neighbours location
      if self._is_location(neighbour):
        self._neighbours[state][self._direction_names.index('S')] = self._get_state_from_loc(neighbour)
      else: # If there is no neighbour in this direction, coming back to current state
        self._neighbours[state][self._direction_names.index('S')] = state

      # West
      neighbour = (loc[0], loc[1]-1) # West neighbours location
      if self._is_location(neighbour):
        self._neighbours[state][self._direction_names.index('W')] = self._get_state_from_loc(neighbour)
      else: # If there is no neighbour in this direction, coming back to current state
        self._neighbours[state][self._direction_names.index('W')] = state

    # Absorbing
    self._absorbing = np.zeros((1, self._state_size))
    for a in self._absorbing_locs:
      absorbing_state = self._get_state_from_loc(a)
      self._absorbing[0, absorbing_state] = 1

    # Transition matrix
    self._T = np.zeros((self._state_size, self._state_size, self._action_size)) # Empty matrix of domension S*S*A
    for action in range(self._action_size):
      for outcome in range(4): # For each direction (N, E, S, W)
        # The agent has prob_success probability to go in the correct direction
        if action == outcome:
          prob = 1 - 3.0 * ((1.0 - self._prob_success) / 3.0) # (theoritically equal to self.prob_success but avoid rounding error and garanty a sum of 1)
        # Equal probability to go into one of the other directions
        else:
          prob = (1.0 - self._prob_success) / 3.0
          
        # Write this probability in the transition matrix
        for prior_state in range(self._state_size):
          # If absorbing state, probability of 0 to go to any other states
          if not self._absorbing[0, prior_state]:
            post_state = self._neighbours[prior_state, outcome] # Post state number
            post_state = int(post_state) # Transform in integer to avoid error
            self._T[prior_state, post_state, action] += prob

    # Reward matrix
    self._R = np.ones((self._state_size, self._state_size, self._action_size)) # Matrix filled with 1
    self._R = self._default_reward * self._R # Set default_reward everywhere
    for i in range(len(self._absorbing_rewards)): # Set absorbing states rewards
      post_state = self._get_state_from_loc(self._absorbing_locs[i])
      self._R[:,post_state,:] = self._absorbing_rewards[i]

    # Creating the graphical Maze world
    self._graphics = GraphicsMaze(self._shape, self._locations, self._default_reward, self._obstacle_locs, self._absorbing_locs, self._absorbing_rewards, self._absorbing)
    
    # Reset the environment
    self.reset()


  def _is_location(self, loc):
    """
    Is the location a valid state (not out of Maze and not an obstacle)
    input: loc {tuple} -- location of the state
    output: _ {bool} -- is the location a valid state
    """
    if (loc[0] < 0 or loc[1] < 0 or loc[0] > self._shape[0]-1 or loc[1] > self._shape[1]-1):
      return False
    elif (loc in self._obstacle_locs):
      return False
    else:
      return True


  def _get_state_from_loc(self, loc):
    """
    Get the state number corresponding to a given location
    input: loc {tuple} -- location of the state
    output: index {int} -- corresponding state number
    """
    return self._locations.index(tuple(loc))


  def _get_loc_from_state(self, state):
    """
    Get the state number corresponding to a given location
    input: index {int} -- state number
    output: loc {tuple} -- corresponding location
    """
    return self._locations[state]

  # Getter functions used only for DP agents
  # You DO NOT NEED to modify them
  def get_T(self):
    return self._T

  def get_R(self):
    return self._R

  def get_absorbing(self):
    return self._absorbing

  # Getter functions used for DP, MC and TD agents
  # You DO NOT NEED to modify them
  def get_graphics(self):
    return self._graphics

  def get_action_size(self):
    return self._action_size

  def get_state_size(self):
    return self._state_size

  def get_gamma(self):
    return self._gamma

  # Functions used to perform episodes in the Maze environment
  def reset(self):
    """
    Reset the environment state to one of the possible starting states
    input: /
    output: 
      - t {int} -- current timestep
      - state {int} -- current state of the envionment
      - reward {int} -- current reward
      - done {bool} -- True if reach a terminal state / 0 otherwise
    """
    self._t = 0
    self._state = self._get_state_from_loc(self._starting_locs[random.randrange(len(self._starting_locs))])
    self._reward = 0
    self._done = False
    return self._t, self._state, self._reward, self._done

  def step(self, action):
    """
    Perform an action in the environment
    input: action {int} -- action to perform
    output: 
      - t {int} -- current timestep
      - state {int} -- current state of the envionment
      - reward {int} -- current reward
      - done {bool} -- True if reach a terminal state / 0 otherwise
    """

    # If environment already finished, print an error
    if self._done or self._absorbing[0, self._state]:
      print("Please reset the environment")
      return self._t, self._state, self._reward, self._done

    # Drawing a random number used for probaility of next state
    probability_success = random.uniform(0,1)

    # Look for the first possible next states (so get a reachable state even if probability_success = 0)
    new_state = 0
    while self._T[self._state, new_state, action] == 0: 
      new_state += 1
    assert self._T[self._state, new_state, action] != 0, "Selected initial state should be probability 0, something might be wrong in the environment."

    # Find the first state for which probability of occurence matches the random value
    total_probability = self._T[self._state, new_state, action]
    while (total_probability < probability_success) and (new_state < self._state_size-1):
     new_state += 1
     total_probability += self._T[self._state, new_state, action]
    assert self._T[self._state, new_state, action] != 0, "Selected state should be probability 0, something might be wrong in the environment."
    
    # Setting new t, state, reward and done
    self._t += 1
    self._reward = self._R[self._state, new_state, action]
    self._done = self._absorbing[0, new_state] or self._t > self._max_t
    self._state = new_state
    return self._t, self._state, self._reward, self._done

##################################################### DP
# This class define the Dynamic Programing agent 
class DP_agent(object):

  # [Action required]
  # WARNING: make sure this function can be called by the auto-marking script
  def solve(self, env):
    """
    Solve a given Maze environment using Dynamic Programming
    input: env {Maze object} -- Maze to solve
    output: 
      - policy {np.array} -- Optimal policy found to solve the given Maze environment 
      - V {np.array} -- Corresponding value function 
    """
    
    # Initialisation (can be edited)
    policy = np.zeros((env.get_state_size(), env.get_action_size())) 
    V = np.zeros(env.get_state_size())

    #### 
    # Add your code here
    # WARNING: for this agent only, you are allowed to access env.get_T(), env.get_R() and env.get_absorbing()
    ####

    # The decision here is to use VALUE ITERATION, rather than policy iteration, since we see that Value Iteration is equivalent to having one
    # backup sweep rather than multiple, as in the case of policy iteration. In Policy Iteration, we have two main areas of computation: the first
    # is policy evaluation, in which we run through all the states and evaluate the value function as according to a specific policy, and the 
    # second is policy improvement/iteration, in which we continue in our iteration over every single state to find the action that maximizes
    # the value function for that state (thus the ideal policy), continued until we reach a stable policy (convergence). In value iteration,
    # on the other hand, we have one main form of computation, which is the actual value function evaluation. Here, rather than evaluating the
    # value function as according to a specific policy, we basically iterate over all actions, rather than just those defined by a policy,
    # and choose the value function that corresponds to the action that gives the maximum possible value. 

    # First, we define a threshold, a delta, and a copy of the V function that we have
    # The threshold is chosen arbitrarily as according to how accurate we want to be, where in this case we want to make sure that we have a
    # maximum difference between the previous V and current V of 0.0001.
    threshold = 0.0001
    gamma = env.get_gamma()
    delta = threshold
    Vnew = np.copy(V)
    # Storing the transition matrix, and R matrix to access later
    T = env.get_T()
    R = env.get_R()

    # Get the absorbing states - FOR EDGE CASE
    absorbing = env.get_absorbing()

    ########## VALUE ITERATION LOOP
    # While the delta is greater than or equal to the threshold
    while delta >= threshold:
      # We set delta to 0, and change it every time we calculate the difference between the V's
      delta = 0

      # For every single state that we have
      for state in range(env.get_state_size()):
        # Ensure that we are not at an absorbing state - FOR EDGE CASE, we don't want to check them
        if not absorbing[0, state]:
          
          # We begin the value iteration function, in which we calculate the Q value, which is defined as, for each state:
          # Q = (for every next state/reward, summation(Transition * (reward + gamma*(V[next_state]))))
          # Thus, Q needs to first be an array of size 4, since we have 4 actions
          Q = np.zeros(4)
          # For every next state, we do exactly what is mentioned above in the equation
          for next_state in range(env.get_state_size()):
            # We calculate Q here, as according to the equation
            Q = Q + (T[state,next_state,:] * (R[state,next_state,:] + gamma*V[next_state]))
          
          # Once we have the Q, we then find the maximum value and set that to be the value in Vnew[state]
          Vnew[state] = np.max(Q)

          # Set the value of delta to be the maximum of delta and the difference between the V's for this state
          delta = max(delta, np.abs(Vnew[state] - V[state]))

          # Set the value of V for this state to the value of Vnew for this state to iterate
          V[state] = Vnew[state]
      
    # Once we're out of the threshold loop for delta, we can now set the policy for each state to be the action that gives the maximum value
    # of the value function. To do this, we again iterate to find the best Q
    for state in range(env.get_state_size()):
      # Set the value of Q to be 0's of size 4 originally
      Q = np.zeros(4)
      # Find the value of Q for this state, over all the next states
      for next_state in range(env.get_state_size()):
        Q = Q + (T[state,next_state,:] * (R[state,next_state,:] + gamma*V[next_state]))
      
      # The policy for this state is going to be at the position where we have the max(Q), set to probability of 1
      policy[state, np.argmax(Q)] = 1

    return policy, V

##################################################### MC
# This class define the Monte-Carlo agent
class MC_agent(object):
  
  def solve(self, env):
    """
    Solve a given Maze environment using Monte Carlo learning
    input: env {Maze object} -- Maze to solve
    output: 
      - policy {np.array} -- Optimal policy found to solve the given Maze environment 
      - values {list of np.array} -- List of successive value functions for each episode 
      - total_rewards {list of float} -- Corresponding list of successive total non-discounted sum of reward for each episode 
    """

    # Initialisation (can be edited)
    Q = np.random.rand(env.get_state_size(), env.get_action_size()) 
    V = np.zeros(env.get_state_size())
    policy = np.zeros((env.get_state_size(), env.get_action_size())) 
    # We initialize the policy to be an epsilon-soft policy, in which all actions have a probability of epsilon/number of actions to be selected
    # Define epsilon to be 0.8
    epsilon = 0.8
    # In general, for all states and all actions, epsilon-soft gives them probability epsilon/cardinality(A) or 1 - epsilon + epsilon/cardinality(A)
    # As an initial policy, we set that for all states, every action has an equal probability of being run
    policy[:,:] = 0.25
    values = [V]
    total_rewards = []
    # Define gamma 
    gamma = env.get_gamma()
    # Define learning rate
    learningRate = 0.1
    # Create a statesSeen to calculate V with it
    statesSeen = set()

    #### 
    # Add your code here
    # WARNING: this agent only has access to env.reset() and env.step()
    # You should not use env.get_T(), env.get_R() or env.get_absorbing() to compute any value
    ####

    # We try first the on-policy first-visit MC control for epsilon-soft policies. Here, what we do is generate an arbitrarily large number of
    # episodes, where for each episode we start using reset(), we then iterate by taking steps in the direction of our policy to generate the
    # episode itself. This is repeated until we either reach an absorbing state, or take 500 steps.
    ################### PHASE 0 - EPISODE GENERATION
    # Define the number of episodes that we want to generate
    numberOfEpisodes = 1000
    # Set the random seed
    np.random.seed(0)

    # While we have less than numberOfEpisodes valid episodes
    for episode in range(numberOfEpisodes):
      # The epsilon is decaying
      epsilon = (1 - episode/numberOfEpisodes)
      # Define the list that will contain all the information/steps of this episode. Note that this will be in the form of a list of tuples, where
      # each tuple will contain (timeStep, state, action, reward, done). Note that we define it within the episodes loop, as we want to reset it every ep
      episodeSteps = []
      # Reset to get the timestep, state, starting state, action, reward, and done boolean
      timeStep, state, reward, done = env.reset()
      # Iterate until ending criteria - either reach absorbing state (done == True), or 500 steps (timeStep >= 500)
      while (done != True):
        # For every episode, we want to take a step in the direction of the policy - we do this randomly with probability given by the policy
        # We can use the np.random.choice() function for this, where the choice must be in the range of number of actions that we have, and the
        # probability is defined by the entry of the action probabilities for that given state
        action = np.random.choice(env.get_action_size(), p=policy[state, :])

        # Append the step to the episodeSteps list
        episodeSteps.append((timeStep, state, action, reward, done))

        # Take a step in the direction of the action defined by the policy
        timeStep, state, reward, done = env.step(action)
        
      # Append the last reward to the list of episodeSteps, where the last step is basically gonna be the TERMINAL STATE, with the reward of
      # that, and no action associated with it
      episodeSteps.append((timeStep, state, None, reward, done)) 
      
      ################### PHASE 1 - LOOPING OVER EPISODE
      # Now that we've generated the episode, the next step is to iterate over the steps defined in the episode, where we define the actual
      # on-policy first-visit MC policy
      # Note that we create two returns values, one that is going to be used discounted, and one undiscounted
      discountedReturns = 0
      undiscountedReturns = 0
      
      # NOTE that we only loop IF the last state was valid
      # For every step (tuple) of the episodeSteps list. Notice that we reverse the order, so that we go from the end to the beginning
      for episodeStep in reversed(list(episodeSteps[:len(episodeSteps) - 1])):
        # The discounted returns is going to be gamme*discountedReturns + reward of next step (technically the same as the reward in
        # the current tuple). Undiscounted is the same without gamma
        # Extract the state, action and reward from each episode tuple
        timeStep = episodeStep[0]
        state = episodeStep[1]
        action = episodeStep[2]
        rewardPlus1 = episodeSteps[timeStep + 1][3]

        # Add the state to the statesSeen set
        statesSeen.add(state)
        
        discountedReturns = gamma*discountedReturns + rewardPlus1
        undiscountedReturns = undiscountedReturns + rewardPlus1

        # Check if the current state-action pair for the current episodeStep appeared in any of the previous tuples before this in the
        # episodeSteps list
        # Get all the visited states and actions
        previousStates = [tup[1] for tup in episodeSteps[:timeStep - 1]]
        previousActions = [tup[2] for tup in episodeSteps[:timeStep - 1]]
        # Make this into a list of tuples
        prevStateActionPair = list(map(lambda x, y: (x, y), previousStates, previousActions))
        # Get current state, action as a tuple
        currentStateActionPair = (state, action)
        # Check if the current episode's state or action is in the visitedStates/Actions list until current index
        if (currentStateActionPair not in prevStateActionPair):
          # Find the qValue for the current state-action pair as the average as an incremental one
          Q[state, action] =  Q[state, action] + learningRate*(discountedReturns - Q[state, action])
          # Find the index of the action that gives the largest value
          bestAction = np.argmax(Q[state,:])
          # For all actions in the state
          for action in range(env.get_action_size()):
            # Update the policy according to the epsilon-greedy policy
            # If current action index is the same as that of the best action
            if (action == bestAction):
              # Place more emphasize on this action, make policy have a greater value
              policy[state, action] = 1 - epsilon + (epsilon/env.get_action_size())
            else:
              policy[state, action] = epsilon/env.get_action_size()
            
      # At the end of the episode, we want to find the value function for every state we've been in
      newV = np.zeros(env.get_state_size())
      newV = copy.deepcopy(V)
      for state in statesSeen:
        newV[state] = 0
        # For every possible action
        for action in range(env.get_action_size()):
          newV[state] += policy[state, action] * Q[state, action]

      # Copy V to a list
      values.append(np.copy(newV))

      # Now that we're finished with the episode, we want to append the undiscounted reward
      total_rewards.append(undiscountedReturns)
        
    return policy, values, total_rewards

##################################################### TD
# This class define the Temporal-Difference agent
class TD_agent(object):

  def solve(self, env):
    """
    Solve a given Maze environment using Temporal Difference learning
    input: env {Maze object} -- Maze to solve
    output: 
      - policy {np.array} -- Optimal policy found to solve the given Maze environment 
      - values {list of np.array} -- List of successive value functions for each episode 
      - total_rewards {list of float} -- Corresponding list of successive total non-discounted sum of reward for each episode 
    """

    # Initialisation (can be edited)
    Q = np.random.rand(env.get_state_size(), env.get_action_size()) 
    V = np.zeros(env.get_state_size())
    policy = np.zeros((env.get_state_size(), env.get_action_size())) 
    # We initialize the policy to be an epsilon-soft policy, in which all actions have a probability of epsilon/number of actions to be selected
    # Define epsilon to be 1
    epsilon = 1
    # Define for all states and all actions, epsilon-soft gives them probability epsilon/cardinality(A) or 1 - epsilon + epsilon/cardinality(A)
    # As an initial policy, we set that for all states, the first action will have the highest probability of being chosen
    # policy[:, 0] = 1 - epsilon + epsilon/env.get_action_size()
    # policy[:, 1:] = epsilon / env.get_action_size()
    policy[:,:] = 0.25
    values = [V]
    total_rewards = []
    # Define gamma 
    gamma = env.get_gamma()
    # Define learning rate
    learningRate = 0.2
    # Define set of states seen in episode
    statesSeen = set()

    #### 
    # Add your code here
    # WARNING: this agent only has access to env.reset() and env.step()
    # You should not use env.get_T(), env.get_R() or env.get_absorbing() to compute any value
    ####

    # In off-policy TD Control, the idea is that we initialize Q, S and set the Q for all terminal states to 0. Then, we initialize the first
    # state. In a loop, we then choose an action A based on our state, take the action to get the reward and nextState, then update Q as
    # Q(s|a) = Q(s|a) + LR*(reward + gamma*maxQ(s'|a) over all a - Q(s|a))
    # Then set the state to the next state

    ################### PHASE 0 - EPISODE GENERATION
    # Define the number of episodes that we want to generate
    numberOfEpisodes = 1000
    # For number of episodes that we have
    for episode in range(numberOfEpisodes):
      # Define epsilon to be decaying
      epsilon = (1 - episode/numberOfEpisodes)
      # Define the list that will contain all the information/steps of this episode. Note that this will be in the form of a list of tuples, where
      # each tuple will contain (timeStep, state, action, reward, done). Note that we define it within the episodes loop, as we want to reset it every ep
      episodeSteps = []
      # Setting the value of undiscounted reward to 0 for every episode
      undiscountedRewards = 0

      # Reset to get the timestep, state, starting state, action, reward, and done boolean
      timeStep, state, reward, done = env.reset()

      # Append state to set of states for this episode
      statesSeen.add(state)

      # Update the undiscounted rewards
      undiscountedRewards = undiscountedRewards + reward

      # Iterate until ending criteria - either reach absorbing state (done == True), or 500 steps (timeStep >= 500)
      while (done != True):
        
        # Find the action from the state
        action = np.random.choice(env.get_action_size(), p=policy[state, :])
        # Append the state,action pair to the episodeSteps
        episodeSteps.append((timeStep, state, action, reward, done))

        # Take a step in the direction of the action defined by the policy
        timeStep, nextState, reward, done = env.step(action)
        
        # Find the Q(s|a), this time with the maximum value over all actions for the next state
        Q[state, action] = Q[state, action] + learningRate*(reward + gamma*np.max(Q[nextState,:]) - Q[state, action])
        
        # Update the undiscounted rewards
        undiscountedRewards = undiscountedRewards + reward

        # Append state to set of states for this episode
        statesSeen.add(nextState)
        
        # Find the index of the action that gives the largest value
        bestAction = np.argmax(Q[state,:])
        # For all actions in the state
        for action in range(env.get_action_size()):
          # Update the policy according to the epsilon-greedy policy
          # If current action index is the same as that of the best action
          if (action == bestAction):
            # Place more emphasize on this action, make policy have a greater value
            policy[state, action] = 1 - epsilon + (epsilon/env.get_action_size())
          else:
            policy[state, action] = epsilon/env.get_action_size()
          
        # Update the next state
        state = nextState
      
      # At the end of the episode, we want to find the value function for every state we've been in
      newV = np.zeros(env.get_state_size())
      newV = copy.deepcopy(V)
      for state in statesSeen:
        newV[state] = 0
        # For every possible action
        for action in range(env.get_action_size()):
          newV[state] += policy[state, action] * Q[state, action]

      # Copy V to a list
      values.append(np.copy(newV))

      # Append the undiscountedRewards to the total_rewards
      total_rewards.append(undiscountedRewards)

    return policy, values, total_rewards

# ##################################################### EXTRAS USED FOR REPORT GENERATION
# if __name__ == '__main__':
#   ###################################### IMPORTS
#   import copy
#   import numpy as np 
#   import random
#   import matplotlib.pyplot as plt # Graphical library
#   import time
#   from sklearn.metrics import mean_squared_error # Mean-squared error function

#   ###################################### CID AND LOGIN
#   # WARNING: fill in these two functions that will be used by the auto-marking script
#   # [Action required]
#   def get_CID():
#     return "xxx" # Return your CID (add 0 at the beginning to ensure it is 8 digits long)

#   def get_login():
#     return "xxx" # Return your short imperial login
  
#   ###################################### GRAPHICS CLASS
#   # This class is used ONLY for graphics
#   # YOU DO NOT NEED to understand it to work on this coursework
#   class GraphicsMaze(object):

#     def __init__(self, shape, locations, default_reward, obstacle_locs, absorbing_locs, absorbing_rewards, absorbing):

#       self.shape = shape
#       self.locations = locations
#       self.absorbing = absorbing

#       # Walls
#       self.walls = np.zeros(self.shape)
#       for ob in obstacle_locs:
#         self.walls[ob] = 20

#       # Rewards
#       self.rewarders = np.ones(self.shape) * default_reward
#       for i, rew in enumerate(absorbing_locs):
#         self.rewarders[rew] = 10 if absorbing_rewards[i] > 0 else -10

#       # Print the map to show it
#       self.paint_maps()

#     def paint_maps(self):
#       """
#       Print the Maze topology (obstacles, absorbing states and rewards)
#       input: /
#       output: /
#       """
#       plt.figure(figsize=(15,10))
#       plt.imshow(self.walls + self.rewarders)
#       plt.show()

#     def paint_state(self, state):
#       """
#       Print one state on the Maze topology (obstacles, absorbing states and rewards)
#       input: /
#       output: /
#       """
#       states = np.zeros(self.shape)
#       states[state] = 30
#       plt.figure(figsize=(15,10))
#       plt.imshow(self.walls + self.rewarders + states)
#       plt.show()

#     def draw_deterministic_policy(self, Policy, title):
#       """
#       Draw a deterministic policy
#       input: Policy {np.array} -- policy to draw (should be an array of values between 0 and 3 (actions))
#       output: /
#       """
#       plt.figure(figsize=(15,10))
#       plt.imshow(self.walls + self.rewarders) # Create the graph of the Maze
#       for state, action in enumerate(Policy):
#         if(self.absorbing[0,state]): # If it is an absorbing state, don't plot any action
#           continue
#         arrows = [r"$\uparrow$",r"$\rightarrow$", r"$\downarrow$", r"$\leftarrow$"] # List of arrows corresponding to each possible action
#         action_arrow = arrows[action] # Take the corresponding action
#         location = self.locations[state] # Compute its location on graph
#         plt.text(location[1], location[0], action_arrow, ha='center', va='center') # Place it on graph
#       plt.title(title)
#       plt.show()

#     def draw_policy(self, Policy, title):
#       """
#       Draw a policy (draw an arrow in the most probable direction)
#       input: Policy {np.array} -- policy to draw as probability
#       output: /
#       """
#       deterministic_policy = np.array([np.argmax(Policy[row,:]) for row in range(Policy.shape[0])])
#       self.draw_deterministic_policy(deterministic_policy, title)

#     def draw_value(self, Value, title):
#       """
#       Draw a policy value
#       input: Value {np.array} -- policy values to draw
#       output: /
#       """
#       plt.figure(figsize=(15,10))
#       plt.imshow(self.walls + self.rewarders) # Create the graph of the Maze
#       for state, value in enumerate(Value):
#         if(self.absorbing[0, state]): # If it is an absorbing state, don't plot any value
#           continue
#         location = self.locations[state] # Compute the value location on graph
#         plt.text(location[1], location[0], round(value,2), ha='center', va='center') # Place it on graph
#       plt.title(title)
#       plt.show()

#     def draw_deterministic_policy_grid(self, Policies, title, n_columns, n_lines):
#       """
#       Draw a grid representing multiple deterministic policies
#       input: Policies {np.array of np.array} -- array of policies to draw (each should be an array of values between 0 and 3 (actions))
#       output: /
#       """
#       plt.figure(figsize=(20,8))
#       for subplot in range (len(Policies)): # Go through all policies
#         ax = plt.subplot(n_columns, n_lines, subplot+1) # Create a subplot for each policy
#         ax.imshow(self.walls+self.rewarders) # Create the graph of the Maze
#         for state, action in enumerate(Policies[subplot]):
#           if(self.absorbing[0,state]): # If it is an absorbing state, don't plot any action
#             continue
#           arrows = [r"$\uparrow$",r"$\rightarrow$", r"$\downarrow$", r"$\leftarrow$"] # List of arrows corresponding to each possible action
#           action_arrow = arrows[action] # Take the corresponding action
#           location = self.locations[state] # Compute its location on graph
#           plt.text(location[1], location[0], action_arrow, ha='center', va='center') # Place it on graph
#         ax.title.set_text(title[subplot]) # Set the title for the graph given as argument
#       plt.show()

#     def draw_policy_grid(self, Policies, title, n_columns, n_lines):
#       """
#       Draw a grid representing multiple policies (draw an arrow in the most probable direction)
#       input: Policy {np.array} -- array of policies to draw as probability
#       output: /
#       """
#       deterministic_policies = np.array([[np.argmax(Policy[row,:]) for row in range(Policy.shape[0])] for Policy in Policies])
#       self.draw_deterministic_policy_grid(deterministic_policies, title, n_columns, n_lines)

#     def draw_value_grid(self, Values, title, n_columns, n_lines):
#       """
#       Draw a grid representing multiple policy values
#       input: Values {np.array of np.array} -- array of policy values to draw
#       output: /
#       """
#       plt.figure(figsize=(20,8))
#       for subplot in range (len(Values)): # Go through all values
#         ax = plt.subplot(n_columns, n_lines, subplot+1) # Create a subplot for each value
#         ax.imshow(self.walls+self.rewarders) # Create the graph of the Maze
#         for state, value in enumerate(Values[subplot]):
#           if(self.absorbing[0,state]): # If it is an absorbing state, don't plot any value
#             continue
#           location = self.locations[state] # Compute the value location on graph
#           plt.text(location[1], location[0], round(value,1), ha='center', va='center', fontsize=7) # Place it on graph
#         ax.title.set_text(title[subplot]) # Set the title for the graoh given as argument
#       plt.show()

#   ###################################### MAZE CLASS
#   # This class defineS the Maze environment
#   class Maze(object):

#     # [Action required]
#     def __init__(self):
#       """
#       Maze initialisation.
#       input: /
#       output: /
#       """
      
#       # [Action required]
#       # Properties set from the CID
#       # p = 0.8 + 0.02 * (9-y), where y is 2nd to last digit
#       self._prob_success = 0.8 + (0.02 * (9 - int(get_CID()[-2]))) # float
#       # gamma = 0.8 + (0.02 * y)
#       self._gamma = 0.8 + (0.02 * int(get_CID()[-2])) # float
#       # Ri = z mod 4
#       self._goal = (int(get_CID()[-1]) % 4) # integer (0 for R0, 1 for R1, 2 for R2, 3 for R3)

#       # Build the maze
#       self._build_maze()
                                

#     # Functions used to build the Maze environment 
#     # You DO NOT NEED to modify them
#     def _build_maze(self):
#       """
#       Maze initialisation.
#       input: /
#       output: /
#       """

#       # Properties of the maze
#       self._shape = (13, 10)
#       self._obstacle_locs = [
#                             (1,0), (1,1), (1,2), (1,3), (1,4), (1,7), (1,8), (1,9), \
#                             (2,1), (2,2), (2,3), (2,7), \
#                             (3,1), (3,2), (3,3), (3,7), \
#                             (4,1), (4,7), \
#                             (5,1), (5,7), \
#                             (6,5), (6,6), (6,7), \
#                             (8,0), \
#                             (9,0), (9,1), (9,2), (9,6), (9,7), (9,8), (9,9), \
#                             (10,0)
#                           ] # Location of obstacles
#       self._absorbing_locs = [(2,0), (2,9), (10,1), (12,9)] # Location of absorbing states
#       self._absorbing_rewards = [ (500 if (i == self._goal) else -50) for i in range (4) ] #Reward of absorbing states
#       self._starting_locs = [(0,0), (0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (0,7), (0,8), (0,9)] 
#       self._default_reward = -1 # Reward for each action performs in the environment
#       self._max_t = 500 # Max number of steps in the environment

#       # Actions
#       self._action_size = 4
#       self._direction_names = ['N','E','S','W'] # Direction 0 is 'N', 1 is 'E' and so on
          
#       # States
#       self._locations = []
#       for i in range (self._shape[0]):
#         for j in range (self._shape[1]):
#           loc = (i,j) 
#           # Adding the state to locations if it is no obstacle
#           if self._is_location(loc):
#             self._locations.append(loc)
#       self._state_size = len(self._locations)

#       # Neighbours - each line is a state, ranked by state-number, each column is a direction (N, E, S, W)
#       self._neighbours = np.zeros((self._state_size, 4)) 
      
#       for state in range(self._state_size):
#         loc = self._get_loc_from_state(state)

#         # North
#         neighbour = (loc[0]-1, loc[1]) # North neighbours location
#         if self._is_location(neighbour):
#           self._neighbours[state][self._direction_names.index('N')] = self._get_state_from_loc(neighbour)
#         else: # If there is no neighbour in this direction, coming back to current state
#           self._neighbours[state][self._direction_names.index('N')] = state

#         # East
#         neighbour = (loc[0], loc[1]+1) # East neighbours location
#         if self._is_location(neighbour):
#           self._neighbours[state][self._direction_names.index('E')] = self._get_state_from_loc(neighbour)
#         else: # If there is no neighbour in this direction, coming back to current state
#           self._neighbours[state][self._direction_names.index('E')] = state

#         # South
#         neighbour = (loc[0]+1, loc[1]) # South neighbours location
#         if self._is_location(neighbour):
#           self._neighbours[state][self._direction_names.index('S')] = self._get_state_from_loc(neighbour)
#         else: # If there is no neighbour in this direction, coming back to current state
#           self._neighbours[state][self._direction_names.index('S')] = state

#         # West
#         neighbour = (loc[0], loc[1]-1) # West neighbours location
#         if self._is_location(neighbour):
#           self._neighbours[state][self._direction_names.index('W')] = self._get_state_from_loc(neighbour)
#         else: # If there is no neighbour in this direction, coming back to current state
#           self._neighbours[state][self._direction_names.index('W')] = state

#       # Absorbing
#       self._absorbing = np.zeros((1, self._state_size))
#       for a in self._absorbing_locs:
#         absorbing_state = self._get_state_from_loc(a)
#         self._absorbing[0, absorbing_state] = 1

#       # Transition matrix
#       self._T = np.zeros((self._state_size, self._state_size, self._action_size)) # Empty matrix of domension S*S*A
#       for action in range(self._action_size):
#         for outcome in range(4): # For each direction (N, E, S, W)
#           # The agent has prob_success probability to go in the correct direction
#           if action == outcome:
#             prob = 1 - 3.0 * ((1.0 - self._prob_success) / 3.0) # (theoritically equal to self.prob_success but avoid rounding error and garanty a sum of 1)
#           # Equal probability to go into one of the other directions
#           else:
#             prob = (1.0 - self._prob_success) / 3.0
            
#           # Write this probability in the transition matrix
#           for prior_state in range(self._state_size):
#             # If absorbing state, probability of 0 to go to any other states
#             if not self._absorbing[0, prior_state]:
#               post_state = self._neighbours[prior_state, outcome] # Post state number
#               post_state = int(post_state) # Transform in integer to avoid error
#               self._T[prior_state, post_state, action] += prob

#       # Reward matrix
#       self._R = np.ones((self._state_size, self._state_size, self._action_size)) # Matrix filled with 1
#       self._R = self._default_reward * self._R # Set default_reward everywhere
#       for i in range(len(self._absorbing_rewards)): # Set absorbing states rewards
#         post_state = self._get_state_from_loc(self._absorbing_locs[i])
#         self._R[:,post_state,:] = self._absorbing_rewards[i]

#       # Creating the graphical Maze world
#       self._graphics = GraphicsMaze(self._shape, self._locations, self._default_reward, self._obstacle_locs, self._absorbing_locs, self._absorbing_rewards, self._absorbing)
      
#       # Reset the environment
#       self.reset()


#     def _is_location(self, loc):
#       """
#       Is the location a valid state (not out of Maze and not an obstacle)
#       input: loc {tuple} -- location of the state
#       output: _ {bool} -- is the location a valid state
#       """
#       if (loc[0] < 0 or loc[1] < 0 or loc[0] > self._shape[0]-1 or loc[1] > self._shape[1]-1):
#         return False
#       elif (loc in self._obstacle_locs):
#         return False
#       else:
#         return True


#     def _get_state_from_loc(self, loc):
#       """
#       Get the state number corresponding to a given location
#       input: loc {tuple} -- location of the state
#       output: index {int} -- corresponding state number
#       """
#       return self._locations.index(tuple(loc))


#     def _get_loc_from_state(self, state):
#       """
#       Get the state number corresponding to a given location
#       input: index {int} -- state number
#       output: loc {tuple} -- corresponding location
#       """
#       return self._locations[state]

#     # Getter functions used only for DP agents
#     # You DO NOT NEED to modify them
#     def get_T(self):
#       return self._T

#     def get_R(self):
#       return self._R

#     def get_absorbing(self):
#       return self._absorbing

#     # Getter functions used for DP, MC and TD agents
#     # You DO NOT NEED to modify them
#     def get_graphics(self):
#       return self._graphics

#     def get_action_size(self):
#       return self._action_size

#     def get_state_size(self):
#       return self._state_size

#     def get_gamma(self):
#       return self._gamma

#     # Functions used to perform episodes in the Maze environment
#     def reset(self):
#       """
#       Reset the environment state to one of the possible starting states
#       input: /
#       output: 
#         - t {int} -- current timestep
#         - state {int} -- current state of the envionment
#         - reward {int} -- current reward
#         - done {bool} -- True if reach a terminal state / 0 otherwise
#       """
#       self._t = 0
#       # Here, the initial state is selected as the state number associated with the starting location indexed at 
#       # a random number from 0-length(starting states)
#       self._state = self._get_state_from_loc(self._starting_locs[random.randrange(len(self._starting_locs))])
#       self._reward = 0
#       self._done = False
#       return self._t, self._state, self._reward, self._done

#     def step(self, action):
#       """
#       Perform an action in the environment
#       input: action {int} -- action to perform
#       output: 
#         - t {int} -- current timestep
#         - state {int} -- current state of the envionment
#         - reward {int} -- current reward
#         - done {bool} -- True if reach a terminal state / 0 otherwise
#       """

#       # If environment already finished, print an error
#       if self._done or self._absorbing[0, self._state]:
#         print("Please reset the environment")
#         return self._t, self._state, self._reward, self._done

#       # Drawing a random number used for probaility of next state
#       probability_success = random.uniform(0,1)

#       # Look for the first possible next states (so get a reachable state even if probability_success = 0)
#       new_state = 0
#       while self._T[self._state, new_state, action] == 0: 
#         new_state += 1
#       assert self._T[self._state, new_state, action] != 0, "Selected initial state should be probability 0, something might be wrong in the environment."

#       # Find the first state for which probability of occurence matches the random value
#       total_probability = self._T[self._state, new_state, action]
#       while (total_probability < probability_success) and (new_state < self._state_size-1):
#         new_state += 1
#         total_probability += self._T[self._state, new_state, action]
#       assert self._T[self._state, new_state, action] != 0, "Selected state should be probability 0, something might be wrong in the environment."
      
#       # Setting new t, state, reward and done
#       self._t += 1
#       self._reward = self._R[self._state, new_state, action]
#       self._done = self._absorbing[0, new_state] or self._t > self._max_t
#       self._state = new_state
#       return self._t, self._state, self._reward, self._done
    
#   ###################################### DP
#   # This class define the Dynamic Programing agent 
#   class DP_agent(object):
#     # [Action required]
#     def policy_evaluation(self, env, policy, threshold = 0.0001, gamma = 0.8):
#       """
#       Policy evaluation on GridWorld
#       input: 
#         - policy {np.array} -- policy to evaluate
#         - threshold {float} -- threshold value used to stop the policy evaluation algorithm
#         - gamma {float} -- discount factor
#       output: 
#         - V {np.array} -- value function corresponding to the policy 
#         - epochs {int} -- number of epochs to find this value function
#       """
      
#       # Ensure inputs are valid
#       assert (policy.shape[0] == env.get_state_size()) and (policy.shape[1] == env.get_action_size()), "The dimensions of the policy are not valid."
#       assert (gamma <=1) and (gamma >= 0), "Discount factor should be in [0, 1]."

#       # Initialisation
#       delta = 2*threshold # Ensure delta is bigger than the threshold to start the loop
#       V = np.zeros(env.get_state_size()) # Initialise value function to 0  
#       epoch = 0
#       # Storing the transition matrix, and R matrix to access later
#       T = env.get_T()
#       R = env.get_R()

#       # Get the absorbing states - FOR EDGE CASE
#       absorbing = env.get_absorbing()

#       #### 
#       # Add your code here
#       # Hint! You might need: env.get_state_size(), env.get_action_size(), T, R, absorbing
#       ####

#       # Basically, policy evaluation is an iterative process in which you iterate, find the V(s) matrix as according to the equation defined
#       # where 
#       # V(s) = for every state: summation(for every action defined by the policy for this state: summation(policy *
#       #                       (for every next state and reward in each action: summation(Transition*(reward + discount * V(next state)))) 
#       # We store the previous v(s) and the current v(s) for every iteration, and compare to find the maximum in the absolute difference between
#       # the two. If this maximum is less than threshold, stop - else, iterate

#       # While the DELTA is greater than threshold
#       while (delta > threshold):
#         # Copy the previous V into the previousValue to store it
#         previousV = np.copy(V)

#         # Update epoch
#         epoch += 1
        
#         # For every state that we have
#         for state in range(env.get_state_size()):
#           # SO LONG AS IT'S NOT AN ABSORBING STATE
#           if not absorbing[0, state]:
#             # Redefine V when we change state
#             tempV = 0
            
#             # Find the current V, as defined by the equation above
#             # For every action defined by the policy for this specific state, where policy[0] is the state and policy[1] is the action
#             for action in range(env.get_action_size()):
#               # For every next state
#               # Redefine Q when we change the action
#               qValue = 0
              
#               for next_state in range(env.get_state_size()):
#                 # We comput the inner loop first, which will be the computation of q(s|a), then multiply by the policy in the outside summation
#                 qValue = qValue + T[state, next_state, action] * (R[state, next_state, action] + gamma*(V[next_state]))
              
#               # We multiply this Q value by the policy for the given action at the given state
#               tempV = tempV + policy[state, action] * qValue
            
#             # Add this tempV value to the V matrix for the current state
#             V[state] = tempV
        
#         # Compare the previousV to the current V, where we find the maximum difference for all abs(previousV - V)
#         delta = max(abs(V - previousV))
              
#       return V, epoch
      
#     # [Action required]
#     def solvePolicy(self, env, threshold = 0.0001, gamma = 0.8):
#       """
#       Policy iteration on GridWorld
#       input: 
#         - threshold {float} -- threshold value used to stop the policy iteration algorithm
#         - gamma {float} -- discount factor
#       output:
#         - policy {np.array} -- policy found using the policy iteration algorithm
#         - V {np.array} -- value function corresponding to the policy 
#         - epochs {int} -- number of epochs to find this policy
#       """

#       # Ensure gamma value is valid
#       assert (gamma <=1) and (gamma >= 0), "Discount factor should be in [0, 1]."

#       # Initialisation
#       policy = np.zeros((env.get_state_size(), env.get_action_size())) # Vector of 0
#       policy[:, 0] = 1 # Initialise policy to choose action 1 systematically
#       V = np.zeros(env.get_state_size()) # Initialise value function to 0  
#       epochs = 0
#       policy_stable = False # Condition to stop the main loop
#       # Define delta for looping
#       delta = threshold
#       # Storing the transition matrix, and R matrix to access later
#       T = env.get_T()
#       R = env.get_R()

#       # Get the absorbing states - FOR EDGE CASE
#       absorbing = env.get_absorbing()
#       # Defining the start time
#       start_time = time.time()
      

#       #### 
#       # Add your code here
#       # Hint! You might need: env.get_state_size(), env.get_action_size(), self.T, self.R, self.absorbing, self.policy_evaluation()
#       ####

#       # Policy iteration is basically a process by which we do two things - first, policy EVALUATION as according a specific policy, then
#       # policy imporvement, in which we find the value of the policy for every action for every state, to see if it is the best one that maximizes
#       # the value function. This is given by the following function, where we do
#       # policy = (for every action, find the MAXIMUM(summation(for every next state and reward, summation(transition * (reward + gamma*next value)))))
#       # The first thing we need to do, then, is invoke the policy evaluation function to find the V as according to the policy

#       # POLICY ITERATION PART
#       # While not converged
#       while not policy_stable:

#         # Set policy_stable to true, changes anyway
#         policy_stable = True

#         # Invoke policy evaluation function to find V according to policy pi
#         V, epochs_eval = self.policy_evaluation(env = env, policy = policy, threshold = threshold, gamma = gamma)
#         # Increment epochs
#         epochs = epochs + epochs_eval

#         # For every single state that we have
#         for state in range(env.get_state_size()):
#           # IF NOT ABSORBING
#           if not absorbing[0, state]:
#             # The old action should be the one that has the maximum action value of the current policy at this state. This returns the index of the
#             # maximum value
#             old_action = np.argmax(policy[state,:])

#             # The way we find the best policy is by finding the action that maximizes the value function. As an equation, it looks as follows
#             # maxPolicy = maximum over all actions(for every next state and reward, summation(transition * (reward + gamma*value(next state))))

#             # For every action that we have, we want to find the qValue 
#             # Initialize the Q to have all the qValues given ALL (4) actions for a specific next state
#             Q = np.zeros(env.get_action_size())
#             # For every possible next state, find the Q value
#             for next_state in range(env.get_state_size()):
#               # Find the Q value for all possible next states, all actions
#               Q = Q + T[state,next_state,:] * (R[state,next_state,:] + gamma*V[next_state])
              
#             # After finding all the Q values, we set the newPolicy to have a probability of 1 where we have a maximum Q value
#             newPolicy = np.zeros(env.get_action_size())
#             newPolicy[np.argmax(Q)] = 1

#             # Add to the policies
#             policy[state, :] = newPolicy

#             # Now that we have the new policy, we want to see if the maximum valued action of the new policy is the same as the old max value
#             # action. If yes, then we converged and policy_stable = true
#             if old_action != np.argmax(newPolicy):
#               policy_stable = False       
      
#       # Getting the time taken
#       timeTaken = time.time() - start_time

#       return policy, V, epochs, timeTaken
    
#     # [Action required]
#     # WARNING: make sure this function can be called by the auto-marking script
#     def solveValue(self, env):
#       """
#       Solve a given Maze environment using Dynamic Programming
#       input: env {Maze object} -- Maze to solve
#       output: 
#         - policy {np.array} -- Optimal policy found to solve the given Maze environment 
#         - V {np.array} -- Corresponding value function 
#       """
      
#       # Initialisation (can be edited)
#       policy = np.zeros((env.get_state_size(), env.get_action_size())) 
#       V = np.zeros(env.get_state_size())

#       #### 
#       # Add your code here
#       # WARNING: for this agent only, you are allowed to access env.get_T(), env.get_R() and env.get_absorbing()
#       ####

#       # The decision here is to use VALUE ITERATION, rather than policy iteration, since we see that Value Iteration is equivalent to having one
#       # backup sweep rather than multiple, as in the case of policy iteration. In Policy Iteration, we have two main areas of computation: the first
#       # is policy evaluation, in which we run through all the states and evaluate the value function as according to a specific policy, and the 
#       # second is policy improvement/iteration, in which we continue in our iteration over every single state to find the action that maximizes
#       # the value function for that state (thus the ideal policy), continued until we reach a stable policy (convergence). In value iteration,
#       # on the other hand, we have one main form of computation, which is the actual value function evaluation. Here, rather than evaluating the
#       # value function as according to a specific policy, we basically iterate over all actions, rather than just those defined by a policy,
#       # and choose the value function that corresponds to the action that gives the maximum possible value. 

#       # First, we define a threshold, a delta, and a copy of the V function that we have
#       # The threshold is chosen arbitrarily as according to how accurate we want to be, where in this case we want to make sure that we have a
#       # maximum difference between the previous V and current V of 0.0001.
#       threshold = 0.0001
#       gamma = env.get_gamma()
#       delta = threshold
#       Vnew = np.copy(V)
#       # Storing the transition matrix, and R matrix to access later
#       T = env.get_T()
#       R = env.get_R()

#       # Get the start time
#       start_time = time.time()
#       # Get the absorbing states - FOR EDGE CASE
#       absorbing = env.get_absorbing()
#       # Get the number of epochs
#       epochs = 0

#       ########## VALUE ITERATION LOOP
#       # While the delta is greater than or equal to the threshold
#       while delta >= threshold:
#         # We set delta to 0, and change it every time we calculate the difference between the V's
#         delta = 0
#         epochs += 1

#         # For every single state that we have
#         for state in range(env.get_state_size()):
#           # Ensure that we are not at an absorbing state - FOR EDGE CASE, we don't want to check them
#           if not absorbing[0, state]:
            
#             # We begin the value iteration function, in which we calculate the Q value, which is defined as, for each state:
#             # Q = (for every next state/reward, summation(Transition * (reward + gamma*(V[next_state]))))
#             # Thus, Q needs to first be an array of size 4, since we have 4 actions
#             Q = np.zeros(4)
#             # For every next state, we do exactly what is mentioned above in the equation
#             for next_state in range(env.get_state_size()):
#               # We calculate Q here, as according to the equation
#               Q = Q + (T[state,next_state,:] * (R[state,next_state,:] + gamma*V[next_state]))
            
#             # Once we have the Q, we then find the maximum value and set that to be the value in Vnew[state]
#             Vnew[state] = np.max(Q)

#             # Set the value of delta to be the maximum of delta and the difference between the V's for this state
#             delta = max(delta, np.abs(Vnew[state] - V[state]))

#             # Set the value of V for this state to the value of Vnew for this state to iterate
#             V[state] = Vnew[state]
        
#       # Once we're out of the threshold loop for delta, we can now set the policy for each state to be the action that gives the maximum value
#       # of the value function. To do this, we again iterate to find the best Q
#       for state in range(env.get_state_size()):
#         # Set the value of Q to be 0's of size 4 originally
#         Q = np.zeros(4)
#         # Find the value of Q for this state, over all the next states
#         for next_state in range(env.get_state_size()):
#           Q = Q + (T[state,next_state,:] * (R[state,next_state,:] + gamma*V[next_state]))
        
#         # The policy for this state is going to be at the position where we have the max(Q), set to probability of 1
#         policy[state, np.argmax(Q)] = 1

#       # Get the end time
#       timeTaken = time.time() - start_time
#       return policy, V, epochs, timeTaken

#     def solveValueVarGamma(self, env, varGamma):
#       """
#       Solve a given Maze environment using Dynamic Programming
#       input: env {Maze object} -- Maze to solve
#       output: 
#         - policy {np.array} -- Optimal policy found to solve the given Maze environment 
#         - V {np.array} -- Corresponding value function 
#       """
      
#       # Initialisation (can be edited)
#       policy = np.zeros((env.get_state_size(), env.get_action_size())) 
#       V = np.zeros(env.get_state_size())

#       #### 
#       # Add your code here
#       # WARNING: for this agent only, you are allowed to access env.get_T(), env.get_R() and env.get_absorbing()
#       ####

#       # The decision here is to use VALUE ITERATION, rather than policy iteration, since we see that Value Iteration is equivalent to having one
#       # backup sweep rather than multiple, as in the case of policy iteration. In Policy Iteration, we have two main areas of computation: the first
#       # is policy evaluation, in which we run through all the states and evaluate the value function as according to a specific policy, and the 
#       # second is policy improvement/iteration, in which we continue in our iteration over every single state to find the action that maximizes
#       # the value function for that state (thus the ideal policy), continued until we reach a stable policy (convergence). In value iteration,
#       # on the other hand, we have one main form of computation, which is the actual value function evaluation. Here, rather than evaluating the
#       # value function as according to a specific policy, we basically iterate over all actions, rather than just those defined by a policy,
#       # and choose the value function that corresponds to the action that gives the maximum possible value. 

#       # First, we define a threshold, a delta, and a copy of the V function that we have
#       # The threshold is chosen arbitrarily as according to how accurate we want to be, where in this case we want to make sure that we have a
#       # maximum difference between the previous V and current V of 0.0001.
#       threshold = 0.0001
#       gamma = varGamma
#       delta = threshold
#       Vnew = np.copy(V)
#       # Storing the transition matrix, and R matrix to access later
#       T = env.get_T()
#       R = env.get_R()

#       # Get the absorbing states - FOR EDGE CASE
#       absorbing = env.get_absorbing()
#       epochs = 0

#       ########## VALUE ITERATION LOOP
#       # While the delta is greater than or equal to the threshold
#       while delta >= threshold:
#         # We set delta to 0, and change it every time we calculate the difference between the V's
#         delta = 0
#         epochs += 1

#         # For every single state that we have
#         for state in range(env.get_state_size()):
#           # Ensure that we are not at an absorbing state - FOR EDGE CASE, we don't want to check them
#           if not absorbing[0, state]:
            
#             # We begin the value iteration function, in which we calculate the Q value, which is defined as, for each state:
#             # Q = (for every next state/reward, summation(Transition * (reward + gamma*(V[next_state]))))
#             # Thus, Q needs to first be an array of size 4, since we have 4 actions
#             Q = np.zeros(4)
#             # For every next state, we do exactly what is mentioned above in the equation
#             for next_state in range(env.get_state_size()):
#               # We calculate Q here, as according to the equation
#               Q = Q + (T[state,next_state,:] * (R[state,next_state,:] + gamma*V[next_state]))
            
#             # Once we have the Q, we then find the maximum value and set that to be the value in Vnew[state]
#             Vnew[state] = np.max(Q)

#             # Set the value of delta to be the maximum of delta and the difference between the V's for this state
#             delta = max(delta, np.abs(Vnew[state] - V[state]))

#             # Set the value of V for this state to the value of Vnew for this state to iterate
#             V[state] = Vnew[state]
        
#       # Once we're out of the threshold loop for delta, we can now set the policy for each state to be the action that gives the maximum value
#       # of the value function. To do this, we again iterate to find the best Q
#       for state in range(env.get_state_size()):
#         # Set the value of Q to be 0's of size 4 originally
#         Q = np.zeros(4)
#         # Find the value of Q for this state, over all the next states
#         for next_state in range(env.get_state_size()):
#           Q = Q + (T[state,next_state,:] * (R[state,next_state,:] + gamma*V[next_state]))
        
#         # The policy for this state is going to be at the position where we have the max(Q), set to probability of 1
#         policy[state, np.argmax(Q)] = 1
    
#       return policy, V, epochs
    
#   #################### Impact of the probability on the value iteration algorithm
#   # HAVE TO CHANGE GRAPHICS AND MAZE FOR THIS
#   prob_range = [0, 0.1, 0.25, 0.4]
#   epochs = []
#   policies = []
#   values = []
#   titles = []
#   timeTakens = []

#   for index, prob in enumerate(prob_range):
#       maze = Maze(prob)
#       dp_agent = DP_agent()
#       policy, V, epoch, timeTaken = dp_agent.solveValue(maze)
#       epochs.append(epoch)
#       policies.append(policy)
#       values.append(V)
#       timeTakens.append(timeTaken)
#       titles.append("probability = {}".format(prob))

#   print("Impact of probability on the number of epochs needed for the value iteration algorithm:\n")
#   plt.figure()
#   plt.plot(prob_range, epochs)
#   plt.xlabel("Probability Range")
#   plt.ylabel("Number of epochs")
#   plt.show()

#   # Print all value functions and policies for different values of gamma
#   print("\nGraphical representation of the value function for each probability:\n")
#   maze.get_graphics().draw_value_grid(values, titles, 1, 4)

#   print("\nGraphical representation of the policy for each probability:\n")
#   maze.get_graphics().draw_policy_grid(policies, titles, 1, 4)

#   #################### Impact of gamma on the value iteration algorithm
#   gamma_range = [0, 0.2, 0.4, 0.6, 0.8, 1]
#   epochs = []
#   policies = []
#   values = []
#   titles = []

#   maze = Maze()
#   dp_agent = DP_agent()

#   # Use value iteration for each gamma value
#   for gamma in gamma_range:
#       policy, V, epoch = dp_agent.solveValueVarGamma(maze, varGamma = gamma)
#       epochs.append(epoch)
#       policies.append(policy)
#       values.append(V)
#       titles.append("gamma = {}".format(gamma))

#   # Plot the number of epochs vs gamma values
#   print("Impact of gamma value on the number of epochs needed for the value iteration algorithm:\n")
#   plt.figure()
#   plt.plot(gamma_range, epochs)
#   plt.xlabel("Gamma range")
#   plt.ylabel("Number of epochs")
#   plt.show()

#   # Print all value functions and policies for different values of gamma
#   print("\nGraphical representation of the value function for each gamma:\n")
#   maze.get_graphics().draw_value_grid(values, titles, 1, 6)


#   print("\nGraphical representation of the policy for each gamma:\n")
#   maze.get_graphics().draw_policy_grid(policies, titles, 1, 6)

#   #################### VALUE VS POLICY ITERATION
#   maze = Maze()
#   dp_agent = DP_agent()
#   dp_valuePol, dp_valueVal, dp_valueEpoch, dp_valueTime = dp_agent.solveValue(maze)
#   dp_policyPol, dp_policyVal, dp_policyEpoch, dp_policyTime = dp_agent.solvePolicy(maze)

#   dpValues = []
#   dpPolicies = []

#   dpValues.append(dp_valueVal)
#   dpValues.append(dp_policyVal)

#   dpPolicies.append(dp_valuePol)
#   dpPolicies.append(dp_policyPol)

#   titles = ["Value Itr. V(s), {} epoch in {} sec".format(dp_valueEpoch, dp_valueTime), "Policy Itr. V(s), {} epoch in {} sec".format(dp_policyEpoch, dp_policyTime), "Value Iteration Policy", "Policy Iteration Policy"]

#   print("Results of the DP agent for VALUE and POLICY iteration:\n")
#   maze.get_graphics().draw_value_grid(dpValues, titles[:2], 1, 2)
#   maze.get_graphics().draw_policy_grid(dpPolicies, titles[2:4], 1, 2)

#   # This class define the Monte-Carlo agent
#   class MC_agent(object):
    
#     def solveONFindRep(self, env):
#       """
#       Solve a given Maze environment using Monte Carlo learning
#       input: env {Maze object} -- Maze to solve
#       output: 
#         - policy {np.array} -- Optimal policy found to solve the given Maze environment 
#         - values {list of np.array} -- List of successive value functions for each episode 
#         - total_rewards {list of float} -- Corresponding list of successive total non-discounted sum of reward for each episode 
#       """

#       # Initialisation (can be edited)
#       Q = np.random.rand(env.get_state_size(), env.get_action_size()) 
#       V = np.zeros(env.get_state_size())
#       policy = np.zeros((env.get_state_size(), env.get_action_size())) 
#       # We initialize the policy to be an epsilon-soft policy, in which all actions have a probability of epsilon/number of actions to be selected
#       # Define epsilon to be 1
#       epsilon = 0.8
#       # In general, for all states and all actions, epsilon-soft gives them probability epsilon/cardinality(A) or 1 - epsilon + epsilon/cardinality(A)
#       # As an initial policy, we set that for all states, every action has an equal probability of being run
#       policy[:,:] = 0.25
#       values = [V]
#       total_rewards = []
#       # Define gamma 
#       gamma = env.get_gamma()
#       # Define learning rate
#       learningRate = 0.1
#       # Create a statesSeen to calculate V with it
#       statesSeen = set()
#       deltaList = []

#       #### 
#       # Add your code here
#       # WARNING: this agent only has access to env.reset() and env.step()
#       # You should not use env.get_T(), env.get_R() or env.get_absorbing() to compute any value
#       ####

#       # Here, we want to find the number of replications before the values of the value function and policy converge to a specific 
#       # threshold. At this threshold, we consider that the number of replications is sufficient
#       # Number of replications, set to 0 initially
#       numberOfReplications = 0
#       # Defining the threshold
#       threshold = 0.07
#       # Initializing delta to equal threshold
#       delta = threshold
#       # Defining the lists that will have this replication and previous replication's rewards, value functions and policies
#       ongoingRepValues = []
#       averageDifferenecesList = []

#       while delta >= threshold:
#         # Increase number of Replications
#         numberOfReplications = numberOfReplications + 1
#         # Reset V, values, policy and total_rewards in every run
#         V = np.zeros(env.get_state_size())
#         values = [V]
#         total_rewards = []
#         policy[:,:] = 0.25

#         ################### PHASE 0 - EPISODE GENERATION
#         # Define the number of episodes that we want to generate
#         numberOfEpisodes = 1000
#         # Set the random seed
#         np.random.seed(0)

#         # While we have less than numberOfEpisodes valid episodes
#         for episode in range(numberOfEpisodes):
#           # The epsilon is decaying
#           epsilon = (1 - episode/numberOfEpisodes)
#           # Define the list that will contain all the information/steps of this episode. Note that this will be in the form of a list of tuples, where
#           # each tuple will contain (timeStep, state, action, reward, done). Note that we define it within the episodes loop, as we want to reset it every ep
#           episodeSteps = []
#           # Reset to get the timestep, state, starting state, action, reward, and done boolean
#           timeStep, state, reward, done = env.reset()
#           # Iterate until ending criteria - either reach absorbing state (done == True), or 500 steps (timeStep >= 500)
#           while (done != True):
#             # For every episode, we want to take a step in the direction of the policy - we do this randomly with probability given by the policy
#             # We can use the np.random.choice() function for this, where the choice must be in the range of number of actions that we have, and the
#             # probability is defined by the entry of the action probabilities for that given state
#             action = np.random.choice(env.get_action_size(), p=policy[state, :])

#             # Append the step to the episodeSteps list
#             episodeSteps.append((timeStep, state, action, reward, done))

#             # Take a step in the direction of the action defined by the policy
#             timeStep, state, reward, done = env.step(action)
            
#           # Append the last reward to the list of episodeSteps, where the last step is basically gonna be the TERMINAL STATE, with the reward of
#           # that, and no action associated with it
#           episodeSteps.append((timeStep, state, None, reward, done)) 
          
#           ################### PHASE 1 - LOOPING OVER EPISODE
#           # Now that we've generated the episode, the next step is to iterate over the steps defined in the episode, where we define the actual
#           # on-policy first-visit MC policy
#           # Note that we create two returns values, one that is going to be used discounted, and one undiscounted
#           discountedReturns = 0
#           undiscountedReturns = 0
          
#           # NOTE that we only loop IF the last state was valid
#           # For every step (tuple) of the episodeSteps list. Notice that we reverse the order, so that we go from the end to the beginning
#           for episodeStep in reversed(list(episodeSteps[:len(episodeSteps) - 1])):
#             # The discounted returns is going to be gamme*discountedReturns + reward of next step (technically the same as the reward in
#             # the current tuple). Undiscounted is the same without gamma
#             # Extract the state, action and reward from each episode tuple
#             timeStep = episodeStep[0]
#             state = episodeStep[1]
#             action = episodeStep[2]
#             rewardPlus1 = episodeSteps[timeStep + 1][3]

#             # Add the state to the statesSeen set
#             statesSeen.add(state)
            
#             discountedReturns = gamma*discountedReturns + rewardPlus1
#             undiscountedReturns = undiscountedReturns + rewardPlus1

#             # Check if the current state-action pair for the current episodeStep appeared in any of the previous tuples before this in the
#             # episodeSteps list
#             # Get all the visited states and actions
#             previousStates = [tup[1] for tup in episodeSteps[:timeStep - 1]]
#             previousActions = [tup[2] for tup in episodeSteps[:timeStep - 1]]
#             # Make this into a list of tuples
#             prevStateActionPair = list(map(lambda x, y: (x, y), previousStates, previousActions))
#             # Get current state, action as a tuple
#             currentStateActionPair = (state, action)
#             # Check if the current episode's state or action is in the visitedStates/Actions list until current index
#             if (currentStateActionPair not in prevStateActionPair):
#               # Find the qValue for the current state-action pair as the average as an incremental one
#               Q[state, action] =  Q[state, action] + learningRate*(discountedReturns - Q[state, action])
#               # Find the index of the action that gives the largest value
#               bestAction = np.argmax(Q[state,:])
#               # For all actions in the state
#               for action in range(env.get_action_size()):
#                 # Update the policy according to the epsilon-greedy policy
#                 # If current action index is the same as that of the best action
#                 if (action == bestAction):
#                   # Place more emphasize on this action, make policy have a greater value
#                   policy[state, action] = 1 - epsilon + (epsilon/env.get_action_size())
#                 else:
#                   policy[state, action] = epsilon/env.get_action_size()
                
#           # At the end of the episode, we want to find the value function for every state we've been in
#           for state in statesSeen:
#             V[state] = 0
#             # For every possible action
#             for action in range(env.get_action_size()):
#               V[state] += policy[state, action] * Q[state, action]

#           # Append the V to the value
#           values.append(V)
#           # Now that we're finished with the episode, we want to append the undiscounted reward
#           total_rewards.append(undiscountedReturns)
        
#         print('Rep ', numberOfReplications)

#         # Find the list of values as an NP array
#         valueNP = np.array(values)
#         #print('Values ', valueNP)
#         #print('ValuesShape ', valueNP.shape)

#         # Have a runningRepValues without the current valuesNP
#         if (numberOfReplications == 1):
#           previousRepValuesNP = np.zeros_like(valueNP) 

#         # Append the current values to the ongoing values
#         ongoingRepValues.append(valueNP)
#         # Make a numpy array of the ongoing values
#         ongoingRepValuesNP = np.asarray(ongoingRepValues)
#         # Find the mean across the columns of this numpy matrix
#         ongoingRepValuesNPMean = np.mean(ongoingRepValuesNP, axis=0)
#         #print('ongoingMean shape ', ongoingRepValuesNPMean.shape)
#         #print('ongoingMean values ', ongoingRepValuesNPMean)
#         #print('previousRepValuesNP shape ', previousRepValuesNP.shape)

#         # Create the averageDifference np array
#         averageDifference = np.zeros_like(ongoingRepValuesNPMean)

#         # Compare between current and ongoing, so long as we arent in first rep
#         averageDifference = np.abs(ongoingRepValuesNPMean - previousRepValuesNP)
        
#         # Make the current thing the previous one
#         previousRepValuesNP = ongoingRepValuesNPMean
#         #print('Avg diff', averageDifference)

#         # Append the averageDifference to the list of differences
#         averageDifferenecesList.append(averageDifference)

#         # Make this the delta
#         delta = np.mean(averageDifference)
#         deltaList.append(delta)

#         print("Rep ", numberOfReplications, ", delta ", delta)

#       return policy, values, total_rewards, deltaList, numberOfReplications

#     def solveON(self, env):
#       """
#       Solve a given Maze environment using Monte Carlo learning
#       input: env {Maze object} -- Maze to solve
#       output: 
#         - policy {np.array} -- Optimal policy found to solve the given Maze environment 
#         - values {list of np.array} -- List of successive value functions for each episode 
#         - total_rewards {list of float} -- Corresponding list of successive total non-discounted sum of reward for each episode 
#       """

#       # Initialisation (can be edited)
#       Q = np.random.rand(env.get_state_size(), env.get_action_size()) 
#       V = np.zeros(env.get_state_size())
#       policy = np.zeros((env.get_state_size(), env.get_action_size())) 
#       # We initialize the policy to be an epsilon-soft policy, in which all actions have a probability of epsilon/number of actions to be selected
#       # Define epsilon to be 1
#       epsilon = 0.8
#       # In general, for all states and all actions, epsilon-soft gives them probability epsilon/cardinality(A) or 1 - epsilon + epsilon/cardinality(A)
#       # As an initial policy, we set that for all states, every action has an equal probability of being run
#       policy[:,:] = 0.25
#       values = [V]
#       total_rewards = []
#       # Define gamma 
#       gamma = env.get_gamma()
#       # Define learning rate
#       learningRate = 0.1
#       # Create a statesSeen to calculate V with it
#       statesSeen = set()

#       #### 
#       # Add your code here
#       # WARNING: this agent only has access to env.reset() and env.step()
#       # You should not use env.get_T(), env.get_R() or env.get_absorbing() to compute any value
#       ####

#       # We try first the on-policy first-visit MC control for epsilon-soft policies. Here, what we do is generate an arbitrarily large number of
#       # episodes, where for each episode we start using reset(), we then iterate by taking steps in the direction of our policy to generate the
#       # episode itself. This is repeated until we either reach an absorbing state, or take 500 steps.
#       ################### PHASE 0 - EPISODE GENERATION
#       # Define the number of episodes that we want to generate
#       numberOfEpisodes = 1000
#       # Set the random seed
#       np.random.seed(0)

#       # While we have less than numberOfEpisodes valid episodes
#       for episode in range(numberOfEpisodes):
#         # The epsilon is decaying
#         epsilon = (1 - episode/numberOfEpisodes)
#         # Define the list that will contain all the information/steps of this episode. Note that this will be in the form of a list of tuples, where
#         # each tuple will contain (timeStep, state, action, reward, done). Note that we define it within the episodes loop, as we want to reset it every ep
#         episodeSteps = []
#         # Reset to get the timestep, state, starting state, action, reward, and done boolean
#         timeStep, state, reward, done = env.reset()
#         # Iterate until ending criteria - either reach absorbing state (done == True), or 500 steps (timeStep >= 500)
#         while (done != True):
#           # For every episode, we want to take a step in the direction of the policy - we do this randomly with probability given by the policy
#           # We can use the np.random.choice() function for this, where the choice must be in the range of number of actions that we have, and the
#           # probability is defined by the entry of the action probabilities for that given state
#           action = np.random.choice(env.get_action_size(), p=policy[state, :])

#           # Append the step to the episodeSteps list
#           episodeSteps.append((timeStep, state, action, reward, done))

#           # Take a step in the direction of the action defined by the policy
#           timeStep, state, reward, done = env.step(action)
          
#         # Append the last reward to the list of episodeSteps, where the last step is basically gonna be the TERMINAL STATE, with the reward of
#         # that, and no action associated with it
#         episodeSteps.append((timeStep, state, None, reward, done)) 
        
#         ################### PHASE 1 - LOOPING OVER EPISODE
#         # Now that we've generated the episode, the next step is to iterate over the steps defined in the episode, where we define the actual
#         # on-policy first-visit MC policy
#         # Note that we create two returns values, one that is going to be used discounted, and one undiscounted
#         discountedReturns = 0
#         undiscountedReturns = 0
        
#         # NOTE that we only loop IF the last state was valid
#         # For every step (tuple) of the episodeSteps list. Notice that we reverse the order, so that we go from the end to the beginning
#         for episodeStep in reversed(list(episodeSteps[:len(episodeSteps) - 1])):
#           # The discounted returns is going to be gamme*discountedReturns + reward of next step (technically the same as the reward in
#           # the current tuple). Undiscounted is the same without gamma
#           # Extract the state, action and reward from each episode tuple
#           timeStep = episodeStep[0]
#           state = episodeStep[1]
#           action = episodeStep[2]
#           rewardPlus1 = episodeSteps[timeStep + 1][3]

#           # Add the state to the statesSeen set
#           statesSeen.add(state)
          
#           discountedReturns = gamma*discountedReturns + rewardPlus1
#           undiscountedReturns = undiscountedReturns + rewardPlus1

#           # Check if the current state-action pair for the current episodeStep appeared in any of the previous tuples before this in the
#           # episodeSteps list
#           # Get all the visited states and actions
#           previousStates = [tup[1] for tup in episodeSteps[:timeStep - 1]]
#           previousActions = [tup[2] for tup in episodeSteps[:timeStep - 1]]
#           # Make this into a list of tuples
#           prevStateActionPair = list(map(lambda x, y: (x, y), previousStates, previousActions))
#           # Get current state, action as a tuple
#           currentStateActionPair = (state, action)
#           # Check if the current episode's state or action is in the visitedStates/Actions list until current index
#           if (currentStateActionPair not in prevStateActionPair):
#             # Find the qValue for the current state-action pair as the average as an incremental one
#             Q[state, action] =  Q[state, action] + learningRate*(discountedReturns - Q[state, action])
#             # Find the index of the action that gives the largest value
#             bestAction = np.argmax(Q[state,:])
#             # For all actions in the state
#             for action in range(env.get_action_size()):
#               # Update the policy according to the epsilon-greedy policy
#               # If current action index is the same as that of the best action
#               if (action == bestAction):
#                 # Place more emphasize on this action, make policy have a greater value
#                 policy[state, action] = 1 - epsilon + (epsilon/env.get_action_size())
#               else:
#                 policy[state, action] = epsilon/env.get_action_size()
              
#         # At the end of the episode, we want to find the value function for every state we've been in
#         newV = np.zeros(env.get_state_size())
#         newV = copy.deepcopy(V)
#         for state in statesSeen:
#           newV[state] = 0
#           # For every possible action
#           for action in range(env.get_action_size()):
#             newV[state] += policy[state, action] * Q[state, action]

#         # Copy V to a list
#         values.append(np.copy(newV))

#         # Now that we're finished with the episode, we want to append the undiscounted reward
#         total_rewards.append(undiscountedReturns)
          
#       return policy, values, total_rewards

#     def solveOFF(self, env):
#       """
#       Solve a given Maze environment using Monte Carlo learning
#       input: env {Maze object} -- Maze to solve
#       output: 
#         - policy {np.array} -- Optimal policy found to solve the given Maze environment 
#         - values {list of np.array} -- List of successive value functions for each episode 
#         - total_rewards {list of float} -- Corresponding list of successive total non-discounted sum of reward for each episode 
#       """

#       # Initialisation (can be edited)
#       Q = np.random.rand(env.get_state_size(), env.get_action_size()) 
#       V = np.zeros(env.get_state_size())
#       behaviorPolicy = np.zeros((env.get_state_size(), env.get_action_size()))
#       targetPolicy = np.zeros((env.get_state_size(), env.get_action_size()))
#       C = np.zeros((env.get_state_size(), env.get_action_size()))
#       # The behavior or episode-generating policy will be an epsilon-soft policy, in which all actions have a minimum probability of epsilon/number of actions to 
#       # be selected
#       # Define epsilon to be 1
#       epsilon = 0.8
#       # For the behavior policy, we set that for all states, every action has an equal probability of being run
#       behaviorPolicy[:,:] = 0.25
#       values = [V]
#       total_rewards = []
#       # Define gamma 
#       gamma = env.get_gamma()
#       # Define learning rate
#       learningRate = 0.05
#       # Create a statesSeen to calculate V with it
#       statesSeen = set()

#       #### 
#       # Add your code here
#       # WARNING: this agent only has access to env.reset() and env.step()
#       # You should not use env.get_T(), env.get_R() or env.get_absorbing() to compute any value
#       ####

#       # We try now the off-policy MC algorithm, in which we have a behavior policy which we generate episodes with, and a target policy which we aim to
#       # optimize. We follow incremental updates to Q here, where the learning rate is essentially the weights W over the matrix C
#       ################### PHASE 0 - EPISODE GENERATION
#       # Define the number of episodes that we want to generate
#       numberOfEpisodes = 6000
#       # Set the random seed
#       np.random.seed(0)

#       # While we have less than numberOfEpisodes valid episodes
#       for episode in range(numberOfEpisodes):
#         # The epsilon is decaying
#         epsilon = (1 - episode/numberOfEpisodes)
#         # Define the list that will contain all the information/steps of this episode. Note that this will be in the form of a list of tuples, where
#         # each tuple will contain (timeStep, state, action, reward, done). Note that we define it within the episodes loop, as we want to reset it every ep
#         episodeSteps = []
#         # Reset to get the timestep, state, starting state, action, reward, and done boolean
#         timeStep, state, reward, done = env.reset()
#         # Iterate until ending criteria - either reach absorbing state (done == True), or 500 steps (timeStep >= 500)
#         while (done != True):
#           # For every episode, we want to take a step in the direction of the policy - we do this randomly with probability given by the behavior policy
#           # We can use the np.random.choice() function for this, where the choice must be in the range of number of actions that we have, and the
#           # probability is defined by the entry of the action probabilities for that given state
#           action = np.random.choice(env.get_action_size(), p=behaviorPolicy[state, :])

#           # Append the step to the episodeSteps list
#           episodeSteps.append((timeStep, state, action, reward, done))

#           # Take a step in the direction of the action defined by the policy
#           timeStep, state, reward, done = env.step(action)
          
#         # Append the last reward to the list of episodeSteps, where the last step is basically gonna be the TERMINAL STATE, with the reward of
#         # that, and no action associated with it
#         episodeSteps.append((timeStep, state, None, reward, done)) 
        
#         ################### PHASE 1 - LOOPING OVER EPISODE
#         # Now that we've generated the episode, the next step is to iterate over the steps defined in the episode, where we define the actual
#         # on-policy first-visit MC policy
#         # Note that we create two returns values, one that is going to be used discounted, and one undiscounted. We also add the weight
#         discountedReturns = 0
#         undiscountedReturns = 0
#         weight = 1
        
#         # NOTE that we only loop IF the last state was valid
#         # For every step (tuple) of the episodeSteps list. Notice that we reverse the order, so that we go from the end to the beginning
#         for episodeStep in reversed(list(episodeSteps[:len(episodeSteps) - 1])):
#           # The discounted returns is going to be gamme*discountedReturns + reward of next step (technically the same as the reward in
#           # the current tuple). Undiscounted is the same without gamma
#           # Extract the state, action and reward from each episode tuple
#           timeStep = episodeStep[0]
#           state = episodeStep[1]
#           action = episodeStep[2]
#           rewardPlus1 = episodeSteps[timeStep + 1][3]

#           # Add the state to the statesSeen set
#           statesSeen.add(state)
          
#           discountedReturns = gamma*discountedReturns + rewardPlus1
#           undiscountedReturns = undiscountedReturns + rewardPlus1

#           # Increase C by the weight
#           C[state, action] = C[state, action] + weight
#           # Find the qValue for the current state-action pair as given by the function
#           Q[state, action] =  Q[state, action] + (weight/C[state,action])*(discountedReturns - Q[state, action])

#           # Create policy where the position of the maximum action for this state is set to 1
#           newPolicy = np.zeros(env.get_action_size())
#           newPolicy[np.argmax(Q[state,:])] = 1

#           # Add to the policies
#           targetPolicy[state, :] = newPolicy
          
#           # If current action is not this max that we found
#           if action != np.argmax(targetPolicy[state,:]):
#             # Break out of the loop
#             break

#           # Update the weight
#           weight = weight * (1/behaviorPolicy[state,action])
              
#         # At the end of the episode, we want to find the value function for every state we've been in
#         for state in statesSeen:
#           V[state] = 0
#           # For every possible action
#           for action in range(env.get_action_size()):
#             V[state] += targetPolicy[state, action] * Q[state, action]

#         # Append the V to the value
#         values.append(V)

#         # Now that we're finished with the episode, we want to append the undiscounted reward
#         total_rewards.append(undiscountedReturns)

#       return targetPolicy, values, total_rewards


#     # WARNING: make sure this function can be called by the auto-marking script
#     def solveAveragedVarEpsilonReps(self, env, varEp):
#       """
#       Solve a given Maze environment using Monte Carlo learning
#       input: env {Maze object} -- Maze to solve
#       output: 
#         - policy {np.array} -- Optimal policy found to solve the given Maze environment 
#         - values {list of np.array} -- List of successive value functions for each episode 
#         - total_rewards {list of float} -- Corresponding list of successive total non-discounted sum of reward for each episode 
#       """

#       # Initialisation (can be edited)
#       Q = np.random.rand(env.get_state_size(), env.get_action_size()) 
#       V = np.zeros(env.get_state_size())
#       policy = np.zeros((env.get_state_size(), env.get_action_size())) 
#       # We initialize the policy to be an epsilon-soft policy, in which all actions have a probability of epsilon/number of actions to be selected
#       # Define epsilon to be 1
#       epsilon = varEp
#       # In general, for all states and all actions, epsilon-soft gives them probability epsilon/cardinality(A) or 1 - epsilon + epsilon/cardinality(A)
#       # As an initial policy, we set that for all states, every action has an equal probability of being run
#       policy[:,:] = 0.25
#       values = [V]
#       total_rewards = []
#       # Define gamma 
#       gamma = env.get_gamma()
#       # Define learning rate
#       learningRate = 0.1
#       # Create a statesSeen to calculate V with it
#       statesSeen = set()

#       #### 
#       # Add your code here
#       # WARNING: this agent only has access to env.reset() and env.step()
#       # You should not use env.get_T(), env.get_R() or env.get_absorbing() to compute any value
#       ####

#       # We try first the on-policy first-visit MC control for epsilon-soft policies. Here, what we do is generate an arbitrarily large number of
#       # episodes, where for each episode we start using reset(), we then iterate by taking steps in the direction of our policy to generate the
#       # episode itself. This is repeated until we either reach an absorbing state, or take 500 steps.
#       ################### PHASE 0 - EPISODE GENERATION
#       # Define the number of episodes that we want to generate
#       numberOfEpisodes = 1000
#       # Set the random seed
#       np.random.seed(0)

#       # NOTE that instead of just doing 1000 episodes, what we do instead we make multiple runs, where every run is 1000 episodes. This is done to
#       # find a more representative show of the actual MC algorithm
#       # Define the number of runs
#       numberOfRuns = 30
#       # Define the total running rewards, policy and values
#       totalRValues = []
#       totalRRewards = []
#       totalPolicy = []
      
#       # For every run that we have
#       for run in range(numberOfRuns):
#         print('Run', run)
#         # Reset V, values, policy and total_rewards in every run
#         V = np.zeros(env.get_state_size())
#         values = [V]
#         total_rewards = []
#         policy[:,:] = 0.25

#         # While we have less than numberOfEpisodes valid episodes
#         for episode in range(numberOfEpisodes):
#           # The epsilon is decaying
#           # Define the list that will contain all the information/steps of this episode. Note that this will be in the form of a list of tuples, where
#           # each tuple will contain (timeStep, state, action, reward, done). Note that we define it within the episodes loop, as we want to reset it every ep
#           episodeSteps = []
#           # Reset to get the timestep, state, starting state, action, reward, and done boolean
#           timeStep, state, reward, done = env.reset()
#           # Iterate until ending criteria - either reach absorbing state (done == True), or 500 steps (timeStep >= 500)
#           while (done != True):
#             # For every episode, we want to take a step in the direction of the policy - we do this randomly with probability given by the policy
#             # We can use the np.random.choice() function for this, where the choice must be in the range of number of actions that we have, and the
#             # probability is defined by the entry of the action probabilities for that given state
#             action = np.random.choice(env.get_action_size(), p=policy[state, :])

#             # Append the step to the episodeSteps list
#             episodeSteps.append((timeStep, state, action, reward, done))

#             # Take a step in the direction of the action defined by the policy
#             timeStep, state, reward, done = env.step(action)
            
#           # Append the last reward to the list of episodeSteps, where the last step is basically gonna be the TERMINAL STATE, with the reward of
#           # that, and no action associated with it
#           episodeSteps.append((timeStep, state, None, reward, done)) 
          
#           ################### PHASE 1 - LOOPING OVER EPISODE
#           # Now that we've generated the episode, the next step is to iterate over the steps defined in the episode, where we define the actual
#           # on-policy first-visit MC policy
#           # Note that we create two returns values, one that is going to be used discounted, and one undiscounted
#           discountedReturns = 0
#           undiscountedReturns = 0
          
#           # NOTE that we only loop IF the last state was valid
#           # For every step (tuple) of the episodeSteps list. Notice that we reverse the order, so that we go from the end to the beginning
#           for episodeStep in reversed(list(episodeSteps[:len(episodeSteps) - 1])):
#             # The discounted returns is going to be gamme*discountedReturns + reward of next step (technically the same as the reward in
#             # the current tuple). Undiscounted is the same without gamma
#             # Extract the state, action and reward from each episode tuple
#             timeStep = episodeStep[0]
#             state = episodeStep[1]
#             action = episodeStep[2]
#             rewardPlus1 = episodeSteps[timeStep + 1][3]

#             # Add the state to the statesSeen set
#             statesSeen.add(state)
            
#             discountedReturns = gamma*discountedReturns + rewardPlus1
#             undiscountedReturns = undiscountedReturns + rewardPlus1

#             # Check if the current state-action pair for the current episodeStep appeared in any of the previous tuples before this in the
#             # episodeSteps list
#             # Get all the visited states and actions
#             previousStates = [tup[1] for tup in episodeSteps[:timeStep - 1]]
#             previousActions = [tup[2] for tup in episodeSteps[:timeStep - 1]]
#             # Make this into a list of tuples
#             prevStateActionPair = list(map(lambda x, y: (x, y), previousStates, previousActions))
#             # Get current state, action as a tuple
#             currentStateActionPair = (state, action)
#             # Check if the current episode's state or action is in the visitedStates/Actions list until current index
#             if (currentStateActionPair not in prevStateActionPair):
#               # Find the qValue for the current state-action pair as the average as an incremental one
#               Q[state, action] =  Q[state, action] + learningRate*(discountedReturns - Q[state, action])
#               # Find the index of the action that gives the largest value
#               bestAction = np.argmax(Q[state,:])
#               # For all actions in the state
#               for action in range(env.get_action_size()):
#                 # Update the policy according to the epsilon-greedy policy
#                 # If current action index is the same as that of the best action
#                 if (action == bestAction):
#                   # Place more emphasize on this action, make policy have a greater value
#                   policy[state, action] = 1 - epsilon + (epsilon/env.get_action_size())
#                 else:
#                   policy[state, action] = epsilon/env.get_action_size()
                
#           # At the end of the episode, we want to find the value function for every state we've been in
#           for state in statesSeen:
#             V[state] = 0
#             # For every possible action
#             for action in range(env.get_action_size()):
#               V[state] += policy[state, action] * Q[state, action]

#           # Append the V to the value
#           values.append(V)

#           # Now that we're finished with the episode, we want to append the undiscounted reward
#           total_rewards.append(undiscountedReturns)

#         # At the end of each run, we want to append the total Values, total Policies and total Rewards for the run to the running sum
#         totalRValues.append(values)
#         totalRRewards.append(total_rewards)
#         totalPolicy.append(policy)
      
#       # Convert to numpy arrays so we can use numpy functions
#       totalPolicyNP = np.asarray(totalPolicy)
#       totalRValuesNP = np.asarray(totalRValues)
#       totalRRewardsNP = np.asarray(totalRRewards)

#       # Find the average of all runs
#       averagedPolicy = np.mean(totalPolicyNP, axis=0)
#       averagedValues = np.mean(totalRValuesNP, axis=0)
#       averagedRewards = np.mean(totalRRewardsNP, axis=0)

#       # Find the std of each of the runs
#       deviationPolicy = np.std(totalPolicyNP, axis=0)
#       deviationValue = np.std(totalRValuesNP, axis=0)
#       deviationRewards = np.std(totalRRewardsNP, axis=0)

#       return averagedPolicy, averagedValues, averagedRewards, deviationPolicy, deviationValue, deviationRewards

#     # WARNING: make sure this function can be called by the auto-marking script
#     def solveAveragedVarLRReps(self, env, varLR):
#       """
#       Solve a given Maze environment using Monte Carlo learning
#       input: env {Maze object} -- Maze to solve
#       output: 
#         - policy {np.array} -- Optimal policy found to solve the given Maze environment 
#         - values {list of np.array} -- List of successive value functions for each episode 
#         - total_rewards {list of float} -- Corresponding list of successive total non-discounted sum of reward for each episode 
#       """

#       # Initialisation (can be edited)
#       Q = np.random.rand(env.get_state_size(), env.get_action_size()) 
#       V = np.zeros(env.get_state_size())
#       policy = np.zeros((env.get_state_size(), env.get_action_size())) 
#       # We initialize the policy to be an epsilon-soft policy, in which all actions have a probability of epsilon/number of actions to be selected
#       # Define epsilon to be 1
#       epsilon = 0.8
#       # In general, for all states and all actions, epsilon-soft gives them probability epsilon/cardinality(A) or 1 - epsilon + epsilon/cardinality(A)
#       # As an initial policy, we set that for all states, every action has an equal probability of being run
#       policy[:,:] = 0.25
#       values = [V]
#       total_rewards = []
#       # Define gamma 
#       gamma = env.get_gamma()
#       # Define learning rate
#       learningRate = varLR
#       # Create a statesSeen to calculate V with it
#       statesSeen = set()

#       #### 
#       # Add your code here
#       # WARNING: this agent only has access to env.reset() and env.step()
#       # You should not use env.get_T(), env.get_R() or env.get_absorbing() to compute any value
#       ####

#       # We try first the on-policy first-visit MC control for epsilon-soft policies. Here, what we do is generate an arbitrarily large number of
#       # episodes, where for each episode we start using reset(), we then iterate by taking steps in the direction of our policy to generate the
#       # episode itself. This is repeated until we either reach an absorbing state, or take 500 steps.
#       ################### PHASE 0 - EPISODE GENERATION
#       # Define the number of episodes that we want to generate
#       numberOfEpisodes = 1000
#       # Set the random seed
#       np.random.seed(0)

#       # NOTE that instead of just doing 1000 episodes, what we do instead we make multiple runs, where every run is 1000 episodes. This is done to
#       # find a more representative show of the actual MC algorithm
#       # Define the number of runs
#       numberOfRuns = 30
#       # Define the total running rewards, policy and values
#       totalRValues = []
#       totalRRewards = []
#       totalPolicy = []
      
#       # For every run that we have
#       for run in range(numberOfRuns):
#         print('Run', run)
#         # Reset V, values, policy and total_rewards in every run
#         V = np.zeros(env.get_state_size())
#         values = [V]
#         total_rewards = []
#         policy[:,:] = 0.25

#         # While we have less than numberOfEpisodes valid episodes
#         for episode in range(numberOfEpisodes):
#           # The epsilon is decaying
#           epsilon = (1 - episode/numberOfEpisodes)
#           # Define the list that will contain all the information/steps of this episode. Note that this will be in the form of a list of tuples, where
#           # each tuple will contain (timeStep, state, action, reward, done). Note that we define it within the episodes loop, as we want to reset it every ep
#           episodeSteps = []
#           # Reset to get the timestep, state, starting state, action, reward, and done boolean
#           timeStep, state, reward, done = env.reset()
#           # Iterate until ending criteria - either reach absorbing state (done == True), or 500 steps (timeStep >= 500)
#           while (done != True):
#             # For every episode, we want to take a step in the direction of the policy - we do this randomly with probability given by the policy
#             # We can use the np.random.choice() function for this, where the choice must be in the range of number of actions that we have, and the
#             # probability is defined by the entry of the action probabilities for that given state
#             action = np.random.choice(env.get_action_size(), p=policy[state, :])

#             # Append the step to the episodeSteps list
#             episodeSteps.append((timeStep, state, action, reward, done))

#             # Take a step in the direction of the action defined by the policy
#             timeStep, state, reward, done = env.step(action)
            
#           # Append the last reward to the list of episodeSteps, where the last step is basically gonna be the TERMINAL STATE, with the reward of
#           # that, and no action associated with it
#           episodeSteps.append((timeStep, state, None, reward, done)) 
          
#           ################### PHASE 1 - LOOPING OVER EPISODE
#           # Now that we've generated the episode, the next step is to iterate over the steps defined in the episode, where we define the actual
#           # on-policy first-visit MC policy
#           # Note that we create two returns values, one that is going to be used discounted, and one undiscounted
#           discountedReturns = 0
#           undiscountedReturns = 0
          
#           # NOTE that we only loop IF the last state was valid
#           # For every step (tuple) of the episodeSteps list. Notice that we reverse the order, so that we go from the end to the beginning
#           for episodeStep in reversed(list(episodeSteps[:len(episodeSteps) - 1])):
#             # The discounted returns is going to be gamme*discountedReturns + reward of next step (technically the same as the reward in
#             # the current tuple). Undiscounted is the same without gamma
#             # Extract the state, action and reward from each episode tuple
#             timeStep = episodeStep[0]
#             state = episodeStep[1]
#             action = episodeStep[2]
#             rewardPlus1 = episodeSteps[timeStep + 1][3]

#             # Add the state to the statesSeen set
#             statesSeen.add(state)
            
#             discountedReturns = gamma*discountedReturns + rewardPlus1
#             undiscountedReturns = undiscountedReturns + rewardPlus1

#             # Check if the current state-action pair for the current episodeStep appeared in any of the previous tuples before this in the
#             # episodeSteps list
#             # Get all the visited states and actions
#             previousStates = [tup[1] for tup in episodeSteps[:timeStep - 1]]
#             previousActions = [tup[2] for tup in episodeSteps[:timeStep - 1]]
#             # Make this into a list of tuples
#             prevStateActionPair = list(map(lambda x, y: (x, y), previousStates, previousActions))
#             # Get current state, action as a tuple
#             currentStateActionPair = (state, action)
#             # Check if the current episode's state or action is in the visitedStates/Actions list until current index
#             if (currentStateActionPair not in prevStateActionPair):
#               # Find the qValue for the current state-action pair as the average as an incremental one
#               Q[state, action] =  Q[state, action] + learningRate*(discountedReturns - Q[state, action])
#               # Find the index of the action that gives the largest value
#               bestAction = np.argmax(Q[state,:])
#               # For all actions in the state
#               for action in range(env.get_action_size()):
#                 # Update the policy according to the epsilon-greedy policy
#                 # If current action index is the same as that of the best action
#                 if (action == bestAction):
#                   # Place more emphasize on this action, make policy have a greater value
#                   policy[state, action] = 1 - epsilon + (epsilon/env.get_action_size())
#                 else:
#                   policy[state, action] = epsilon/env.get_action_size()
                
#           # At the end of the episode, we want to find the value function for every state we've been in
#           for state in statesSeen:
#             V[state] = 0
#             # For every possible action
#             for action in range(env.get_action_size()):
#               V[state] += policy[state, action] * Q[state, action]

#           # Append the V to the value
#           values.append(V)

#           # Now that we're finished with the episode, we want to append the undiscounted reward
#           total_rewards.append(undiscountedReturns)

#         # At the end of each run, we want to append the total Values, total Policies and total Rewards for the run to the running sum
#         totalRValues.append(values)
#         totalRRewards.append(total_rewards)
#         totalPolicy.append(policy)
      
#       # Convert to numpy arrays so we can use numpy functions
#       totalPolicyNP = np.asarray(totalPolicy)
#       totalRValuesNP = np.asarray(totalRValues)
#       totalRRewardsNP = np.asarray(totalRRewards)

#       # Find the average of all runs
#       averagedPolicy = np.mean(totalPolicyNP, axis=0)
#       averagedValues = np.mean(totalRValuesNP, axis=0)
#       averagedRewards = np.mean(totalRRewardsNP, axis=0)

#       # Find the std of each of the runs
#       deviationPolicy = np.std(totalPolicyNP, axis=0)
#       deviationValue = np.std(totalRValuesNP, axis=0)
#       deviationRewards = np.std(totalRRewardsNP, axis=0)

#       return averagedPolicy, averagedValues, averagedRewards, deviationPolicy, deviationValue, deviationRewards

#     # [Action required]
#     # WARNING: make sure this function can be called by the auto-marking script
#     def solveAveraged(self, env):
#       """
#       Solve a given Maze environment using Monte Carlo learning
#       input: env {Maze object} -- Maze to solve
#       output: 
#         - policy {np.array} -- Optimal policy found to solve the given Maze environment 
#         - values {list of np.array} -- List of successive value functions for each episode 
#         - total_rewards {list of float} -- Corresponding list of successive total non-discounted sum of reward for each episode 
#       """

#       # Initialisation (can be edited)
#       Q = np.random.rand(env.get_state_size(), env.get_action_size()) 
#       V = np.zeros(env.get_state_size())
#       policy = np.zeros((env.get_state_size(), env.get_action_size())) 
#       # We initialize the policy to be an epsilon-soft policy, in which all actions have a probability of epsilon/number of actions to be selected
#       # Define epsilon to be 1
#       epsilon = 0.8
#       # In general, for all states and all actions, epsilon-soft gives them probability epsilon/cardinality(A) or 1 - epsilon + epsilon/cardinality(A)
#       # As an initial policy, we set that for all states, every action has an equal probability of being run
#       policy[:,:] = 0.25
#       values = [V]
#       total_rewards = []
#       # Define gamma 
#       gamma = env.get_gamma()
#       # Define learning rate
#       learningRate = 0.1
#       # Create a statesSeen to calculate V with it
#       statesSeen = set()

#       #### 
#       # Add your code here
#       # WARNING: this agent only has access to env.reset() and env.step()
#       # You should not use env.get_T(), env.get_R() or env.get_absorbing() to compute any value
#       ####

#       # We try first the on-policy first-visit MC control for epsilon-soft policies. Here, what we do is generate an arbitrarily large number of
#       # episodes, where for each episode we start using reset(), we then iterate by taking steps in the direction of our policy to generate the
#       # episode itself. This is repeated until we either reach an absorbing state, or take 500 steps.
#       ################### PHASE 0 - EPISODE GENERATION
#       # Define the number of episodes that we want to generate
#       numberOfEpisodes = 1000
#       # Set the random seed
#       np.random.seed(0)

#       # NOTE that instead of just doing 1000 episodes, what we do instead we make multiple runs, where every run is 1000 episodes. This is done to
#       # find a more representative show of the actual MC algorithm
#       # Define the number of runs
#       numberOfRuns = 30
#       # Define the total running rewards, policy and values
#       totalRValues = []
#       totalRRewards = []
#       totalPolicy = []
      
#       # For every run that we have
#       for run in range(numberOfRuns):
#         print('Run ', run)
#         # Reset V, values, policy and total_rewards in every run
#         V = np.zeros(env.get_state_size())
#         values = [V]
#         total_rewards = []
#         policy[:,:] = 0.25

#         # While we have less than numberOfEpisodes valid episodes
#         for episode in range(numberOfEpisodes):
#           # The epsilon is decaying
#           epsilon = (1 - episode/numberOfEpisodes)
#           # Define the list that will contain all the information/steps of this episode. Note that this will be in the form of a list of tuples, where
#           # each tuple will contain (timeStep, state, action, reward, done). Note that we define it within the episodes loop, as we want to reset it every ep
#           episodeSteps = []
#           # Reset to get the timestep, state, starting state, action, reward, and done boolean
#           timeStep, state, reward, done = env.reset()
#           # Iterate until ending criteria - either reach absorbing state (done == True), or 500 steps (timeStep >= 500)
#           while (done != True):
#             # For every episode, we want to take a step in the direction of the policy - we do this randomly with probability given by the policy
#             # We can use the np.random.choice() function for this, where the choice must be in the range of number of actions that we have, and the
#             # probability is defined by the entry of the action probabilities for that given state
#             action = np.random.choice(env.get_action_size(), p=policy[state, :])

#             # Append the step to the episodeSteps list
#             episodeSteps.append((timeStep, state, action, reward, done))

#             # Take a step in the direction of the action defined by the policy
#             timeStep, state, reward, done = env.step(action)
            
#           # Append the last reward to the list of episodeSteps, where the last step is basically gonna be the TERMINAL STATE, with the reward of
#           # that, and no action associated with it
#           episodeSteps.append((timeStep, state, None, reward, done)) 
          
#           ################### PHASE 1 - LOOPING OVER EPISODE
#           # Now that we've generated the episode, the next step is to iterate over the steps defined in the episode, where we define the actual
#           # on-policy first-visit MC policy
#           # Note that we create two returns values, one that is going to be used discounted, and one undiscounted
#           discountedReturns = 0
#           undiscountedReturns = 0
          
#           # NOTE that we only loop IF the last state was valid
#           # For every step (tuple) of the episodeSteps list. Notice that we reverse the order, so that we go from the end to the beginning
#           for episodeStep in reversed(list(episodeSteps[:len(episodeSteps) - 1])):
#             # The discounted returns is going to be gamme*discountedReturns + reward of next step (technically the same as the reward in
#             # the current tuple). Undiscounted is the same without gamma
#             # Extract the state, action and reward from each episode tuple
#             timeStep = episodeStep[0]
#             state = episodeStep[1]
#             action = episodeStep[2]
#             rewardPlus1 = episodeSteps[timeStep + 1][3]

#             # Add the state to the statesSeen set
#             statesSeen.add(state)
            
#             discountedReturns = gamma*discountedReturns + rewardPlus1
#             undiscountedReturns = undiscountedReturns + rewardPlus1

#             # Check if the current state-action pair for the current episodeStep appeared in any of the previous tuples before this in the
#             # episodeSteps list
#             # Get all the visited states and actions
#             previousStates = [tup[1] for tup in episodeSteps[:timeStep - 1]]
#             previousActions = [tup[2] for tup in episodeSteps[:timeStep - 1]]
#             # Make this into a list of tuples
#             prevStateActionPair = list(map(lambda x, y: (x, y), previousStates, previousActions))
#             # Get current state, action as a tuple
#             currentStateActionPair = (state, action)
#             # Check if the current episode's state or action is in the visitedStates/Actions list until current index
#             if (currentStateActionPair not in prevStateActionPair):
#               # Find the qValue for the current state-action pair as the average as an incremental one
#               Q[state, action] =  Q[state, action] + learningRate*(discountedReturns - Q[state, action])
#               # Find the index of the action that gives the largest value
#               bestAction = np.argmax(Q[state,:])
#               # For all actions in the state
#               for action in range(env.get_action_size()):
#                 # Update the policy according to the epsilon-greedy policy
#                 # If current action index is the same as that of the best action
#                 if (action == bestAction):
#                   # Place more emphasize on this action, make policy have a greater value
#                   policy[state, action] = 1 - epsilon + (epsilon/env.get_action_size())
#                 else:
#                   policy[state, action] = epsilon/env.get_action_size()
                
#           # At the end of the episode, we want to find the value function for every state we've been in
#           newV = np.zeros(env.get_state_size())
#           newV = copy.deepcopy(V)
#           for state in statesSeen:
#             newV[state] = 0
#             # For every possible action
#             for action in range(env.get_action_size()):
#               newV[state] += policy[state, action] * Q[state, action]

#           # Copy V to a list
#           values.append(np.copy(newV))

#           # Now that we're finished with the episode, we want to append the undiscounted reward
#           total_rewards.append(undiscountedReturns)

#         # At the end of each run, we want to append the total Values, total Policies and total Rewards for the run to the running sum
#         totalRValues.append(values)
#         totalRRewards.append(total_rewards)
#         totalPolicy.append(policy)
      
#       # Convert to numpy arrays so we can use numpy functions
#       totalPolicyNP = np.asarray(totalPolicy)
#       totalRValuesNP = np.asarray(totalRValues)
#       totalRRewardsNP = np.asarray(totalRRewards)

#       # Find the average of all runs
#       averagedPolicy = np.mean(totalPolicyNP, axis=0)
#       averagedValues = np.mean(totalRValuesNP, axis=0)
#       averagedRewards = np.mean(totalRRewardsNP, axis=0)

#       # Find the std of each of the runs
#       deviationPolicy = np.std(totalPolicyNP, axis=0)
#       deviationValue = np.std(totalRValuesNP, axis=0)
#       deviationRewards = np.std(totalRRewardsNP, axis=0)

#       return averagedPolicy, averagedValues, averagedRewards, deviationPolicy, deviationValue, deviationRewards

#     def solveAveragedDPCompare(self, env):
#       """
#       Solve a given Maze environment using Monte Carlo learning
#       input: env {Maze object} -- Maze to solve
#       output: 
#         - policy {np.array} -- Optimal policy found to solve the given Maze environment 
#         - values {list of np.array} -- List of successive value functions for each episode 
#         - total_rewards {list of float} -- Corresponding list of successive total non-discounted sum of reward for each episode 
#       """

#       # Initialisation (can be edited)
#       Q = np.random.rand(env.get_state_size(), env.get_action_size()) 
#       V = np.zeros(env.get_state_size())
#       policy = np.zeros((env.get_state_size(), env.get_action_size())) 
#       # We initialize the policy to be an epsilon-soft policy, in which all actions have a probability of epsilon/number of actions to be selected
#       # Define epsilon to be 1
#       epsilon = 0.8
#       # In general, for all states and all actions, epsilon-soft gives them probability epsilon/cardinality(A) or 1 - epsilon + epsilon/cardinality(A)
#       # As an initial policy, we set that for all states, every action has an equal probability of being run
#       policy[:,:] = 0.25
#       values = [V]
#       total_rewards = []
#       # Define gamma 
#       gamma = env.get_gamma()
#       # Define learning rate
#       learningRate = 0.1
#       # Create a statesSeen to calculate V with it
#       statesSeen = set()

#       #### 
#       # Add your code here
#       # WARNING: this agent only has access to env.reset() and env.step()
#       # You should not use env.get_T(), env.get_R() or env.get_absorbing() to compute any value
#       ####

#       # We try first the on-policy first-visit MC control for epsilon-soft policies. Here, what we do is generate an arbitrarily large number of
#       # episodes, where for each episode we start using reset(), we then iterate by taking steps in the direction of our policy to generate the
#       # episode itself. This is repeated until we either reach an absorbing state, or take 500 steps.
#       ################### PHASE 0 - EPISODE GENERATION
#       # Define the number of episodes that we want to generate
#       numberOfEpisodes = 1000
#       # Set the random seed
#       np.random.seed(0)

#       # NOTE that instead of just doing 1000 episodes, what we do instead we make multiple runs, where every run is 1000 episodes. This is done to
#       # find a more representative show of the actual MC algorithm
#       # Define the number of runs
#       numberOfRuns = 30
#       # Define the total running rewards, policy and values
#       totalRValues = []
#       totalRRewards = []
#       totalPolicy = []
#       # MC MSE
#       mc_MSE = []
#       mc_MSERuns = []

#       #DP
#       maze = Maze()
#       dp_agent = DP_agent()
#       dp_policy, dp_value, dp_epoch, dp_time = dp_agent.solveValue(maze)
      
#       # For every run that we have
#       for run in range(numberOfRuns):
#         print('Run ', run)
#         # Reset V, values, policy and total_rewards in every run
#         V = np.zeros(env.get_state_size())
#         values = [V]
#         total_rewards = []
#         policy[:,:] = 0.25
#         mc_MSE = []

#         # While we have less than numberOfEpisodes valid episodes
#         for episode in range(numberOfEpisodes):
#           # The epsilon is decaying
#           epsilon = (1 - episode/numberOfEpisodes)
#           # Define the list that will contain all the information/steps of this episode. Note that this will be in the form of a list of tuples, where
#           # each tuple will contain (timeStep, state, action, reward, done). Note that we define it within the episodes loop, as we want to reset it every ep
#           episodeSteps = []
#           # Reset to get the timestep, state, starting state, action, reward, and done boolean
#           timeStep, state, reward, done = env.reset()
          
#           # Iterate until ending criteria - either reach absorbing state (done == True), or 500 steps (timeStep >= 500)
#           while (done != True):
#             # For every episode, we want to take a step in the direction of the policy - we do this randomly with probability given by the policy
#             # We can use the np.random.choice() function for this, where the choice must be in the range of number of actions that we have, and the
#             # probability is defined by the entry of the action probabilities for that given state
#             action = np.random.choice(env.get_action_size(), p=policy[state, :])

#             # Append the step to the episodeSteps list
#             episodeSteps.append((timeStep, state, action, reward, done))

#             # Take a step in the direction of the action defined by the policy
#             timeStep, state, reward, done = env.step(action)
            
#           # Append the last reward to the list of episodeSteps, where the last step is basically gonna be the TERMINAL STATE, with the reward of
#           # that, and no action associated with it
#           episodeSteps.append((timeStep, state, None, reward, done)) 
          
#           ################### PHASE 1 - LOOPING OVER EPISODE
#           # Now that we've generated the episode, the next step is to iterate over the steps defined in the episode, where we define the actual
#           # on-policy first-visit MC policy
#           # Note that we create two returns values, one that is going to be used discounted, and one undiscounted
#           discountedReturns = 0
#           undiscountedReturns = 0
          
#           # NOTE that we only loop IF the last state was valid
#           # For every step (tuple) of the episodeSteps list. Notice that we reverse the order, so that we go from the end to the beginning
#           for episodeStep in reversed(list(episodeSteps[:len(episodeSteps) - 1])):
#             # The discounted returns is going to be gamme*discountedReturns + reward of next step (technically the same as the reward in
#             # the current tuple). Undiscounted is the same without gamma
#             # Extract the state, action and reward from each episode tuple
#             timeStep = episodeStep[0]
#             state = episodeStep[1]
#             action = episodeStep[2]
#             rewardPlus1 = episodeSteps[timeStep + 1][3]

#             # Add the state to the statesSeen set
#             statesSeen.add(state)
            
#             discountedReturns = gamma*discountedReturns + rewardPlus1
#             undiscountedReturns = undiscountedReturns + rewardPlus1

#             # Check if the current state-action pair for the current episodeStep appeared in any of the previous tuples before this in the
#             # episodeSteps list
#             # Get all the visited states and actions
#             previousStates = [tup[1] for tup in episodeSteps[:timeStep - 1]]
#             previousActions = [tup[2] for tup in episodeSteps[:timeStep - 1]]
#             # Make this into a list of tuples
#             prevStateActionPair = list(map(lambda x, y: (x, y), previousStates, previousActions))
#             # Get current state, action as a tuple
#             currentStateActionPair = (state, action)
#             # Check if the current episode's state or action is in the visitedStates/Actions list until current index
#             if (currentStateActionPair not in prevStateActionPair):
#               # Find the qValue for the current state-action pair as the average as an incremental one
#               Q[state, action] =  Q[state, action] + learningRate*(discountedReturns - Q[state, action])
#               # Find the index of the action that gives the largest value
#               bestAction = np.argmax(Q[state,:])
#               # For all actions in the state
#               for action in range(env.get_action_size()):
#                 # Update the policy according to the epsilon-greedy policy
#                 # If current action index is the same as that of the best action
#                 if (action == bestAction):
#                   # Place more emphasize on this action, make policy have a greater value
#                   policy[state, action] = 1 - epsilon + (epsilon/env.get_action_size())
#                 else:
#                   policy[state, action] = epsilon/env.get_action_size()
                
#           # At the end of the episode, we want to find the value function for every state we've been in
#           newV = np.zeros(env.get_state_size())
#           newV = copy.deepcopy(V)
#           for state in statesSeen:
#             newV[state] = 0
#             # For every possible action
#             for action in range(env.get_action_size()):
#               newV[state] += policy[state, action] * Q[state, action]

#           # Copy V to a list
#           values.append(np.copy(newV))

#           # Now that we're finished with the episode, we want to append the undiscounted reward
#           total_rewards.append(undiscountedReturns)

#           # Find MSE to DP at the end of each episode
#           mc_MSE.append(mean_squared_error(dp_value, newV))

#         # At the end of each run, we want to append the total Values, total Policies and total Rewards for the run to the running sum
#         totalRValues.append(values)
#         totalRRewards.append(total_rewards)
#         totalPolicy.append(policy)
#         mc_MSERuns.append(mc_MSE)
      
#       # Convert to numpy arrays so we can use numpy functions
#       totalPolicyNP = np.asarray(totalPolicy)
#       totalRValuesNP = np.asarray(totalRValues)
#       totalRRewardsNP = np.asarray(totalRRewards)

#       # Find the average of all runs
#       averagedPolicy = np.mean(totalPolicyNP, axis=0)
#       averagedValues = np.mean(totalRValuesNP, axis=0)
#       averagedRewards = np.mean(totalRRewardsNP, axis=0)

#       # Find the std of each of the runs
#       deviationPolicy = np.std(totalPolicyNP, axis=0)
#       deviationValue = np.std(totalRValuesNP, axis=0)
#       deviationRewards = np.std(totalRRewardsNP, axis=0)

#       mc_MSERunsNP = np.asarray(mc_MSERuns)
#       averageMSE = np.mean(mc_MSERunsNP, axis=0)
#       deviationMSE = np.std(mc_MSERunsNP, axis=0)

#       return averagedPolicy, averagedValues, averagedRewards, deviationPolicy, deviationValue, deviationRewards, averageMSE, deviationMSE
    
#     def solveVarLR(self, env, varLR):
#       """
#       Solve a given Maze environment using Monte Carlo learning
#       input:  - env {Maze object} -- Maze to solve
#               - Gamma {float} -- Used to determine how much we discount the return by 
#       output: 
#         - policy {np.array} -- Optimal policy found to solve the given Maze environment 
#         - values {list of np.array} -- List of successive value functions for each episode 
#         - total_rewards {list of float} -- Corresponding list of successive total non-discounted sum of reward for each episode 
#       """

#           # Initialisation (can be edited)
#       Q = np.random.rand(env.get_state_size(), env.get_action_size()) 
#       V = np.zeros(env.get_state_size())
#       policy = np.zeros((env.get_state_size(), env.get_action_size())) 
#       # We initialize the policy to be an epsilon-soft policy, in which all actions have a probability of epsilon/number of actions to be selected
#       # Define epsilon to be 1
#       epsilon = 0.8
#       # In general, for all states and all actions, epsilon-soft gives them probability epsilon/cardinality(A) or 1 - epsilon + epsilon/cardinality(A)
#       # As an initial policy, we set that for all states, every action has an equal probability of being run
#       policy[:,:] = 0.25
#       values = [V]
#       total_rewards = []
#       # Define gamma 
#       gamma = env.get_gamma()
#       # Define learning rate
#       learningRate = varLR
#       # Create a statesSeen to calculate V with it
#       statesSeen = set()

#       #### 
#       # Add your code here
#       # WARNING: this agent only has access to env.reset() and env.step()
#       # You should not use env.get_T(), env.get_R() or env.get_absorbing() to compute any value
#       ####

#       # We try first the on-policy first-visit MC control for epsilon-soft policies. Here, what we do is generate an arbitrarily large number of
#       # episodes, where for each episode we start using reset(), we then iterate by taking steps in the direction of our policy to generate the
#       # episode itself. This is repeated until we either reach an absorbing state, or take 500 steps.
#       ################### PHASE 0 - EPISODE GENERATION
#       # Define the number of episodes that we want to generate
#       numberOfEpisodes = 6000
#       # Set the random seed
#       np.random.seed(0)

#       # Start time
#       start_time = time.time()

#       # While we have less than numberOfEpisodes valid episodes
#       for episode in range(numberOfEpisodes):
#         # The epsilon is decaying
#         epsilon = (1 - episode/numberOfEpisodes)
#         # Define the list that will contain all the information/steps of this episode. Note that this will be in the form of a list of tuples, where
#         # each tuple will contain (timeStep, state, action, reward, done). Note that we define it within the episodes loop, as we want to reset it every ep
#         episodeSteps = []
#         # Reset to get the timestep, state, starting state, action, reward, and done boolean
#         timeStep, state, reward, done = env.reset()
#         # Iterate until ending criteria - either reach absorbing state (done == True), or 500 steps (timeStep >= 500)
#         while (done != True):
#           # For every episode, we want to take a step in the direction of the policy - we do this randomly with probability given by the policy
#           # We can use the np.random.choice() function for this, where the choice must be in the range of number of actions that we have, and the
#           # probability is defined by the entry of the action probabilities for that given state
#           action = np.random.choice(env.get_action_size(), p=policy[state, :])

#           # Append the step to the episodeSteps list
#           episodeSteps.append((timeStep, state, action, reward, done))

#           # Take a step in the direction of the action defined by the policy
#           timeStep, state, reward, done = env.step(action)
          
#         # Append the last reward to the list of episodeSteps, where the last step is basically gonna be the TERMINAL STATE, with the reward of
#         # that, and no action associated with it
#         episodeSteps.append((timeStep, state, None, reward, done)) 
        
#         ################### PHASE 1 - LOOPING OVER EPISODE
#         # Now that we've generated the episode, the next step is to iterate over the steps defined in the episode, where we define the actual
#         # on-policy first-visit MC policy
#         # Note that we create two returns values, one that is going to be used discounted, and one undiscounted
#         discountedReturns = 0
#         undiscountedReturns = 0
        
#         # NOTE that we only loop IF the last state was valid
#         # For every step (tuple) of the episodeSteps list. Notice that we reverse the order, so that we go from the end to the beginning
#         for episodeStep in reversed(list(episodeSteps[:len(episodeSteps) - 1])):
#           # The discounted returns is going to be gamme*discountedReturns + reward of next step (technically the same as the reward in
#           # the current tuple). Undiscounted is the same without gamma
#           # Extract the state, action and reward from each episode tuple
#           timeStep = episodeStep[0]
#           state = episodeStep[1]
#           action = episodeStep[2]
#           rewardPlus1 = episodeSteps[timeStep + 1][3]

#           # Add the state to the statesSeen set
#           statesSeen.add(state)
          
#           discountedReturns = gamma*discountedReturns + rewardPlus1
#           undiscountedReturns = undiscountedReturns + rewardPlus1

#           # Check if the current state-action pair for the current episodeStep appeared in any of the previous tuples before this in the
#           # episodeSteps list
#           # Get all the visited states and actions
#           previousStates = [tup[1] for tup in episodeSteps[:timeStep - 1]]
#           previousActions = [tup[2] for tup in episodeSteps[:timeStep - 1]]
#           # Make this into a list of tuples
#           prevStateActionPair = list(map(lambda x, y: (x, y), previousStates, previousActions))
#           # Get current state, action as a tuple
#           currentStateActionPair = (state, action)
#           # Check if the current episode's state or action is in the visitedStates/Actions list until current index
#           if (currentStateActionPair not in prevStateActionPair):
#             # Find the qValue for the current state-action pair as the average as an incremental one
#             Q[state, action] =  Q[state, action] + learningRate*(discountedReturns - Q[state, action])
#             # Find the index of the action that gives the largest value
#             bestAction = np.argmax(Q[state,:])
#             # For all actions in the state
#             for action in range(env.get_action_size()):
#               # Update the policy according to the epsilon-greedy policy
#               # If current action index is the same as that of the best action
#               if (action == bestAction):
#                 # Place more emphasize on this action, make policy have a greater value
#                 policy[state, action] = 1 - epsilon + (epsilon/env.get_action_size())
#               else:
#                 policy[state, action] = epsilon/env.get_action_size()
              
#         # At the end of the episode, we want to find the value function for every state we've been in
#         for state in statesSeen:
#           V[state] = 0
#           # For every possible action
#           for action in range(env.get_action_size()):
#             V[state] += policy[state, action] * Q[state, action]

#         # Append the V to the value
#         values.append(V)

#         # Now that we're finished with the episode, we want to append the undiscounted reward
#         total_rewards.append(undiscountedReturns)

#       totalTime = time.time() - start_time

#       return policy, values, total_rewards, totalTime
    
#     def solveVarEp(self, env, varEp):
#       """
#       Solve a given Maze environment using Monte Carlo learning
#       input:  - env {Maze object} -- Maze to solve
#               - Gamma {float} -- Used to determine how much we discount the return by 
#       output: 
#         - policy {np.array} -- Optimal policy found to solve the given Maze environment 
#         - values {list of np.array} -- List of successive value functions for each episode 
#         - total_rewards {list of float} -- Corresponding list of successive total non-discounted sum of reward for each episode 
#       """

#           # Initialisation (can be edited)
#       Q = np.random.rand(env.get_state_size(), env.get_action_size()) 
#       V = np.zeros(env.get_state_size())
#       policy = np.zeros((env.get_state_size(), env.get_action_size())) 
#       # We initialize the policy to be an epsilon-soft policy, in which all actions have a probability of epsilon/number of actions to be selected
#       # Define epsilon to be 1
#       epsilon = varEp
#       # In general, for all states and all actions, epsilon-soft gives them probability epsilon/cardinality(A) or 1 - epsilon + epsilon/cardinality(A)
#       # As an initial policy, we set that for all states, every action has an equal probability of being run
#       policy[:,:] = 0.25
#       values = [V]
#       total_rewards = []
#       # Define gamma 
#       gamma = env.get_gamma()
#       # Define learning rate
#       learningRate = 0.1
#       # Create a statesSeen to calculate V with it
#       statesSeen = set()

#       #### 
#       # Add your code here
#       # WARNING: this agent only has access to env.reset() and env.step()
#       # You should not use env.get_T(), env.get_R() or env.get_absorbing() to compute any value
#       ####

#       # We try first the on-policy first-visit MC control for epsilon-soft policies. Here, what we do is generate an arbitrarily large number of
#       # episodes, where for each episode we start using reset(), we then iterate by taking steps in the direction of our policy to generate the
#       # episode itself. This is repeated until we either reach an absorbing state, or take 500 steps.
#       ################### PHASE 0 - EPISODE GENERATION
#       # Define the number of episodes that we want to generate
#       numberOfEpisodes = 6000
#       # Set the random seed
#       np.random.seed(0)

#       # Start time
#       start_time = time.time()

#       # While we have less than numberOfEpisodes valid episodes
#       for episode in range(numberOfEpisodes):
#         # Define the list that will contain all the information/steps of this episode. Note that this will be in the form of a list of tuples, where
#         # each tuple will contain (timeStep, state, action, reward, done). Note that we define it within the episodes loop, as we want to reset it every ep
#         episodeSteps = []
#         # Reset to get the timestep, state, starting state, action, reward, and done boolean
#         timeStep, state, reward, done = env.reset()
#         # Iterate until ending criteria - either reach absorbing state (done == True), or 500 steps (timeStep >= 500)
#         while (done != True):
#           # For every episode, we want to take a step in the direction of the policy - we do this randomly with probability given by the policy
#           # We can use the np.random.choice() function for this, where the choice must be in the range of number of actions that we have, and the
#           # probability is defined by the entry of the action probabilities for that given state
#           action = np.random.choice(env.get_action_size(), p=policy[state, :])

#           # Append the step to the episodeSteps list
#           episodeSteps.append((timeStep, state, action, reward, done))

#           # Take a step in the direction of the action defined by the policy
#           timeStep, state, reward, done = env.step(action)
          
#         # Append the last reward to the list of episodeSteps, where the last step is basically gonna be the TERMINAL STATE, with the reward of
#         # that, and no action associated with it
#         episodeSteps.append((timeStep, state, None, reward, done)) 
        
#         ################### PHASE 1 - LOOPING OVER EPISODE
#         # Now that we've generated the episode, the next step is to iterate over the steps defined in the episode, where we define the actual
#         # on-policy first-visit MC policy
#         # Note that we create two returns values, one that is going to be used discounted, and one undiscounted
#         discountedReturns = 0
#         undiscountedReturns = 0
        
#         # NOTE that we only loop IF the last state was valid
#         # For every step (tuple) of the episodeSteps list. Notice that we reverse the order, so that we go from the end to the beginning
#         for episodeStep in reversed(list(episodeSteps[:len(episodeSteps) - 1])):
#           # The discounted returns is going to be gamme*discountedReturns + reward of next step (technically the same as the reward in
#           # the current tuple). Undiscounted is the same without gamma
#           # Extract the state, action and reward from each episode tuple
#           timeStep = episodeStep[0]
#           state = episodeStep[1]
#           action = episodeStep[2]
#           rewardPlus1 = episodeSteps[timeStep + 1][3]

#           # Add the state to the statesSeen set
#           statesSeen.add(state)
          
#           discountedReturns = gamma*discountedReturns + rewardPlus1
#           undiscountedReturns = undiscountedReturns + rewardPlus1

#           # Check if the current state-action pair for the current episodeStep appeared in any of the previous tuples before this in the
#           # episodeSteps list
#           # Get all the visited states and actions
#           previousStates = [tup[1] for tup in episodeSteps[:timeStep - 1]]
#           previousActions = [tup[2] for tup in episodeSteps[:timeStep - 1]]
#           # Make this into a list of tuples
#           prevStateActionPair = list(map(lambda x, y: (x, y), previousStates, previousActions))
#           # Get current state, action as a tuple
#           currentStateActionPair = (state, action)
#           # Check if the current episode's state or action is in the visitedStates/Actions list until current index
#           if (currentStateActionPair not in prevStateActionPair):
#             # Find the qValue for the current state-action pair as the average as an incremental one
#             Q[state, action] =  Q[state, action] + learningRate*(discountedReturns - Q[state, action])
#             # Find the index of the action that gives the largest value
#             bestAction = np.argmax(Q[state,:])
#             # For all actions in the state
#             for action in range(env.get_action_size()):
#               # Update the policy according to the epsilon-greedy policy
#               # If current action index is the same as that of the best action
#               if (action == bestAction):
#                 # Place more emphasize on this action, make policy have a greater value
#                 policy[state, action] = 1 - epsilon + (epsilon/env.get_action_size())
#               else:
#                 policy[state, action] = epsilon/env.get_action_size()
              
#         # At the end of the episode, we want to find the value function for every state we've been in
#         for state in statesSeen:
#           V[state] = 0
#           # For every possible action
#           for action in range(env.get_action_size()):
#             V[state] += policy[state, action] * Q[state, action]

#         # Append the V to the value
#         values.append(V)

#         # Now that we're finished with the episode, we want to append the undiscounted reward
#         total_rewards.append(undiscountedReturns)

#       totalTime = time.time() - start_time

#       return policy, values, total_rewards, totalTime

#   ################## AVERAGING
#   maze = Maze()
#   mc_agent = MC_agent()

#   averagedPolicy, averagedValues, averagedRewards, deviationPolicy, deviationValue, deviationRewards = mc_agent.solveAveraged(maze)

#   print("Results of the MC agent:\n")
#   maze.get_graphics().draw_policy(averagedPolicy, "Policy for On-Policy Averaged MC")
#   maze.get_graphics().draw_value(averagedValues[-1], "Value Function for On-Policy Averaged MC")

#   # Since we know we have 1000 episodes
#   episodes_range = np.arange(1000)
#   plt.plot(episodes_range, averagedRewards, label="Mean of Rewards")
#   plt.fill_between(episodes_range, averagedRewards - deviationRewards, averagedRewards + deviationRewards, label="STD of Rewards", color="lightsteelblue")
#   plt.ylabel('Rewards')
#   plt.xlabel('Number of Runs')
#   plt.title('Mean and STD of Rewards from On-Policy MC')
#   plt.legend()
#   plt.show()

#   ################## FIND THE NUMBER OF REPLICATION
#   maze = Maze()
#   mc_agent = MC_agent()
#   policy, values, total_rewards, averageDifference, numberOfReplications = mc_agent.solveONFindRep(maze)

#   print("Results of the MC agent:\n")
#   maze.get_graphics().draw_policy(policy, "Policy for On-Policy MC")
#   maze.get_graphics().draw_value(values[-1], "Value Function for On-Policy MC")
#   print("The number of replications is ", numberOfReplications)

#   plt.plot(averageDifference[1:])
#   plt.ylabel('Average difference of Value Function V(s)')
#   plt.xlabel('Number of Runs')
#   plt.title('Value Function V(s) values until convergence')
#   plt.legend()
#   plt.show()

#   # Defining for the plots
#   # plt.plot(total_rewards)
#   # plt.ylabel('Rewards')
#   # plt.xlabel('Episode')
#   # plt.title('Total Undiscounted Rewards - 1000 Episodes')
#   # plt.legend()
#   # plt.show()

#   ################### Impact of LR and epsilon on the MC algorithm
#   LR_range = [0.01, 0.1, 0.2, 0.4]
#   avgPolicies = []
#   avgValues = []
#   avgRew = []
#   avgRewDeviation = []
#   titles = []

#   maze = Maze()
#   mc_agent = MC_agent()

#   # Use value iteration for each LR value
#   for LR in LR_range:
#       averagedPolicy, averagedValues, averagedRewards, deviationPolicy, deviationValue, deviationRewards = mc_agent.solveAveragedVarLRReps(maze, varLR = LR)
#       avgPolicies.append(averagedPolicy)
#       avgValues.append(averagedValues[-1])
#       avgRew.append(averagedRewards)
#       avgRewDeviation.append(deviationRewards)
#       print('At LR ', LR)

#   # Show the learning curve for every one
#   episodes_range = np.arange(1000)

#   for i in range(len(avgRew)):
#       plt.plot(episodes_range, avgRew[i], label="Mean of Rewards for LR = {}".format(LR_range[i]))
#       plt.fill_between(episodes_range, avgRew[i] - avgRewDeviation[i], avgRew[i] + avgRewDeviation[i], label="STD of Rewards", color="lightsteelblue")
#       plt.ylabel('Rewards')
#       plt.xlabel('Number of Runs')
#       plt.title('Mean and STD of Rewards from On-Policy MC for LR = {}'.format(LR_range[i]))
#       plt.legend()
#       plt.show()
  
#   EP_range = [0.2, 0.4, 0.6, 0.8]
#   avgPolicies = []
#   avgValues = []
#   avgRew = []
#   avgRewDeviation = []
#   titles = []

#   maze = Maze()
#   mc_agent = MC_agent()

#   # Use value iteration for each Ep value
#   for Ep in EP_range:
#       averagedPolicy, averagedValues, averagedRewards, deviationPolicy, deviationValue, deviationRewards = mc_agent.solveAveragedVarEpsilonReps(maze, varEp = Ep)
#       avgPolicies.append(averagedPolicy)
#       avgValues.append(averagedValues[-1])
#       avgRew.append(averagedRewards)
#       avgRewDeviation.append(deviationRewards)
#       print('At Epsilon ', Ep)

#   # Show the learning curve for every one
#   episodes_range = np.arange(1000)

#   for i in range(len(avgRew)):
#       plt.plot(episodes_range, avgRew[i], label="Mean of Rewards for Epsilon = {}".format(EP_range[i]))
#       plt.fill_between(episodes_range, avgRew[i] - avgRewDeviation[i], avgRew[i] + avgRewDeviation[i], label="STD of Rewards", color="lightsteelblue")
#       plt.ylabel('Rewards')
#       plt.xlabel('Number of Runs')
#       plt.title('Mean and STD of Rewards from On-Policy MC for Epsilon = {}'.format(EP_range[i]))
#       plt.legend()
#       plt.show()
  
#   ################## SHOWING POLICY
#   maze = Maze()
#   mc_agent = MC_agent()
#   mc_policy, mc_values, total_rewards = mc_agent.solveON(maze)

#   print("Results of the MC agent:\n")
#   maze.get_graphics().draw_policy(mc_policy, "Policy for On-Policy MC")
#   maze.get_graphics().draw_value(mc_values[-1], "Value Function for On-Policy MC")

#   # Defining for the plots
#   plt.plot(total_rewards)
#   plt.ylabel('Rewards')
#   plt.xlabel('Episode')
#   plt.title('Total Undiscounted Rewards - 1000 Episodes')
#   plt.legend()
#   plt.show()

#   ################# SHOWING OFF POLICY
#   maze = Maze()
#   mc_agent = MC_agent()
#   mc_policy, mc_values, total_rewards = mc_agent.solveOFF(maze)

#   print("Results of the MC agent:\n")
#   maze.get_graphics().draw_policy(mc_policy, "Policy for Off-Policy MC")
#   maze.get_graphics().draw_value(mc_values[-1], "Value Function for Off-Policy MC")

#   # Defining for the plots
#   plt.plot(total_rewards)
#   plt.ylabel('Rewards')
#   plt.xlabel('Episode')
#   plt.title('Total Undiscounted Rewards - 1000 Episodes')
#   plt.legend()
#   plt.show()

#   # This class define the Temporal-Difference agent
#   class TD_agent(object):

#     # [Action required]
#     # WARNING: make sure this function can be called by the auto-marking script
#     def solveON(self, env):
#       """
#       Solve a given Maze environment using Temporal Difference learning
#       input: env {Maze object} -- Maze to solve
#       output: 
#         - policy {np.array} -- Optimal policy found to solve the given Maze environment 
#         - values {list of np.array} -- List of successive value functions for each episode 
#         - total_rewards {list of float} -- Corresponding list of successive total non-discounted sum of reward for each episode 
#       """

#       # Initialisation (can be edited)
#       Q = np.random.rand(env.get_state_size(), env.get_action_size()) 
#       V = np.zeros(env.get_state_size())
#       policy = np.zeros((env.get_state_size(), env.get_action_size())) 
#       # We initialize the policy to be an epsilon-soft policy, in which all actions have a probability of epsilon/number of actions to be selected
#       # Define epsilon to be decaying
#       epsilon = 1
#       # Define for all states and all actions, epsilon-soft gives them probability epsilon/cardinality(A) or 1 - epsilon + epsilon/cardinality(A)
#       # As an initial policy, we set that for all states, the first action will have the highest probability of being chosen
#       # policy[:, 0] = 1 - epsilon + epsilon/env.get_action_size()
#       # policy[:, 1:] = epsilon / env.get_action_size()
#       policy[:,:] = 0.25
#       values = [V]
#       total_rewards = []
#       # Define gamma 
#       gamma = env.get_gamma()
#       # Define learning rate
#       learningRate = 0.2
#       # Define set of states seen in episode
#       statesSeen = set()

#       #### 
#       # Add your code here
#       # WARNING: this agent only has access to env.reset() and env.step()
#       # You should not use env.get_T(), env.get_R() or env.get_absorbing() to compute any value
#       ####

#       # In on-policy TD control, or SARSA, the idea is to loop over all possible episodes, and then for each episode, we first generate an S,
#       # then choose an A based on epsilon-greedy policy. Then, for that episode, for every step, we take an action and see results, meaning
#       # we see nextState. Then, from nextState, we generate nextAction. Then the Q(s|a) will be:
#       # Q(s|a) = Q(s|a) + learningRate(Return + gamma(Q(s'|a') - Q(s|a)))
#       # Then we set the current state to next state, and action to next action. We do this for each episode until S is terminal
#       # Don't forget to also update the policy once we finish finding Q!

#       ################### PHASE 0 - EPISODE GENERATION
#       # Define the number of episodes that we want to generate
#       numberOfEpisodes = 1000
#       # Timer for knowing when we start
#       start_time = time.time()
      
#       # For number of episodes that we have
#       for episode in range(numberOfEpisodes):
#         # Define the epsilon
#         epsilon = (1 - episode/numberOfEpisodes)
#         # Define the list that will contain all the information/steps of this episode. Note that this will be in the form of a list of tuples, where
#         # each tuple will contain (timeStep, state, action, reward, done). Note that we define it within the episodes loop, as we want to reset it every ep
#         episodeSteps = []
#         # Setting the value of undiscounted reward to 0 for every episode
#         undiscountedRewards = 0

#         # Reset to get the timestep, state, starting state, action, reward, and done boolean
#         timeStep, state, reward, done = env.reset()
#         # Perform the first action, as according to the probability defined according to the policy for this state
#         action = np.random.choice(env.get_action_size(), p=policy[state, :])
#         episodeSteps.append((timeStep, state, action, reward, done))

#         # Append state to set of states for this episode
#         statesSeen.add(state)

#         # Update the undiscounted rewards
#         undiscountedRewards = undiscountedRewards + reward

#         # Iterate until ending criteria - either reach absorbing state (done == True), or 500 steps (timeStep >= 500)
#         while (done != True):

#           # Take a step in the direction of the action defined by the policy
#           timeStep, nextState, reward, done = env.step(action)
#           # Get the next action from the nextState
#           nextAction = np.random.choice(env.get_action_size(), p=policy[nextState, :])
#           # Append the newly found nextState, nextAction pair to the episodeSteps
#           episodeSteps.append((timeStep, nextState, nextAction, reward, done))
#           # Find the Q(s|a)
#           Q[state, action] = Q[state, action] + learningRate*(reward + gamma*Q[nextState, nextAction] - Q[state, action])
          
#           # Update the undiscounted rewards
#           undiscountedRewards = undiscountedRewards + reward
#           # Append state to set of states for this episode
#           statesSeen.add(nextState)
          
#           # Find the index of the action that gives the largest value
#           bestAction = np.argmax(Q[state,:])
#           # For all actions in the state
#           for action in range(env.get_action_size()):
#             # Update the policy according to the epsilon-greedy policy
#             # If current action index is the same as that of the best action
#             if (action == bestAction):
#               # Place more emphasize on this action, make policy have a greater value
#               policy[state, action] = 1 - epsilon + (epsilon/env.get_action_size())
#             else:
#               policy[state, action] = epsilon/env.get_action_size()
            
#           # Update the next state, next action
#           state = nextState
#           action = nextAction
        
#         # At the end of the episode, we want to find the value function for every state we've been in
#         newV = np.zeros(env.get_state_size())
#         newV = copy.deepcopy(V)
#         for state in statesSeen:
#           newV[state] = 0
#           # For every possible action
#           for action in range(env.get_action_size()):
#             newV[state] += policy[state, action] * Q[state, action]

#         # Copy V to a list
#         values.append(np.copy(newV))

#         # Append the undiscountedRewards to the total_rewards
#         total_rewards.append(undiscountedRewards)
        
#       totalTime = time.time() - start_time

#       return policy, values, total_rewards, totalTime

#     def solveOFF(self, env):
#       """
#       Solve a given Maze environment using Temporal Difference learning
#       input: env {Maze object} -- Maze to solve
#       output: 
#         - policy {np.array} -- Optimal policy found to solve the given Maze environment 
#         - values {list of np.array} -- List of successive value functions for each episode 
#         - total_rewards {list of float} -- Corresponding list of successive total non-discounted sum of reward for each episode 
#       """

#       # Initialisation (can be edited)
#       Q = np.random.rand(env.get_state_size(), env.get_action_size()) 
#       V = np.zeros(env.get_state_size())
#       policy = np.zeros((env.get_state_size(), env.get_action_size())) 
#       # We initialize the policy to be an epsilon-soft policy, in which all actions have a probability of epsilon/number of actions to be selected
#       # Define epsilon to be 0.1
#       epsilon = 1
#       # Define for all states and all actions, epsilon-soft gives them probability epsilon/cardinality(A) or 1 - epsilon + epsilon/cardinality(A)
#       # As an initial policy, we set that for all states, the first action will have the highest probability of being chosen
#       # policy[:, 0] = 1 - epsilon + epsilon/env.get_action_size()
#       # policy[:, 1:] = epsilon / env.get_action_size()
#       policy[:,:] = 0.25
#       values = [V]
#       total_rewards = []
#       # Define gamma 
#       gamma = env.get_gamma()
#       # Define learning rate
#       learningRate = 0.2
#       # Define set of states seen in episode
#       statesSeen = set()

#       #### 
#       # Add your code here
#       # WARNING: this agent only has access to env.reset() and env.step()
#       # You should not use env.get_T(), env.get_R() or env.get_absorbing() to compute any value
#       ####

#       # In off-policy TD Control, the idea is that we initialize Q, S and set the Q for all terminal states to 0. Then, we initialize the first
#       # state. In a loop, we then choose an action A based on our state, take the action to get the reward and nextState, then update Q as
#       # Q(s|a) = Q(s|a) + LR*(reward + gamma*maxQ(s'|a) over all a - Q(s|a))
#       # Then set the state to the next state
#       # Timer for knowing when we start
#       start_time = time.time()

#       ################### PHASE 0 - EPISODE GENERATION
#       # Define the number of episodes that we want to generate
#       numberOfEpisodes = 1000
#       # For number of episodes that we have
#       for episode in range(numberOfEpisodes):
#         # Define epsilon to be decaying
#         epsilon = (1 - episode/numberOfEpisodes)
#         # Define the list that will contain all the information/steps of this episode. Note that this will be in the form of a list of tuples, where
#         # each tuple will contain (timeStep, state, action, reward, done). Note that we define it within the episodes loop, as we want to reset it every ep
#         episodeSteps = []
#         # Setting the value of undiscounted reward to 0 for every episode
#         undiscountedRewards = 0

#         # Reset to get the timestep, state, starting state, action, reward, and done boolean
#         timeStep, state, reward, done = env.reset()

#         # Append state to set of states for this episode
#         statesSeen.add(state)

#         # Update the undiscounted rewards
#         undiscountedRewards = undiscountedRewards + reward

#         # Iterate until ending criteria - either reach absorbing state (done == True), or 500 steps (timeStep >= 500)
#         while (done != True):
          
#           # Find the action from the state
#           action = np.random.choice(env.get_action_size(), p=policy[state, :])
#           # Append the state,action pair to the episodeSteps
#           episodeSteps.append((timeStep, state, action, reward, done))

#           # Take a step in the direction of the action defined by the policy
#           timeStep, nextState, reward, done = env.step(action)
          
#           # Find the Q(s|a), this time with the maximum value over all actions for the next state
#           Q[state, action] = Q[state, action] + learningRate*(reward + gamma*np.max(Q[nextState,:]) - Q[state, action])
          
#           # Update the undiscounted rewards
#           undiscountedRewards = undiscountedRewards + reward

#           # Append state to set of states for this episode
#           statesSeen.add(nextState)
          
#           # Find the index of the action that gives the largest value
#           bestAction = np.argmax(Q[state,:])
#           # For all actions in the state
#           for action in range(env.get_action_size()):
#             # Update the policy according to the epsilon-greedy policy
#             # If current action index is the same as that of the best action
#             if (action == bestAction):
#               # Place more emphasize on this action, make policy have a greater value
#               policy[state, action] = 1 - epsilon + (epsilon/env.get_action_size())
#             else:
#               policy[state, action] = epsilon/env.get_action_size()
            
#           # Update the next state
#           state = nextState
        
#         # At the end of the episode, we want to find the value function for every state we've been in
#         newV = np.zeros(env.get_state_size())
#         newV = copy.deepcopy(V)
#         for state in statesSeen:
#           newV[state] = 0
#           # For every possible action
#           for action in range(env.get_action_size()):
#             newV[state] += policy[state, action] * Q[state, action]

#         # Copy V to a list
#         values.append(np.copy(newV))

#         # Append the undiscountedRewards to the total_rewards
#         total_rewards.append(undiscountedRewards)
      
#       totalTime = time.time() - start_time

#       return policy, values, total_rewards, totalTime
    
#     def solveONAverage(self, env):
#       """
#       Solve a given Maze environment using Temporal Difference learning
#       input: env {Maze object} -- Maze to solve
#       output: 
#         - policy {np.array} -- Optimal policy found to solve the given Maze environment 
#         - values {list of np.array} -- List of successive value functions for each episode 
#         - total_rewards {list of float} -- Corresponding list of successive total non-discounted sum of reward for each episode 
#       """

#       # Initialisation (can be edited)
#       Q = np.random.rand(env.get_state_size(), env.get_action_size()) 
#       V = np.zeros(env.get_state_size())
#       policy = np.zeros((env.get_state_size(), env.get_action_size())) 
#       # We initialize the policy to be an epsilon-soft policy, in which all actions have a probability of epsilon/number of actions to be selected
#       # Define epsilon to be 0.1
#       epsilon = 1
#       # Define for all states and all actions, epsilon-soft gives them probability epsilon/cardinality(A) or 1 - epsilon + epsilon/cardinality(A)
#       # As an initial policy, we set that for all states, the first action will have the highest probability of being chosen
#       # policy[:, 0] = 1 - epsilon + epsilon/env.get_action_size()
#       # policy[:, 1:] = epsilon / env.get_action_size()
#       policy[:,:] = 0.25
#       values = [V]
#       total_rewards = []
#       # Define gamma 
#       gamma = env.get_gamma()
#       # Define learning rate
#       learningRate = 0.2
#       # Define set of states seen in episode
#       statesSeen = set()
#       # Define the list of totals
#       totalRValues = []
#       totalRRewards = []
#       totalPolicy = []

#       #### 
#       # Add your code here
#       # WARNING: this agent only has access to env.reset() and env.step()
#       # You should not use env.get_T(), env.get_R() or env.get_absorbing() to compute any value
#       ####

#       # In off-policy TD Control, the idea is that we initialize Q, S and set the Q for all terminal states to 0. Then, we initialize the first
#       # state. In a loop, we then choose an action A based on our state, take the action to get the reward and nextState, then update Q as
#       # Q(s|a) = Q(s|a) + LR*(reward + gamma*maxQ(s'|a) over all a - Q(s|a))
#       # Then set the state to the next state
#       # Timer for knowing when we start
#       start_time = time.time()

#       ################### PHASE 0 - EPISODE GENERATION
#       # Define the number of episodes that we want to generate
#       numberOfEpisodes = 1000
#       # Define the number of runs
#       numberOfRuns = 30

#       for run in range(numberOfRuns):
#         print('Run ', run)
#         # Reset V, values, policy and total_rewards in every run
#         V = np.zeros(env.get_state_size())
#         values = [V]
#         total_rewards = []
#         policy[:,:] = 0.25
        
#         # For number of episodes that we have
#         for episode in range(numberOfEpisodes):
#           # Define epsilon to be decaying
#           epsilon = (1 - episode/numberOfEpisodes)
#           # Define the list that will contain all the information/steps of this episode. Note that this will be in the form of a list of tuples, where
#           # each tuple will contain (timeStep, state, action, reward, done). Note that we define it within the episodes loop, as we want to reset it every ep
#           episodeSteps = []
#           # Setting the value of undiscounted reward to 0 for every episode
#           undiscountedRewards = 0

#           # Reset to get the timestep, state, starting state, action, reward, and done boolean
#           timeStep, state, reward, done = env.reset()
#           # Perform the first action, as according to the probability defined according to the policy for this state
#           action = np.random.choice(env.get_action_size(), p=policy[state, :])
#           episodeSteps.append((timeStep, state, action, reward, done))

#           # Append state to set of states for this episode
#           statesSeen.add(state)

#           # Update the undiscounted rewards
#           undiscountedRewards = undiscountedRewards + reward

#           # Iterate until ending criteria - either reach absorbing state (done == True), or 500 steps (timeStep >= 500)
#           while (done != True):

#             # Take a step in the direction of the action defined by the policy
#             timeStep, nextState, reward, done = env.step(action)
#             # Get the next action from the nextState
#             nextAction = np.random.choice(env.get_action_size(), p=policy[nextState, :])
#             # Append the newly found nextState, nextAction pair to the episodeSteps
#             episodeSteps.append((timeStep, nextState, nextAction, reward, done))
#             # Find the Q(s|a)
#             Q[state, action] = Q[state, action] + learningRate*(reward + gamma*Q[nextState, nextAction] - Q[state, action])
            
#             # Update the undiscounted rewards
#             undiscountedRewards = undiscountedRewards + reward
#             # Append state to set of states for this episode
#             statesSeen.add(nextState)
            
#             # Find the index of the action that gives the largest value
#             bestAction = np.argmax(Q[state,:])
#             # For all actions in the state
#             for action in range(env.get_action_size()):
#               # Update the policy according to the epsilon-greedy policy
#               # If current action index is the same as that of the best action
#               if (action == bestAction):
#                 # Place more emphasize on this action, make policy have a greater value
#                 policy[state, action] = 1 - epsilon + (epsilon/env.get_action_size())
#               else:
#                 policy[state, action] = epsilon/env.get_action_size()
              
#             # Update the next state, next action
#             state = nextState
#             action = nextAction
          
#           # At the end of the episode, we want to find the value function for every state we've been in
#           newV = np.zeros(env.get_state_size())
#           newV = copy.deepcopy(V)
#           for state in statesSeen:
#             newV[state] = 0
#             # For every possible action
#             for action in range(env.get_action_size()):
#               newV[state] += policy[state, action] * Q[state, action]

#           # Copy V to a list
#           values.append(np.copy(newV))

#           # Append the undiscountedRewards to the total_rewards
#           total_rewards.append(undiscountedRewards)
      
#         # At the end of each run, we want to append the total Values, total Policies and total Rewards for the run to the running sum
#         totalRValues.append(values)
#         totalRRewards.append(total_rewards)
#         totalPolicy.append(policy)
      
#       # Convert to numpy arrays so we can use numpy functions
#       totalPolicyNP = np.asarray(totalPolicy)
#       totalRValuesNP = np.asarray(totalRValues)
#       totalRRewardsNP = np.asarray(totalRRewards)

#       # Find the average of all runs
#       averagedPolicy = np.mean(totalPolicyNP, axis=0)
#       averagedValues = np.mean(totalRValuesNP, axis=0)
#       averagedRewards = np.mean(totalRRewardsNP, axis=0)

#       # Find the std of each of the runs
#       deviationPolicy = np.std(totalPolicyNP, axis=0)
#       deviationValue = np.std(totalRValuesNP, axis=0)
#       deviationRewards = np.std(totalRRewardsNP, axis=0)

#       return averagedPolicy, averagedValues, averagedRewards, deviationPolicy, deviationValue, deviationRewards

#     def solveOFFAverage(self, env):
#       """
#       Solve a given Maze environment using Temporal Difference learning
#       input: env {Maze object} -- Maze to solve
#       output: 
#         - policy {np.array} -- Optimal policy found to solve the given Maze environment 
#         - values {list of np.array} -- List of successive value functions for each episode 
#         - total_rewards {list of float} -- Corresponding list of successive total non-discounted sum of reward for each episode 
#       """

#       # Initialisation (can be edited)
#       Q = np.random.rand(env.get_state_size(), env.get_action_size()) 
#       V = np.zeros(env.get_state_size())
#       policy = np.zeros((env.get_state_size(), env.get_action_size())) 
#       # We initialize the policy to be an epsilon-soft policy, in which all actions have a probability of epsilon/number of actions to be selected
#       # Define epsilon to be 0.1
#       epsilon = 1
#       # Define for all states and all actions, epsilon-soft gives them probability epsilon/cardinality(A) or 1 - epsilon + epsilon/cardinality(A)
#       # As an initial policy, we set that for all states, the first action will have the highest probability of being chosen
#       # policy[:, 0] = 1 - epsilon + epsilon/env.get_action_size()
#       # policy[:, 1:] = epsilon / env.get_action_size()
#       policy[:,:] = 0.25
#       values = [V]
#       total_rewards = []
#       # Define gamma 
#       gamma = env.get_gamma()
#       # Define learning rate
#       learningRate = 0.2
#       # Define set of states seen in episode
#       statesSeen = set()
#       # Define the list of totals
#       totalRValues = []
#       totalRRewards = []
#       totalPolicy = []

#       #### 
#       # Add your code here
#       # WARNING: this agent only has access to env.reset() and env.step()
#       # You should not use env.get_T(), env.get_R() or env.get_absorbing() to compute any value
#       ####

#       # In off-policy TD Control, the idea is that we initialize Q, S and set the Q for all terminal states to 0. Then, we initialize the first
#       # state. In a loop, we then choose an action A based on our state, take the action to get the reward and nextState, then update Q as
#       # Q(s|a) = Q(s|a) + LR*(reward + gamma*maxQ(s'|a) over all a - Q(s|a))
#       # Then set the state to the next state
#       # Timer for knowing when we start
#       start_time = time.time()

#       ################### PHASE 0 - EPISODE GENERATION
#       # Define the number of episodes that we want to generate
#       numberOfEpisodes = 1000
#       # Define the number of runs
#       numberOfRuns = 30

#       for run in range(numberOfRuns):
#         print('Run ', run)
#         # Reset V, values, policy and total_rewards in every run
#         V = np.zeros(env.get_state_size())
#         values = [V]
#         total_rewards = []
#         policy[:,:] = 0.25
        
#         # For number of episodes that we have
#         for episode in range(numberOfEpisodes):
#           # Define epsilon to be decaying
#           epsilon = (1 - episode/numberOfEpisodes)
#           # Define the list that will contain all the information/steps of this episode. Note that this will be in the form of a list of tuples, where
#           # each tuple will contain (timeStep, state, action, reward, done). Note that we define it within the episodes loop, as we want to reset it every ep
#           episodeSteps = []
#           # Setting the value of undiscounted reward to 0 for every episode
#           undiscountedRewards = 0

#           # Reset to get the timestep, state, starting state, action, reward, and done boolean
#           timeStep, state, reward, done = env.reset()

#           # Append state to set of states for this episode
#           statesSeen.add(state)

#           # Update the undiscounted rewards
#           undiscountedRewards = undiscountedRewards + reward

#           # Iterate until ending criteria - either reach absorbing state (done == True), or 500 steps (timeStep >= 500)
#           while (done != True):
            
#             # Find the action from the state
#             action = np.random.choice(env.get_action_size(), p=policy[state, :])
#             # Append the state,action pair to the episodeSteps
#             episodeSteps.append((timeStep, state, action, reward, done))

#             # Take a step in the direction of the action defined by the policy
#             timeStep, nextState, reward, done = env.step(action)
            
#             # Find the Q(s|a), this time with the maximum value over all actions for the next state
#             Q[state, action] = Q[state, action] + learningRate*(reward + gamma*np.max(Q[nextState,:]) - Q[state, action])
            
#             # Update the undiscounted rewards
#             undiscountedRewards = undiscountedRewards + reward

#             # Append state to set of states for this episode
#             statesSeen.add(nextState)
            
#             # Find the index of the action that gives the largest value
#             bestAction = np.argmax(Q[state,:])
#             # For all actions in the state
#             for action in range(env.get_action_size()):
#               # Update the policy according to the epsilon-greedy policy
#               # If current action index is the same as that of the best action
#               if (action == bestAction):
#                 # Place more emphasize on this action, make policy have a greater value
#                 policy[state, action] = 1 - epsilon + (epsilon/env.get_action_size())
#               else:
#                 policy[state, action] = epsilon/env.get_action_size()
              
#             # Update the next state
#             state = nextState
          
#           # At the end of the episode, we want to find the value function for every state we've been in
#           newV = np.zeros(env.get_state_size())
#           newV = copy.deepcopy(V)
#           for state in statesSeen:
#             newV[state] = 0
#             # For every possible action
#             for action in range(env.get_action_size()):
#               newV[state] += policy[state, action] * Q[state, action]

#           # Copy V to a list
#           values.append(np.copy(newV))

#           # Append the undiscountedRewards to the total_rewards
#           total_rewards.append(undiscountedRewards)
      
#         # At the end of each run, we want to append the total Values, total Policies and total Rewards for the run to the running sum
#         totalRValues.append(values)
#         totalRRewards.append(total_rewards)
#         totalPolicy.append(policy)
      
#       # Convert to numpy arrays so we can use numpy functions
#       totalPolicyNP = np.asarray(totalPolicy)
#       totalRValuesNP = np.asarray(totalRValues)
#       totalRRewardsNP = np.asarray(totalRRewards)

#       # Find the average of all runs
#       averagedPolicy = np.mean(totalPolicyNP, axis=0)
#       averagedValues = np.mean(totalRValuesNP, axis=0)
#       averagedRewards = np.mean(totalRRewardsNP, axis=0)

#       # Find the std of each of the runs
#       deviationPolicy = np.std(totalPolicyNP, axis=0)
#       deviationValue = np.std(totalRValuesNP, axis=0)
#       deviationRewards = np.std(totalRRewardsNP, axis=0)

#       return averagedPolicy, averagedValues, averagedRewards, deviationPolicy, deviationValue, deviationRewards
    
#     def solveOFFAverageDPCompare(self, env):
#       """
#       Solve a given Maze environment using Temporal Difference learning
#       input: env {Maze object} -- Maze to solve
#       output: 
#         - policy {np.array} -- Optimal policy found to solve the given Maze environment 
#         - values {list of np.array} -- List of successive value functions for each episode 
#         - total_rewards {list of float} -- Corresponding list of successive total non-discounted sum of reward for each episode 
#       """
#       maze = Maze()
#       dp_agent = DP_agent()
#       dp_policy, dp_value, dp_epoch, dp_time = dp_agent.solveValue(maze)

#       # Initialisation (can be edited)
#       Q = np.random.rand(env.get_state_size(), env.get_action_size()) 
#       V = np.zeros(env.get_state_size())
#       policy = np.zeros((env.get_state_size(), env.get_action_size())) 
#       # We initialize the policy to be an epsilon-soft policy, in which all actions have a probability of epsilon/number of actions to be selected
#       # Define epsilon to be 0.1
#       epsilon = 1
#       # Define for all states and all actions, epsilon-soft gives them probability epsilon/cardinality(A) or 1 - epsilon + epsilon/cardinality(A)
#       # As an initial policy, we set that for all states, the first action will have the highest probability of being chosen
#       # policy[:, 0] = 1 - epsilon + epsilon/env.get_action_size()
#       # policy[:, 1:] = epsilon / env.get_action_size()
#       policy[:,:] = 0.25
#       values = [V]
#       total_rewards = []
#       # Define gamma 
#       gamma = env.get_gamma()
#       # Define learning rate
#       learningRate = 0.2
#       # Define set of states seen in episode
#       statesSeen = set()
#       # Define the list of totals
#       totalRValues = []
#       totalRRewards = []
#       totalPolicy = []

#       #### 
#       # Add your code here
#       # WARNING: this agent only has access to env.reset() and env.step()
#       # You should not use env.get_T(), env.get_R() or env.get_absorbing() to compute any value
#       ####

#       # In off-policy TD Control, the idea is that we initialize Q, S and set the Q for all terminal states to 0. Then, we initialize the first
#       # state. In a loop, we then choose an action A based on our state, take the action to get the reward and nextState, then update Q as
#       # Q(s|a) = Q(s|a) + LR*(reward + gamma*maxQ(s'|a) over all a - Q(s|a))
#       # Then set the state to the next state
#       # Timer for knowing when we start
#       start_time = time.time()

#       ################### PHASE 0 - EPISODE GENERATION
#       # Define the number of episodes that we want to generate
#       numberOfEpisodes = 1000
#       # Define the number of runs
#       numberOfRuns = 30
#       # mse List
#       mc_MSE = []
#       mc_MSERun = []

#       for run in range(numberOfRuns):
#         print('Run ', run)
#         # Reset V, values, policy and total_rewards in every run
#         V = np.zeros(env.get_state_size())
#         values = [V]
#         total_rewards = []
#         policy[:,:] = 0.25
#         mc_MSE = []
        
#         # For number of episodes that we have
#         for episode in range(numberOfEpisodes):
#           # Define epsilon to be decaying
#           epsilon = (1 - episode/numberOfEpisodes)
#           # Define the list that will contain all the information/steps of this episode. Note that this will be in the form of a list of tuples, where
#           # each tuple will contain (timeStep, state, action, reward, done). Note that we define it within the episodes loop, as we want to reset it every ep
#           episodeSteps = []
#           # Setting the value of undiscounted reward to 0 for every episode
#           undiscountedRewards = 0

#           # Reset to get the timestep, state, starting state, action, reward, and done boolean
#           timeStep, state, reward, done = env.reset()

#           # Append state to set of states for this episode
#           statesSeen.add(state)

#           # Update the undiscounted rewards
#           undiscountedRewards = undiscountedRewards + reward

#           # Iterate until ending criteria - either reach absorbing state (done == True), or 500 steps (timeStep >= 500)
#           while (done != True):
            
#             # Find the action from the state
#             action = np.random.choice(env.get_action_size(), p=policy[state, :])
#             # Append the state,action pair to the episodeSteps
#             episodeSteps.append((timeStep, state, action, reward, done))

#             # Take a step in the direction of the action defined by the policy
#             timeStep, nextState, reward, done = env.step(action)
            
#             # Find the Q(s|a), this time with the maximum value over all actions for the next state
#             Q[state, action] = Q[state, action] + learningRate*(reward + gamma*np.max(Q[nextState,:]) - Q[state, action])
            
#             # Update the undiscounted rewards
#             undiscountedRewards = undiscountedRewards + reward

#             # Append state to set of states for this episode
#             statesSeen.add(nextState)
            
#             # Find the index of the action that gives the largest value
#             bestAction = np.argmax(Q[state,:])
#             # For all actions in the state
#             for action in range(env.get_action_size()):
#               # Update the policy according to the epsilon-greedy policy
#               # If current action index is the same as that of the best action
#               if (action == bestAction):
#                 # Place more emphasize on this action, make policy have a greater value
#                 policy[state, action] = 1 - epsilon + (epsilon/env.get_action_size())
#               else:
#                 policy[state, action] = epsilon/env.get_action_size()
              
#             # Update the next state
#             state = nextState
          
#           # At the end of the episode, we want to find the value function for every state we've been in
#           newV = np.zeros(env.get_state_size())
#           newV = copy.deepcopy(V)
#           for state in statesSeen:
#             newV[state] = 0
#             # For every possible action
#             for action in range(env.get_action_size()):
#               newV[state] += policy[state, action] * Q[state, action]

#           # Copy V to a list
#           values.append(np.copy(newV))

#           # Append the undiscountedRewards to the total_rewards
#           total_rewards.append(undiscountedRewards)

#           # Find the difference between the current newV and the DP
#           mc_MSE.append(mean_squared_error(dp_value, newV))
      
#         # At the end of each run, we want to append the total Values, total Policies and total Rewards for the run to the running sum
#         totalRValues.append(values)
#         totalRRewards.append(total_rewards)
#         totalPolicy.append(policy)

#         # At the end of every run, we want to add the last run's MSE to the master list
#         mc_MSERun.append(mc_MSE)
      
#       # Convert to numpy arrays so we can use numpy functions
#       totalPolicyNP = np.asarray(totalPolicy)
#       totalRValuesNP = np.asarray(totalRValues)
#       totalRRewardsNP = np.asarray(totalRRewards)

#       # Find the average of all runs
#       averagedPolicy = np.mean(totalPolicyNP, axis=0)
#       averagedValues = np.mean(totalRValuesNP, axis=0)
#       averagedRewards = np.mean(totalRRewardsNP, axis=0)

#       # Find the std of each of the runs
#       deviationPolicy = np.std(totalPolicyNP, axis=0)
#       deviationValue = np.std(totalRValuesNP, axis=0)
#       deviationRewards = np.std(totalRRewardsNP, axis=0)

#       # Average MSE
#       mc_MSERUN_NP = np.asarray(mc_MSERun)
#       averagedMSE = np.mean(mc_MSERUN_NP, axis=0)
#       deviationMSE = np.std(mc_MSERUN_NP, axis=0)

#       return averagedPolicy, averagedValues, averagedRewards, deviationPolicy, deviationValue, deviationRewards, averagedMSE, deviationMSE
    

#     def solveOFFAverageVarLR(self, env, varLR):
#       """
#       Solve a given Maze environment using Temporal Difference learning
#       input: env {Maze object} -- Maze to solve
#       output: 
#         - policy {np.array} -- Optimal policy found to solve the given Maze environment 
#         - values {list of np.array} -- List of successive value functions for each episode 
#         - total_rewards {list of float} -- Corresponding list of successive total non-discounted sum of reward for each episode 
#       """

#       # Initialisation (can be edited)
#       Q = np.random.rand(env.get_state_size(), env.get_action_size()) 
#       V = np.zeros(env.get_state_size())
#       policy = np.zeros((env.get_state_size(), env.get_action_size())) 
#       # We initialize the policy to be an epsilon-soft policy, in which all actions have a probability of epsilon/number of actions to be selected
#       # Define epsilon to be 0.1
#       epsilon = 1
#       # Define for all states and all actions, epsilon-soft gives them probability epsilon/cardinality(A) or 1 - epsilon + epsilon/cardinality(A)
#       # As an initial policy, we set that for all states, the first action will have the highest probability of being chosen
#       # policy[:, 0] = 1 - epsilon + epsilon/env.get_action_size()
#       # policy[:, 1:] = epsilon / env.get_action_size()
#       policy[:,:] = 0.25
#       values = [V]
#       total_rewards = []
#       # Define gamma 
#       gamma = env.get_gamma()
#       # Define learning rate
#       learningRate = varLR
#       # Define set of states seen in episode
#       statesSeen = set()
#       # Define the list of totals
#       totalRValues = []
#       totalRRewards = []
#       totalPolicy = []

#       #### 
#       # Add your code here
#       # WARNING: this agent only has access to env.reset() and env.step()
#       # You should not use env.get_T(), env.get_R() or env.get_absorbing() to compute any value
#       ####

#       # In off-policy TD Control, the idea is that we initialize Q, S and set the Q for all terminal states to 0. Then, we initialize the first
#       # state. In a loop, we then choose an action A based on our state, take the action to get the reward and nextState, then update Q as
#       # Q(s|a) = Q(s|a) + LR*(reward + gamma*maxQ(s'|a) over all a - Q(s|a))
#       # Then set the state to the next state
#       # Timer for knowing when we start
#       start_time = time.time()

#       ################### PHASE 0 - EPISODE GENERATION
#       # Define the number of episodes that we want to generate
#       numberOfEpisodes = 1000
#       # Define the number of runs
#       numberOfRuns = 30

#       for run in range(numberOfRuns):
#         print('Run ', run)
#         # Reset V, values, policy and total_rewards in every run
#         V = np.zeros(env.get_state_size())
#         values = [V]
#         total_rewards = []
#         policy[:,:] = 0.25
        
#         # For number of episodes that we have
#         for episode in range(numberOfEpisodes):
#           # Define epsilon to be decaying
#           epsilon = (1 - episode/numberOfEpisodes)
#           # Define the list that will contain all the information/steps of this episode. Note that this will be in the form of a list of tuples, where
#           # each tuple will contain (timeStep, state, action, reward, done). Note that we define it within the episodes loop, as we want to reset it every ep
#           episodeSteps = []
#           # Setting the value of undiscounted reward to 0 for every episode
#           undiscountedRewards = 0

#           # Reset to get the timestep, state, starting state, action, reward, and done boolean
#           timeStep, state, reward, done = env.reset()

#           # Append state to set of states for this episode
#           statesSeen.add(state)

#           # Update the undiscounted rewards
#           undiscountedRewards = undiscountedRewards + reward

#           # Iterate until ending criteria - either reach absorbing state (done == True), or 500 steps (timeStep >= 500)
#           while (done != True):
            
#             # Find the action from the state
#             action = np.random.choice(env.get_action_size(), p=policy[state, :])
#             # Append the state,action pair to the episodeSteps
#             episodeSteps.append((timeStep, state, action, reward, done))

#             # Take a step in the direction of the action defined by the policy
#             timeStep, nextState, reward, done = env.step(action)
            
#             # Find the Q(s|a), this time with the maximum value over all actions for the next state
#             Q[state, action] = Q[state, action] + learningRate*(reward + gamma*np.max(Q[nextState,:]) - Q[state, action])
            
#             # Update the undiscounted rewards
#             undiscountedRewards = undiscountedRewards + reward

#             # Append state to set of states for this episode
#             statesSeen.add(nextState)
            
#             # Find the index of the action that gives the largest value
#             bestAction = np.argmax(Q[state,:])
#             # For all actions in the state
#             for action in range(env.get_action_size()):
#               # Update the policy according to the epsilon-greedy policy
#               # If current action index is the same as that of the best action
#               if (action == bestAction):
#                 # Place more emphasize on this action, make policy have a greater value
#                 policy[state, action] = 1 - epsilon + (epsilon/env.get_action_size())
#               else:
#                 policy[state, action] = epsilon/env.get_action_size()
              
#             # Update the next state
#             state = nextState
          
#           # At the end of the episode, we want to find the value function for every state we've been in
#           for state in statesSeen:
#             V[state] = 0
#             # For every possible action
#             for action in range(env.get_action_size()):
#               V[state] += policy[state, action] * Q[state, action]

#           # Append the undiscountedRewards to the total_rewards
#           total_rewards.append(undiscountedRewards)
#           # Append the V to the values list
#           values.append(V)
      
#         # At the end of each run, we want to append the total Values, total Policies and total Rewards for the run to the running sum
#         totalRValues.append(values)
#         totalRRewards.append(total_rewards)
#         totalPolicy.append(policy)
      
#       # Convert to numpy arrays so we can use numpy functions
#       totalPolicyNP = np.asarray(totalPolicy)
#       totalRValuesNP = np.asarray(totalRValues)
#       totalRRewardsNP = np.asarray(totalRRewards)

#       # Find the average of all runs
#       averagedPolicy = np.mean(totalPolicyNP, axis=0)
#       averagedValues = np.mean(totalRValuesNP, axis=0)
#       averagedRewards = np.mean(totalRRewardsNP, axis=0)

#       # Find the std of each of the runs
#       deviationPolicy = np.std(totalPolicyNP, axis=0)
#       deviationValue = np.std(totalRValuesNP, axis=0)
#       deviationRewards = np.std(totalRRewardsNP, axis=0)

#       return averagedPolicy, averagedValues, averagedRewards, deviationPolicy, deviationValue, deviationRewards

#     def solveOFFAverageVarEP(self, env, varEP):
#       """
#       Solve a given Maze environment using Temporal Difference learning
#       input: env {Maze object} -- Maze to solve
#       output: 
#         - policy {np.array} -- Optimal policy found to solve the given Maze environment 
#         - values {list of np.array} -- List of successive value functions for each episode 
#         - total_rewards {list of float} -- Corresponding list of successive total non-discounted sum of reward for each episode 
#       """

#       # Initialisation (can be edited)
#       Q = np.random.rand(env.get_state_size(), env.get_action_size()) 
#       V = np.zeros(env.get_state_size())
#       policy = np.zeros((env.get_state_size(), env.get_action_size())) 
#       # We initialize the policy to be an epsilon-soft policy, in which all actions have a probability of epsilon/number of actions to be selected
#       # Define epsilon to be 0.1
#       epsilon = varEP
#       # Define for all states and all actions, epsilon-soft gives them probability epsilon/cardinality(A) or 1 - epsilon + epsilon/cardinality(A)
#       # As an initial policy, we set that for all states, the first action will have the highest probability of being chosen
#       # policy[:, 0] = 1 - epsilon + epsilon/env.get_action_size()
#       # policy[:, 1:] = epsilon / env.get_action_size()
#       policy[:,:] = 0.25
#       values = [V]
#       total_rewards = []
#       # Define gamma 
#       gamma = env.get_gamma()
#       # Define learning rate
#       learningRate = 0.2
#       # Define set of states seen in episode
#       statesSeen = set()
#       # Define the list of totals
#       totalRValues = []
#       totalRRewards = []
#       totalPolicy = []

#       #### 
#       # Add your code here
#       # WARNING: this agent only has access to env.reset() and env.step()
#       # You should not use env.get_T(), env.get_R() or env.get_absorbing() to compute any value
#       ####

#       # In off-policy TD Control, the idea is that we initialize Q, S and set the Q for all terminal states to 0. Then, we initialize the first
#       # state. In a loop, we then choose an action A based on our state, take the action to get the reward and nextState, then update Q as
#       # Q(s|a) = Q(s|a) + LR*(reward + gamma*maxQ(s'|a) over all a - Q(s|a))
#       # Then set the state to the next state
#       # Timer for knowing when we start
#       start_time = time.time()

#       ################### PHASE 0 - EPISODE GENERATION
#       # Define the number of episodes that we want to generate
#       numberOfEpisodes = 1000
#       # Define the number of runs
#       numberOfRuns = 30

#       for run in range(numberOfRuns):
#         print('Run ', run)
#         # Reset V, values, policy and total_rewards in every run
#         V = np.zeros(env.get_state_size())
#         values = [V]
#         total_rewards = []
#         policy[:,:] = 0.25
        
#         # For number of episodes that we have
#         for episode in range(numberOfEpisodes):
#           # Define the list that will contain all the information/steps of this episode. Note that this will be in the form of a list of tuples, where
#           # each tuple will contain (timeStep, state, action, reward, done). Note that we define it within the episodes loop, as we want to reset it every ep
#           episodeSteps = []
#           # Setting the value of undiscounted reward to 0 for every episode
#           undiscountedRewards = 0

#           # Reset to get the timestep, state, starting state, action, reward, and done boolean
#           timeStep, state, reward, done = env.reset()

#           # Append state to set of states for this episode
#           statesSeen.add(state)

#           # Update the undiscounted rewards
#           undiscountedRewards = undiscountedRewards + reward

#           # Iterate until ending criteria - either reach absorbing state (done == True), or 500 steps (timeStep >= 500)
#           while (done != True):
            
#             # Find the action from the state
#             action = np.random.choice(env.get_action_size(), p=policy[state, :])
#             # Append the state,action pair to the episodeSteps
#             episodeSteps.append((timeStep, state, action, reward, done))

#             # Take a step in the direction of the action defined by the policy
#             timeStep, nextState, reward, done = env.step(action)
            
#             # Find the Q(s|a), this time with the maximum value over all actions for the next state
#             Q[state, action] = Q[state, action] + learningRate*(reward + gamma*np.max(Q[nextState,:]) - Q[state, action])
            
#             # Update the undiscounted rewards
#             undiscountedRewards = undiscountedRewards + reward

#             # Append state to set of states for this episode
#             statesSeen.add(nextState)
            
#             # Find the index of the action that gives the largest value
#             bestAction = np.argmax(Q[state,:])
#             # For all actions in the state
#             for action in range(env.get_action_size()):
#               # Update the policy according to the epsilon-greedy policy
#               # If current action index is the same as that of the best action
#               if (action == bestAction):
#                 # Place more emphasize on this action, make policy have a greater value
#                 policy[state, action] = 1 - epsilon + (epsilon/env.get_action_size())
#               else:
#                 policy[state, action] = epsilon/env.get_action_size()
              
#             # Update the next state
#             state = nextState
          
#           # At the end of the episode, we want to find the value function for every state we've been in
#           newV = np.zeros(env.get_state_size())
#           newV = copy.deepcopy(V)
#           for state in statesSeen:
#             newV[state] = 0
#             # For every possible action
#             for action in range(env.get_action_size()):
#               newV[state] += policy[state, action] * Q[state, action]

#           # Copy V to a list
#           values.append(np.copy(newV))

#           # Append the undiscountedRewards to the total_rewards
#           total_rewards.append(undiscountedRewards)
      
#         # At the end of each run, we want to append the total Values, total Policies and total Rewards for the run to the running sum
#         totalRValues.append(values)
#         totalRRewards.append(total_rewards)
#         totalPolicy.append(policy)
      
#       # Convert to numpy arrays so we can use numpy functions
#       totalPolicyNP = np.asarray(totalPolicy)
#       totalRValuesNP = np.asarray(totalRValues)
#       totalRRewardsNP = np.asarray(totalRRewards)

#       # Find the average of all runs
#       averagedPolicy = np.mean(totalPolicyNP, axis=0)
#       averagedValues = np.mean(totalRValuesNP, axis=0)
#       averagedRewards = np.mean(totalRRewardsNP, axis=0)

#       # Find the std of each of the runs
#       deviationPolicy = np.std(totalPolicyNP, axis=0)
#       deviationValue = np.std(totalRValuesNP, axis=0)
#       deviationRewards = np.std(totalRRewardsNP, axis=0)

#       return averagedPolicy, averagedValues, averagedRewards, deviationPolicy, deviationValue, deviationRewards
  
#   ############ COMPARISON
#   print("Creating the Maze:\n")
#   maze = Maze()

#   td_agent = TD_agent()
#   dp_agent = DP_agent()

#   td_policyON, td_valuesON, total_rewardsON, td_timeON = td_agent.solveON(maze)
#   td_policyOFF, td_valuesOFF, total_rewardsOFF, td_timeOFF = td_agent.solveOFF(maze)

#   dp_policy, dp_V, dp_epoch, dp_timeTaken = dp_agent.solveValue(maze)

#   differenceTDOn = dp_V - td_valuesON[-1]
#   differenceTDOff = dp_V - td_valuesOFF[-1]

#   titles = ["TD On-Policy V(s), in {}".format(td_timeON), "TD Off-Policy V(s), in {}".format(td_timeOFF), 
#               "TD On-Policy vs DP Value V(s), {:.1f} secs".format(td_timeON - dp_timeTaken),
#               "TD Off-Policy vs DP Value V(s), {:.1f} secs".format(td_timeOFF - dp_timeTaken), 
#               "TD On-Policy Policy", "TD Off-Policy Policy", "DP Value Itr. Policy"]

#   compareValues = []
#   comparePolicies = []

#   compareValues.append(td_valuesON[-1])
#   compareValues.append(td_valuesOFF[-1])
#   compareValues.append(differenceTDOn)
#   compareValues.append(differenceTDOff)

#   comparePolicies.append(td_policyON)
#   comparePolicies.append(td_policyOFF)
#   comparePolicies.append(dp_policy)

#   print("Results of TD both On- and Off-Policy VS DP Value Itr.:\n")
#   maze.get_graphics().draw_value_grid(compareValues, titles[:4], 1, 4)
#   maze.get_graphics().draw_policy_grid(comparePolicies[:2], titles[4:6], 1, 2)
#   maze.get_graphics().draw_policy(comparePolicies[2], titles[6:])

#   ################## ON VS OFF
#   print("Creating the Maze:\n")
#   maze = Maze()

#   td_agent = TD_agent()

#   td_policyOFF, td_valuesOFF, total_rewardsOFF, td_timeOFF = td_agent.solveOFF(maze)
#   td_policyON, td_valuesON, total_rewardsON, td_timeON = td_agent.solveON(maze)

#   titles = ["TD Off-Policy V(s), in {}".format(td_timeOFF), 
#               "TD Off-Policy Policy", "TD Off-Policy Undiscounted Rewards"]

#   print("Results of Off-Policy :\n")
#   maze.get_graphics().draw_value(td_valuesOFF[-1], titles[0])
#   maze.get_graphics().draw_policy(td_policyOFF, titles[1])

#   plt.plot(total_rewardsOFF)
#   plt.ylabel('Rewards')
#   plt.xlabel('Episode')
#   plt.title('Total Undiscounted Rewards - 1000 Episodes')
#   plt.legend()
#   plt.show()

#   ################### AVERAGE OFF
#   maze = Maze()
#   td_agent = TD_agent()

#   averagedPolicy, averagedValues, averagedRewards, deviationPolicy, deviationValue, deviationRewards = td_agent.solveOFFAverage(maze)

#   print("Results of the TD agent:\n")
#   maze.get_graphics().draw_policy(averagedPolicy, "Policy for Averaged Q-Learning")
#   maze.get_graphics().draw_value(averagedValues[-1], "Value Function for Averaged Q-Learning")

#   # Since we know we have 6000 episodes
#   episodes_range = np.arange(1000)
#   plt.plot(episodes_range, averagedRewards, label="Mean of Rewards")
#   plt.fill_between(episodes_range, averagedRewards - deviationRewards, averagedRewards + deviationRewards, label="STD of Rewards", color="lightsteelblue")
#   plt.ylabel('Rewards')
#   plt.xlabel('Number of Runs')
#   plt.title('Mean and STD of Rewards from Averaged Q-Learning')
#   plt.legend()
#   plt.show()

#   ################### Impact of EP and LR
#   ### Impact of epsilon on the td algorithm
#   EP_range = [0.2, 0.4, 0.6, 0.8]
#   avgPolicies = []
#   avgValues = []
#   avgRew = []
#   avgRewDeviation = []
#   titles = []

#   maze = Maze()
#   td_agent = TD_agent()

#   # Use value iteration for each Ep value
#   for Ep in EP_range:
#       averagedPolicy, averagedValues, averagedRewards, deviationPolicy, deviationValue, deviationRewards = td_agent.solveOFFAverageVarEP(maze, varEP = Ep)
#       avgPolicies.append(averagedPolicy)
#       avgValues.append(averagedValues[-1])
#       avgRew.append(averagedRewards)
#       avgRewDeviation.append(deviationRewards)
#       print('At Epsilon ', Ep)

#   # Show the learning curve for every one
#   episodes_range = np.arange(1000)

#   for i in range(len(avgRew)):
#       plt.plot(episodes_range, avgRew[i], label="Mean of Rewards for Epsilon = {}".format(EP_range[i]))
#       plt.fill_between(episodes_range, avgRew[i] - avgRewDeviation[i], avgRew[i] + avgRewDeviation[i], label="STD of Rewards", color="lightsteelblue")
#       plt.ylabel('Rewards')
#       plt.xlabel('Number of Runs')
#       plt.title('Mean and STD of Rewards from Averaged Q-Learning for Epsilon = {}'.format(EP_range[i]))
#       plt.legend()
#       plt.show()
  
#   ### Impact of epsilon on the MC algorithm
#   LR_range = [0.01, 0.1, 0.4, 0.6]
#   avgPolicies = []
#   avgValues = []
#   avgRew = []
#   avgRewDeviation = []
#   titles = []

#   maze = Maze()
#   td_agent = TD_agent()

#   # Use value iteration for each LR value
#   for LR in LR_range:
#       averagedPolicy, averagedValues, averagedRewards, deviationPolicy, deviationValue, deviationRewards = td_agent.solveOFFAverageVarLR(maze, varLR = LR)
#       avgPolicies.append(averagedPolicy)
#       avgValues.append(averagedValues[-1])
#       avgRew.append(averagedRewards)
#       avgRewDeviation.append(deviationRewards)
#       print('At LR ', LR)

#   # Show the learning curve for every one
#   episodes_range = np.arange(1000)

#   for i in range(len(avgRew)):
#       plt.plot(episodes_range, avgRew[i], label="Mean of Rewards for LR = {}".format(LR_range[i]))
#       plt.fill_between(episodes_range, avgRew[i] - avgRewDeviation[i], avgRew[i] + avgRewDeviation[i], label="STD of Rewards", color="lightsteelblue")
#       plt.ylabel('Rewards')
#       plt.xlabel('Number of Runs')
#       plt.title('Mean and STD of Rewards from Averaged Q-Learning for LR = {}'.format(LR_range[i]))
#       plt.legend()
#       plt.show()
  
#   ##################### FIND AVERAGE MSE
#   ########## COMPARISON OF LEARNERS
#   # Creating maze
#   maze = Maze()
#   # Create DP, MC and TD learners
#   dp_agent = DP_agent()
#   mc_agent = MC_agent()
#   td_agent = TD_agent()
#   # Initializing the MSE arrays
#   mc_MSE = []
#   td_MSE = []
#   mc_STD = []
#   td_STD = []

#   dp_policy, dp_value, dp_epoch, dp_time = dp_agent.solveValue(maze)
#   mc_averagedPolicy, mc_averagedValues, mc_averagedRewards, mc_deviationPolicy, mc_deviationValue, mc_deviationRewards, mc_averageMSE, mc_averageSTD = mc_agent.solveAveragedDPCompare(maze)
#   td_averagedPolicy, td_averagedValues, td_averagedRewards, td_deviationPolicy, td_deviationValue, td_deviationRewards, td_averageMSE, td_averageSTD = td_agent.solveOFFAverageDPCompare(maze)

#   episodes_range = np.arange(1000)
#   print("MC lowest MSE Mean", mc_averageMSE[-1])
#   print("TD lowest MSE Mean", td_averageMSE[-1])

#   plt.plot(episodes_range, mc_averageMSE, label="Mean of MSE")
#   plt.fill_between(episodes_range, mc_averageMSE - mc_averageSTD, mc_averageMSE + mc_averageSTD, label="STD of MSE", color="lightsteelblue")
#   plt.ylabel('Mean Squared Error')
#   plt.xlabel('Episode')
#   plt.title('Average MSE of MC and DP Baseline')
#   plt.legend()
#   plt.show()

#   plt.plot(episodes_range, td_averageMSE, label="Mean of MSE")
#   plt.fill_between(episodes_range, td_averageMSE - td_averageSTD, td_averageMSE + td_averageSTD, label="STD of MSE", color="lightsteelblue")
#   plt.ylabel('Mean Squared Error')
#   plt.xlabel('Episode')
#   plt.title('Average MSE of TD and DP Baseline')
#   plt.legend()
#   plt.show()

#   ##################### SCATTER PLOT
#   # Scatter plot of estimation eror vs undiscounted sum of reward for MC and TD
#   # Creating maze
#   maze = Maze()
#   # Create DP, MC and TD learners
#   dp_agent = DP_agent()
#   mc_agent = MC_agent()
#   td_agent = TD_agent()
#   # Initializing the MSE arrays
#   mc_MSE = []
#   td_MSE = []

#   dp_policy, dp_value, dp_epoch, dp_time = dp_agent.solveValue(maze)
#   mc_policy, mc_values, mc_totalRewards = mc_agent.solveON(maze)
#   td_policyOFF, td_valuesOFF, td_totalRewards, td_timeOFF = td_agent.solveOFF(maze)

#   td_totalRewardsNP = np.asarray(td_totalRewards)
#   print('totalRewards shape', td_totalRewardsNP.shape)

#   dp_valuesNP = np.asarray(dp_value)

#   for i in range(len(mc_values) - 1):
#       mc_MSE.append(mean_squared_error(dp_valuesNP, mc_values[i+1]))
      
#   for i in range(len(td_valuesOFF) - 1):
#       td_MSE.append(mean_squared_error(dp_valuesNP, td_valuesOFF[i+1]))

#   # Defining for the plots
#   plt.figure()
#   plt.scatter(mc_MSE, mc_totalRewards, color="orange", label="MSE vs Reward for MC")
#   plt.scatter(td_MSE, td_totalRewards, label="MSE vs Reward for TD")
#   plt.ylabel('Rewards')
#   plt.xlabel('Mean Squared Error')
#   plt.title('Total Undiscounted Rewards vs TD and MC MSE for 1000 episode')
#   plt.legend()
#   plt.show()

  # ##################################################### EXAMPLE MAIN
  # # Example main (can be edited)

  # ### Question 0: Defining the environment

  # print("Creating the Maze:\n")
  # maze = Maze()


  # ### Question 1: Dynamic programming

  # dp_agent = DP_agent()
  # dp_policy, dp_value = dp_agent.solve(maze)

  # print("Results of the DP agent:\n")
  # maze.get_graphics().draw_policy(dp_policy)
  # maze.get_graphics().draw_value(dp_value)


  # ### Question 2: Monte-Carlo learning

  # mc_agent = MC_agent()
  # mc_policy, mc_values, total_rewards = mc_agent.solve(maze)

  # print("Results of the MC agent:\n")
  # maze.get_graphics().draw_policy(mc_policy)
  # maze.get_graphics().draw_value(mc_values[-1])


  # ### Question 3: Temporal-Difference learning

  # td_agent = TD_agent()
  # td_policy, td_values, total_rewards = td_agent.solve(maze)

  # print("Results of the TD agent:\n")
  # maze.get_graphics().draw_policy(td_policy)
  # maze.get_graphics().draw_value(td_values[-1])