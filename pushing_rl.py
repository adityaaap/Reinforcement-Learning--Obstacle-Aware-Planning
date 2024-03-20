import numpy as np
from tqdm import tqdm

from gym.spaces import Box
# PPO Implementation reference:
# https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py

# DQN Implementation reference:
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html


from panda_pushing_env import TARGET_POSE, OBSTACLE_CENTRE, OBSTACLE_HALFDIMS, BOX_SIZE

## MY CODE
dist = []
##

class RandomPolicy(object):
    """
    A random policy for any environment.
    It has the same method as a stable-baselines3 policy object for compatibility.
    """

    def __init__(self, env):
        self.env = env

    def predict(self, state):
        action = self.env.action_space.sample()  # random sample the env action space
        return action, None


def execute_policy(env, policy, num_steps=5):
    """
    Given a policy and an environment, execute the policy for num_steps steps.
    The policy is a stable-baselines3 policy object.
    :param env:
    :param policy:
    :param num_steps:
    :return:
    """
    states = []
    rewards = []
    goal_reached = False
    # --- Your code here
    obs_init = env.reset()
    states.append(obs_init)
    obs = obs_init
    for i in range(num_steps):
      # states.append(obs)
      action, _ =  policy.predict(obs)

      ## print actions
      print(action)
      obs, reward, done, _ = env.step(action)

      states.append(obs)
      rewards.append(reward)
      #print(obs)
      if(np.linalg.norm(obs[:2] - TARGET_POSE[:2]) < 0.1):
        goal_reached = True
        break



    # ---
    return states, rewards, goal_reached




def obstacle_free_pushing_reward_function_object_pose_space(state, action):
    """
    Defines the state reward function for the action transition (prev_state, action, state)
    :param state: numpy array of shape (state_dim)
    :param action:numpy array of shape (action_dim)
    :return: reward value. <float>
    """
    reward = 0
    # --- Your code here
    # pos = state[:2]
    # posNext = pos + action[:2]
    # distToGoal = np.linalg.norm(posNext - TARGET_POSE[:2])
    # distToGoalPrev = np.linalg.norm(pos - TARGET_POSE[:2])
    # thresh = 0.3
    # delta = 1
    # if distToGoal < distToGoalPrev:
    #   ## REWARD 1 ##
    #   # reward += 10*delta 
    #   # delta += 0.1

    #   ## REWARD 2 ##
    #   reward = 10

    # elif distToGoal > distToGoalPrev:
    #   ## REWARD 1 ##
    #   # reward += -10*delta

    #   ## REWARD 2 ##
    #   reward = -10

    # ## REWARD 2 ##
    # if(distToGoal < 0.11):
    #   reward = 500

    
    # else:
    #   distance_change = np.linalg.norm(pos - TARGET_POSE[:2]) - distToGoal
    #   reward = distance_change
    #   if(distance_change < 0):
    #     reward -= 10

    ##########################################
    pos = state[:2]
    distToGoal = np.linalg.norm(pos - TARGET_POSE[:2])
    flag = return_distance(distToGoal, pos) # function written at end

    if(flag == True):
      reward = 10
    else:
      reward = -10

    if(distToGoal < 0.15):
      reward = 500



    # ---
    return reward


def pushing_with_obstacles_reward_function_object_pose_space(state, action):
    """
    Defines the state reward function for the action transition (prev_state, action, state)
    :param state: numpy array of shape (state_dim)
    :param action:numpy array of shape (action_dim)
    :return: reward value. <float>
    """
    reward = None
    # --- Your code here
    # SIMPLE REWARD FUNCTION - Based on if states collides or not
    # Rest of reward function will be same as non-obs case - based on distance

    pos = state[:2]
    distToGoal = np.linalg.norm(pos - TARGET_POSE[:2])
    flag = return_distance(distToGoal, pos) # function written at end

    flag2 = isColliding(state)

    if(flag2 == False):
      reward = -500
      return reward

    if(flag == True):
      reward = 10
    else:
      reward = -10

    if(distToGoal < 0.15):
      reward = 500



    # ---
    return reward

# Ancillary functions
# --- Your code here

def return_distance(curr_distance, pos):
  if(len(dist) == 0):
    flag = False
    if(pos[1] - TARGET_POSE[1] < 0.01): # Want the y coord of first action to the aligned with goals, so starting point is good
      flag = True
    dist.append(curr_distance)
    return flag

  dist_prev = dist[-1]
  if(curr_distance < dist_prev):
    dist.append(curr_distance)
    return True

  dist.append(curr_distance)
  return False

  
def isColliding(state):
  x = state[0]
  y = state[1]

  corner1 = [x+0.5, y + 0.5]
  corner2 = [x - 0.5, y + 0.5]
  corner3 = [x - 0.5, y - 0.5]
  corner4 = [x + 0.5, y - 0.5]

  flag1 = True
  flag2 = True
  # if(corner1[0] - 0.5 < 0 or corner2[0] - 0.5 < 0 or corner3[0] - 0.5 < 0 or corner4[0] - 0.5 < 0):
  #   flag1 = False

  if((0.5 < corner1[0] < 0.6) or (0.5 < corner2[0] < 0.6) or (0.5 < corner3[0] < 0.6) or (0.5 < corner4[0] < 0.6)):
    flag1 = False

  if(corner1[1] > 0.5 or corner2[1] > 0.5 or corner3[1] > 0.5 or corner4[1] > 0.5):
    flag2 = False


  if(flag1 == False and flag2 == False):
    return False
  
  if(flag1 == True or flag2 == True):
    return True

  


# ---