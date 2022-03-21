import math
import random

import numpy as np
from collections import namedtuple, deque
from itertools import count
import game
import os
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

# Uncomment if you wanna see the AI playing
# is_ipython = 'inline' in matplotlib.get_backend()
# if is_ipython:
#     from IPython import display
#
# plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# release all unoccupied GPU cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Number of GPUs to use (only applied if GPU count > 1)
gpus_per_trial = 2

# Intantiate Board
connect = game.connect4()

## Define Replay Memory
# Define a named tuple that represents a single transition on the board
Transition = namedtuple("Transition",
                        ("state", "action", "next_state", "reward"))


# Define a cyclic buffer of bounded size that holds the transitions observed recently. Will later also be used
# for training

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)  # list-like sequence

    # Method for saving a transition
    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Function that returns a batch of randomly sampled previously saved transition of size batch_size"""
        return random.sample(self.memory, batch_size)

    # Len method
    def __len__(self):
        return len(self.memory)


# Specify if the model should be saved
save = True


### Description
# The aim will be to approximate the optimal Q-value function denoted with Q'(s,a), where s denotes the
# state s (i.e. the current board) and a the taken action at time step t (i.e. the columns selected to drop
# a chip). If we know Q'(s,a) we could easily infer the optimal policy P'(s) = argmax_a Q'(s,a). To approximate
# Q'(s,a) we will make use of the Bellman equation: Q^P(s,a) = r + g Q^P(s', P(s')), where g is a discount factor.
# We do the following: Create a CNN that takes as input the current board and outputs Q(s, c), with c = {1,...,7}
# being the columns selectable. The network thus outputs the expected return of taking each action given the current
# board. We can then select the action based on the argmax of the output.
# To approximate the optimal Q'(s,a) we make use of the temporal difference error (temporal in the sense that we start
# in s and go to s'): error = Q(s,a) - (r + g max_a Q(s',P(s'))), where we obtain Q(s,a) from our network (now called
# policy network) and max_a Q(s', a) from one instance of our network (from now on called target network) which is based
# on the policy network but has its weights froozen during training and gets updated only so often.
# We use the Huber loss to obtain error and optimize our policy network based on this.
# To train our DQN (Deep Q Network) we use an epsilon greedy policy. The probability of choosing a random action will
# decay exponentially fast.
# Since connect4 is a 2 player game we will mainly train player 1 and then update the weights of the network of
# player 2 after each episode (i.e. a game). This saves memory and speeds up training.

## Create the Q-Network
# (h,w): image height and width, outputs: numbers of actions
class DQN(nn.Module):
    def __init__(self, h, w, outputs, l1=200, l2=100):
        super(DQN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2),  # First Convolutional Layer
            nn.BatchNorm2d(16),  # Normalise layer output,
            nn.ReLU(),  # RElu activation function
            nn.Conv2d(16, 32, kernel_size=5, stride=2),  # Second Convolutional Layer
            nn.BatchNorm2d(32),  # Normalise layer output
            nn.ReLU(),  # RElu activation function
            nn.Conv2d(32, 32, kernel_size=5, stride=2),  # Third Convolutional Layer
            nn.BatchNorm2d(32),  # Normalise layer output
            nn.ReLU()  # RElu activation function
        )

        # Calculate conv_layer output size to be able to pass it through the final linear layer
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        # Calculate width and height of conv_layer output based on input w and h
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        self.linear_input_size = convw * convh * 32

        # Final linear layers
        self.linear_layers = nn.Sequential(
            nn.Linear(self.linear_input_size, l1),
            nn.ReLU(),
            nn.Linear(l1, l2),
            nn.ReLU(),
            nn.Linear(l2, outputs)
        )

    def forward(self, x):
        x = x.to(device)  # Convert to GPU or CPU tensor
        x = self.conv_layer(x)  # Pass through conv net

        # To be able to pass it through the final layer the conv output gets flattened
        return self.linear_layers(x.view(x.size(0), -1))


### Training

## Hyperparameters and utilities

# Define the epsilon decay and when the target net will be updated

BATCH_SIZE = 80
GAMMA = 0.99  # Discount factor
EPS_START = 0.9  # starting value epsilon
EPS_END = 0.05  # final value epsilon
EPS_DECAY = 300  # decay factor of epsilon
TARGET_UPDATE = 10
steps_done = 0  # Helper for epsilon decay

# Reward dictionary
rewards = {"win": 50, "draw": 5, "lose": -50, "surviving": 0}

# Get image size (will be constant for all iterations)
_, _, screen_height, screen_width = connect.export_board().shape

# get numbers of actions
n_actions = torch.tensor(connect.actions_performable().shape).item()

# Instatiate networks
policy_net = DQN(screen_height, screen_width, n_actions).to(device)
policy_net_player_2 = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)

# Multi GPU
if torch.cuda.device_count() > 1:
    policy_net = nn.DataParallel(policy_net)
    policy_net_player_2 = nn.DataParallel(policy_net_player_2)
    target_net = nn.DataParallel(target_net)

# Give target_net the same parameters as policy_net and set to eval mode (no weight updates), do the same with
# second player
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
policy_net_player_2.load_state_dict(policy_net.state_dict())
policy_net_player_2.eval()

# Create optimizer here we use SGD
optimizer = optim.RMSprop(policy_net.parameters(), lr=0.01, momentum=0.90)

# Create an scheduler which reduced the learning rate when the loss is not decreasing over a long period
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.05, patience=TARGET_UPDATE)

# Instatiate Replay memory (only for player 1)
memory = ReplayMemory(10000)

# Define empty lists for player 1 wins, draws, reward and loss
loss_player_2 = 0
wins_player_2 = 0
draws_player_2 = 0
reward_player_2 = 0


# Define the action selection function based on an epsilon greedy approach
def select_action(state, conn, model):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            # However, it might happen, that the algorithm proposed a column, that is already full.
            # Thus we have to check if the column selected is already full and if so, select the one with the next highest
            # expected reward
            avail = False
            k = 0
            actions_proposed = model(state).topk(k=7)[1][0]  # Get actions proposed by network order by largest value
            available_columns = torch.tensor(np.flatnonzero(np.any(conn.board == " ", 0)))  # Get available columns

            # Cycle through every actions proposed by the network and check if column is available
            while avail == False:
                action_proposed = actions_proposed[k].item()
                avail = any(action_proposed == available_columns)
                k += 1
            action_selected = model(state).topk(k=7)[1][0][k - 1].view(1, 1)
            return action_selected
    else:
        # Same for random selection
        return torch.tensor([[random.choice(np.flatnonzero(np.any(connect.board == " ", 0)))]],
                            device=device, dtype=torch.long)


## Training Loop

## Optimization function
# We now define a function that optimizes the model given a batch of transitions, if there are enough transitions
# stored in the memory. It will only use states which are not final states, i.e. states after which the game ended, for
# final states we set the value function equal to 0.

def optimize_model():
    global loss_player_2
    if len(memory) < BATCH_SIZE:
        return
    # Get a Batch of transitions
    transitions = memory.sample(BATCH_SIZE)

    # Transpose the batch. This converts batch-array of Transitions to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                  device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken. These are the actions
    # which would've been taken for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states. Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0]. This is merged based on the mask, such that
    # we'll have either the expected state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Save loss
    loss_player_2 += loss.item()

    # Optimize model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


## Reward calculation
# We will now write a function that plays one round, i.e. places a marker based on the action and then retrieves the
# reward. We will award +20 for winning and +10 for a draw and 0 for a loss or a simple placement.

def reward_getter(action, connect, marker_type):
    connect, win = game.connect_4_train(action, connect, marker_type)

    draw = game.draw_check(connect.board)

    done = any([draw, win])

    if win:
        reward = rewards["win"]
        return reward, connect, done
    if draw:
        reward = rewards["draw"]
        return reward, connect, done
    else:
        reward = rewards["surviving"]
        return reward, connect, done


# Main loop
num_episodes = 1000
perf_update = 50  # When to receive performance updates during training
episode_durations = []  # count number of moves per game

if __name__ == "__main__":
    for i_episode in range(num_episodes):
        # Initialize environment and state
        connect.reset()
        last_screen = connect.export_board()
        current_screen = connect.export_board()
        state = current_screen - last_screen
        for t in count():
            # Player 1
            # action_player_1 = select_action(state, connect, policy_net_player_2) # Player 1 AI performs action
            # reward_1, connect, done = reward_getter(action_player_1, connect, "B") # Player 1 AI gets reward
            # reward_1 = torch.tensor([reward_1], device=device) # convert reward to tensor

            # Monkey Player 1
            action_player_1 = random.choice(np.flatnonzero(np.any(connect.board == " ", axis=0)))
            reward_1, connect, done = reward_getter(action_player_1, connect, "B")  # Player 1 AI gets reward
            reward_1 = torch.tensor([reward_1], device=device)  # convert reward to tensor

            # Player_2 observes new state
            last_screen = current_screen
            current_screen = connect.export_board()

            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            # Move to next state
            state = next_state

            # If player 1 has already won, then we don't have to have player 2 select an action. Since in this case
            # player 2 wouldn't get any strict positive reward we don't lose any experience not performing an action
            # selection for player 2
            if done:
                episode_durations.append(t + 1)
                if reward_1.item() == rewards["draw"]:  # Just for safety shouldn't occur for the first player
                    draws_player_2 += 1  # Increase draws player 2 by one
                elif reward_1.item() == rewards["win"]:  # If player 1 wins, we punish player 2
                    reward_player_2 += rewards["lose"]
                break

            # If player_1 already ended the game this will not get executed
            # Player 2
            action_player_2 = select_action(state, connect, policy_net)  # Player 2 AI performs action
            reward_2, connect, done = reward_getter(action_player_2, connect, "R")  # Player 2 AI gets reward
            reward_2 = torch.tensor([reward_2], device=device)  # convert reward to tensor

            # Player_1 observes new state
            last_screen = current_screen
            current_screen = connect.export_board()

            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            # Store the transition of player 2 in memory
            memory.push(state, action_player_2, next_state, reward_2)
            # memory_player_2.push(state, action_player_2, next_state, reward_2)

            # Move to next state
            state = next_state

            # Perform one step of the optimization (player 2) and retrieve loss
            optimize_model()

            if done:
                # if this gets executed the result is either, that player 1 has lost or that the game ended in a
                # draw
                episode_durations.append(t + 1)
                reward_player_2 += reward_2.item()  # Append player 2 reward
                if reward_2.item() == rewards["win"]:
                    wins_player_2 += 1  # Increase wins player 2 by one
                elif reward_2.item() == rewards["draw"]:
                    draws_player_2 += 1  # Increase draws player 2 by one
                break
            # Note: Since the state space is of order 2^42 a draw would only occur after player 2 has performed
            # an action since at round 1 player 1 performs an action and in round 2 player 2, hence the 42nd round
            # will definitely be performed by player 2. But this would also result in done = True in the
            # reward_getter function.

        # Update learning rate if necessary
        scheduler.step(loss_player_2 / (sum(episode_durations) - (BATCH_SIZE - 1)))

        # Update the target network, copying all weights and biases in DQN after 10 episodes
        if (i_episode + 1) % (TARGET_UPDATE) == 0:
            target_net.load_state_dict(policy_net.state_dict())
            # Update player 2 ai
            # policy_net_player_2.load_state_dict(policy_net.state_dict())

        # Calculate average loss, win-, draw-ratio and average reward per episode for the last 20 episodes
        if (i_episode + 1) % (perf_update) == 0:
            if (sum(episode_durations) - (BATCH_SIZE - 1)) == 0:
                div_loss = 1
            else:
                div_loss = sum(episode_durations) - (BATCH_SIZE - 1)

            print("Rounds played: {}; Mean loss of last {} moves: {};\n"
                  "Mean reward: {}; Mean wins of player 2 : {} \n"
                  "Mean draws of player 2: {}; Learning rate: {}"
                  .format(i_episode + 1, sum(episode_durations),
                          round(loss_player_2 / div_loss, 4),
                          round(reward_player_2 / (i_episode + 1), 4),
                          round(wins_player_2 / (i_episode + 1), 4),
                          round(draws_player_2 / (i_episode + 1), 4),
                          optimizer.state_dict()["param_groups"][0]["lr"]))
    print("Complete")

    # Save Model if specified
    if save:
        torch.save(policy_net.state_dict(), "C:/Users/TonyG/Documents/GitHub/connect4/q_learning_mod")
