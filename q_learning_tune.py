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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# release all unoccupied GPU cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Number of GPUs to use
gpus_per_trial = 1

# Intantiate Board
connect = game.connect4()

# Get image size (will be constant for all iterations)
_, _, screen_height, screen_width = connect.export_board().shape

# get numbers of actions
n_actions = torch.tensor(connect.actions_performable().shape).item()

# get current path
path = os.getcwd()
checkpoint_dir = path

## Define Replay Memory
# Define a named tuple that represents a single transition on the board
Transition = namedtuple("Transition",
                        ("state", "action", "next_state", "reward"))

# Define a cyclic buffer of bounded size that holds the transitions observed recently. Will later also be used
# for training

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory =deque([], maxlen = capacity) # list-like sequence

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

## Create the Q-Network
# (h,w): image height and width, outputs: numbers of actions
class DQN(nn.Module):
    def __init__(self, h, w, outputs, l1=200, l2=100):
        super(DQN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2), # First Convolutional Layer
            nn.BatchNorm2d(16), # Normalise layer output,
            nn.ReLU(), # RElu activation function
            nn.Conv2d(16,32, kernel_size=5, stride=2), # Second Convolutional Layer
            nn.BatchNorm2d(32), # Normalise layer output
            nn.ReLU(),  # RElu activation function
            nn.Conv2d(32, 32, kernel_size=5, stride=2), # Third Convolutional Layer
            nn.BatchNorm2d(32),  # Normalise layer output
            nn.ReLU() # RElu activation function
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
        x = x.to(device) # Convert to GPU or CPU tensor
        x = self.conv_layer(x) # Pass through conv net

        # To be able to pass it through the final layer the conv output gets flattened
        return self.linear_layers(x.view(x.size(0), -1))

# Define the action selection function based on an epsilon greedy approach
def select_action(state, conn, model, EPS_END, EPS_START, EPS_DECAY):
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
            actions_proposed =  model(state).topk(k=7)[1][0] # Get actions proposed by network order by largest value
            available_columns = torch.tensor(np.flatnonzero(np.any(conn.board == " ", 0))) # Get available columns

            # Cycle through every actions proposed by the network and check if column is available
            while avail == False:
                action_proposed = actions_proposed[k].item()
                avail = any(action_proposed == available_columns)
                k += 1
            action_selected = model(state).topk(k=7)[1][0][k-1].view(1,1)
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

def optimize_model(GAMMA, BATCH_SIZE):
    global loss_player_1, memory, policy_net, target_net, optimizer
    if len(memory) < BATCH_SIZE:
        return
    # Get a Batch of transitions
    transitions = memory.sample(BATCH_SIZE)

    # Transpose the batch. This converts batch-array of Transitions to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                  device=device, dtype = torch.bool)
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
    loss_player_1 += loss.item()

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
        reward = 50
        return reward, connect, done
    if draw:
        reward = 5
        return reward, connect, done
    else:
        reward = 0
        return reward, connect, done

episode_durations = []
def train_q_learn(config, checkpoint_dir=None):
    global connect
    ## Hyperparameters and utilities

    # Define the epsilon decay and when the target net will be updated
    BATCH_SIZE = int(config["batch_size"])
    GAMMA = config["gamma"]
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200
    TARGET_UPDATE = 10

    # Get image size (will be constant for all iterations)
    _, _, screen_height, screen_width = connect.export_board().shape

    # get numbers of actions
    n_actions = torch.tensor(connect.actions_performable().shape).item()

    # Instatiate networks
    policy_net = DQN(screen_height, screen_width, n_actions, config["l1"], config["l2"]).to(device)
    policy_net_player_2 = DQN(screen_height, screen_width, n_actions, config["l1"], config["l2"]).to(device)
    target_net = DQN(screen_height, screen_width, n_actions).to(device)

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

    # Create optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=config["lr"])

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
        policy_net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    # Instatiate Replay memory (only for player 1)
    memory = ReplayMemory(10000)
    steps_done = 0

    # Initialize Variables to measure performance per games
    loss_player_2 = 0
    wins_player_2 = 0
    draws_player_2 = 0
    reward_player_2 = 0
    reward_to_end = {50: "win", 5: "draw"}

    for i_episode in range(50):
        # Initialize environment and state
        connect.reset()
        last_screen = connect.export_board()
        current_screen = connect.export_board()
        state = current_screen - last_screen

        # Start Game
        for t in count():
            # Player 1
            action_player_1 = select_action(state, connect, policy_net_player_2)  # Player 1 AI performs action
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
                if reward_to_end[reward_1.item()] == "draw":  # Just for safety shouldn't occur for the first player
                    draws_player_2 += 1  # Increase draws player 2 by one
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
                if reward_to_end[reward_2.item()] == "win":
                    wins_player_2 += 1  # Increase wins player 2 by one
                elif reward_to_end[reward_2.item()] == "draw":
                    draws_player_2 += 1  # Increase draws player 2 by one
                break
            # Note: Since the state space is of order 2^42 a draw would only occur after player 2 has performed
            # an action since at round 1 player 1 performs an action and in round 2 player 2, hence the 42nd round
            # will definitely be performed by player 2. But this would also result in done = True in the
            # reward_getter function.
        with tune.checkpoint_dir(i_episode) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((policy_net.state_dict(), optimizer.state_dict()), path)

        # Calculate average loss, win-, draw-ratio and average reward per episode for the last 20 episodes
        tune.report(loss=(loss_player_2 / (sum(episode_durations) - (BATCH_SIZE - 1))), wins=wins_player_2)

        # Update the target network, copying all weights and biases in DQN after 10 episodes
        if (i_episode + 1) % (TARGET_UPDATE) == 0:
            target_net.load_state_dict(policy_net.state_dict())
            # Update player 2 ai
            policy_net_player_2.load_state_dict(policy_net.state_dict())
    print("Complete")



def main(num_samples=10, max_num_rounds=25):

    # Define Search space
    config = {
        "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "lr": tune.loguniform(1e-4, 1e-1),
        "gamma": tune.loguniform(0.95, 0.999),
        "batch_size": tune.choice([2, 4, 8, 16])
    }

    # Set Scheduler (terminate early if candidates are not promising)
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_rounds,
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(
        metric_columns=["loss", "wins", "training_iteration"])

    result = tune.run(
        train_q_learn,
        resources_per_trial={"cpu": 6, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final number of wins: {}".format(
        best_trial.last_result["wins"]))


    best_trained_model = DQN(screen_height, screen_width, n_actions, best_trial.config["l1"],
                             best_trial.config["l2"]).to(device)
    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

if __name__ == "__main__":
    main(num_samples=5, max_num_rounds=25)

