##### Connect 4 Game ####
import numpy as np
from random import choice
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from torchvision import transforms
import torch
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Main Class
class connect4:
    def __init__(self):
        self.csize = 7
        self.rsize = 6
        self.board = np.full((self.rsize, self.csize), " ")
        # self.marker_selection_skip = False

    # Display board as plot
    def display(self):
        # close previous figure
        plt.close(fig="all")
        # Create grid for plot
        grid_x, grid_y = np.meshgrid(np.arange(1, self.csize + 1, 1), np.arange(1, self.rsize + 1, 1)[::-1])

        # Create plot
        fig, ax = plt.subplots() # Instantiate subplot classe
        ax.scatter(grid_x[self.board == "R"], grid_y[self.board == "R"], c="red", s=700) # Get indices for red
        ax.scatter(grid_x[self.board == "B"], grid_y[self.board == "B"], c="black", s=700) # Get indices for black
        ax.set_xlim(0.5, 7.5) # Change x and y limits to always show the whole board
        ax.set_ylim(0.5, 6.5)
        ax.set_title("Current Board") # Set title
        ax.grid(which="minor") # Display in between column grid
        ax.xaxis.set_minor_locator(AutoMinorLocator(2)) # change tick size for inbetween column grid
        plt.show() # show
    # Reset board
    def reset(self):
        self.__init__()

    # Return current board as torch image
    def export_board(self):
        # Create grid for plot
        grid_x, grid_y = np.meshgrid(np.arange(1, self.csize + 1, 1), np.arange(1, self.rsize + 1, 1)[::-1])

        fig, ax = plt.subplots()  # Instantiate subplot classe
        ax.scatter(grid_x[self.board == "R"], grid_y[self.board == "R"], c="red", s=700)  # Get indices for red
        ax.scatter(grid_x[self.board == "B"], grid_y[self.board == "B"], c="black", s=700)  # Get indices for black
        ax.set_xlim(0.5, 7.5)  # Change x and y limits to always show the whole board
        ax.set_ylim(0.5, 6.5)
        ax.set_title("Current Board")  # Set title
        ax.grid(which="minor")  # Display in between column grid
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))  # change tick size for inbetween column grid
        fig.canvas.draw()
        image_data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        image_data = image_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # Close figure
        plt.close(fig="all")
        # Transform numpy array to torch image
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((round(480*0.3), round(640*0.3))),
            transforms.ToTensor()
        ]
        )
        # Transform image and add batch dimension (BCHW)
        image_data_transformed = transform(image_data).unsqueeze(0)
        return image_data_transformed

    def actions_performable(self):
        return torch.tensor(np.flatnonzero(np.any(self.board == " ", axis = 0)))



## Functions for playability

# Check if line accessable
def ifAvailable(input, current_board):
    # Assert if column in current board is not full
    return sum(current_board[:, input-1] == " ") == 0

# Make random column selection
def random_select(current_board):
    cols = np.arange(1,8)
    # Find columns with empty spaces left
    cols_to_select = cols[np.any(current_board == " ", axis = 0)]

    # Select column at random
    random_selection = choice(cols_to_select)
    return random_selection

# Prompt player to input column
def player_input(player_name, current_board):
    i = 0
    while True:
        i = i + 1
        if i >4:
            rand_choice = random_select(current_board)
            print("Too many wrong inputs. Random selection of column {}." .format(rand_choice))
            return rand_choice

        # Check input is integer and between 1 and 7
        try:
            input_player = int(input(player_name + " select a column: "))
            assert 1 <= input_player <= 7
        except:
            print("Input should be an integer between 1 and 7, please try again.")
            continue
        if ifAvailable(input_player, current_board):
            print("Selected column is full. Please, select another one.")
            continue
        return input_player

# Function for selecting the marker each player uses
def marker_selection():
    markers = ["R", "B"]
    text_markers = {"R" : "Red", "B" : "Black"}
    i = 0
    while True:
        i = i +1
        if i >5:
            player_1_marker = choice(markers)
            print("Too many wrong inputs, random selection applied. Player 1 will be: {}".format(text_markers[player_1_marker]))
            return player_1_marker
        try:
            player_1_marker = input("Player 1 choose your marker (either red (R) or black (B)):").capitalize()
            assert player_1_marker in markers
        except:
            print("Please, type either the latter R for red or B for black (not case sensitive).")
            continue
        print("Player 1 has choosen: {}" .format(text_markers[player_1_marker]))
        if player_1_marker == markers[0]:
            player_2_marker = markers[1]
        else:
            player_2_marker = markers[0]
        print("Thus player 2 will be: {}".format(text_markers[player_2_marker]))
        return [player_1_marker,player_2_marker]

# function for placing the marker in the selected column

def place_marker(marker_type, player_input, current_board):
    # Retrieve row index of placement, i.e. the highest index of the empty entries
    row_select = np.flatnonzero((current_board[:, player_input-1] == " "))[-1]

    # Alter board and return
    current_board[row_select, player_input-1] = marker_type
    return current_board

def second_player_select():
    i = 0
    while True:
        i = i +1
        if i>5:
            print("Too many wrong inputs. AI set as second player.")
            return True
        try:
            selected = input("Do you want to play against the AI? (Y/N): ").capitalize()
            assert selected in ["Y", "N"]
        except:
            print("Wrong input, please type only Y for Yes and N for No.")
            continue
        return (selected == "Y")

# Function that asks for rematch

def rematch():
    i = 0
    while True:
        i = i + 1
        if i > 5:
            print("Too many wrong inputs. Rematch cancelled.")
            return False
        try:
            selected = input("Do you want a rematch? (Y/N): ").capitalize()
            assert selected in ["Y", "N"]
        except:
            print("Wrong input, please type only Y for Yes and N for No.")
            continue
        return (selected == "Y")

## Functions for winning conditions

# Helper function for counting consecutive occurences
def helper_count(cond_array):
    return np.diff(np.where(np.concatenate(([cond_array[0]],
                                            cond_array[:-1] != cond_array[1:],
                                            [True])))[0])[::2]

# Vertical/horizontal check
def win_check_vert(player_marker, current_board):

    ## Vertical check
    # Return columns with at least four markers of one type
    cols = np.flatnonzero(np.sum(current_board == player_marker, axis = 0) >= 4)

    # Return winning condition false if not enough markers are placed in any of the columns
    if len(cols) == 0:
        return False
    else:
        # Convert candidate columns into bool arrays, indicating player's markers placement
        cond_arrays = (current_board[:, cols] == player_marker).transpose()

        # Apply helper count function to get lengths of consecutive appearing of True's, if >=4 return true
        for column in cond_arrays:
            if any(helper_count(column) >= 4):
                return True
        return False

def win_check_hor(player_marker, current_board):
    ## Horizontal check
    # Return rows with at least four markers of one type
    rows = np.flatnonzero(np.sum(current_board == player_marker, axis=1) >= 4)

    # Return winning condition false if not enough markers are placed in any of the rows
    if len(rows) == 0:
        return False
    else:
        # Convert candidate rows into bool arrays, indicating player's markers placement
        cond_arrays = (current_board[rows, :] == player_marker)

        # Apply helper count function to get lengths of consecutive appearing of True's, if >=4 return true
        for r in cond_arrays:
            if any(helper_count(r) >= 4):
                return True
        return False

def win_check_diag(player_marker, current_board):
    # Retrieve all relevant diagonals of the current board, as well as the vertically flipped board

    diags = [np.diagonal(current_board, offset = x) for x in range(-2,4)]
    diags_flipped = [np.diagonal(np.fliplr(current_board), offset = x) for x in range(-2,4)]

    # Retrieve diagonals with more than four player markers
    diag_candidates = np.array(diags, dtype = "object")[np.flatnonzero(np.array([sum(x == player_marker) for x in diags]) >= 4)]
    diag_flipped_candidates = np.array(diags_flipped,
                                       dtype = "object")[np.flatnonzero(np.array([sum(x == player_marker) for x in diags_flipped]) >= 4)]

    # Check if there are even diagonals which contain at least 4 of the player's markers
    if len(diag_candidates) == 0 and len(diag_flipped_candidates) == 0:
        return False
    # Check now every diagonal candidate, retrieve entries which contain the player's marker,
    # calculate differences of entry indices and return true, if there are 3 ones,
    # meaning 4 adjacent player marks in the respective diagonal
    else:
        for candidate in diag_candidates:
            if any(helper_count(candidate == player_marker) >= 4):
                return True
        for candidate in diag_flipped_candidates:
            if any(helper_count(candidate == player_marker) >= 4):
                return True
        return False

# Function that checks if the game ended in a draw
def draw_check(current_board):
    return ~(np.any(current_board == " "))

# Function to select AI
def ai_select():
    i = 0
    while True:
        i = i + 1
        if i > 5:
            print("Too many wrong inputs. Selected Monkey AI.")
            return True
        try:
            selected = input("Do you want to play against Monkey (type M), \n"
                             "RL-AI (type R) or perfect AI (type P)?: ").capitalize()
            assert selected in ["M", "R", "P"]
        except:
            print("Wrong input, please type only M, R or P.")
            continue
        return selected

# Manual Game function
def connect_4_manual(player_1_marker, player_2_marker, connect, rematch_dec=True, game=True):
    # If one player wants a rematch, this while loop will continue to run, if not, the while loop terminates
    while rematch_dec:
        # Reset board
        connect.reset()

        # Game loop
        while game:
            if draw_check(connect.board):
                print("The game ended in a draw.")
                rematch_dec = rematch()
                break

            # Display current board
            connect.display()

            # Turn Player 1
            input_1 = player_input("Player 1", connect.board)
            connect.board = place_marker(player_1_marker, input_1, connect.board)

            # Check if player has won
            win_1 = any([win_check_vert(player_1_marker, connect.board),
                         win_check_hor(player_1_marker, connect.board),
                         win_check_diag(player_1_marker, connect.board)])

            if win_1:
                print("Player 1 has won!")
                connect.display()
                rematch_dec = rematch()
                break
            connect.display()

            # Turn Player 2
            input_2 = player_input("Player 2", connect.board)
            connect.board = place_marker(player_2_marker, input_2, connect.board)

            # Check if player has won
            win_2 = any([win_check_vert(player_2_marker, connect.board),
                         win_check_hor(player_2_marker, connect.board),
                         win_check_diag(player_2_marker, connect.board)])

            if win_2:
                print("Player 2 has won!")
                connect.display()
                rematch_dec = rematch()
                break
            connect.display()
        if rematch_dec == False:
            print("See ya later aligator!")

# AI game function
def connect_4_ai(algorithm, player_1_marker, player_2_marker, connect, rematch_dec = True, game = True):
    # If one player wants a rematch, this while loop will continue to run, if not, the while loop terminates
    last_screen = connect.export_board()
    current_screen = connect.export_board()
    while rematch_dec:
        # Reset board
        connect.reset()
        # Game loop
        while game:
            if draw_check(connect.board):
                print("The game ended in a draw.")
                rematch_dec = rematch()
                break

            # Display current board
            connect.display()

            # Turn Player 1
            input_1 = player_input("Player 1", connect.board)
            connect.board = place_marker(player_1_marker, input_1, connect.board)

            # Check if player has won
            win_1 = any([win_check_vert(player_1_marker, connect.board),
                         win_check_hor(player_1_marker, connect.board),
                         win_check_diag(player_1_marker, connect.board)])

            if win_1:
                print("Player 1 has won!")
                connect.display()
                rematch_dec = rematch()
                break
            connect.display()

            # Create current state (only used in the RL AI
            last_screen = current_screen
            current_screen = connect.export_board()
            global state
            state = current_screen - last_screen

            # Turn Player 2
            connect, win_2 = algorithm(connect, player_2_marker)
            if win_2:
                print("Player 2 has won!")
                connect.display()
                rematch_dec = rematch()
                break
        if rematch_dec == False:
            print("See ya later aligator!")

## Add Algorithms for AI

# Monkey player

def monkey_player(connect, marker_type):
    action = random.choice(np.flatnonzero(np.any(connect.board == " ", axis=0)))

    return connect_4_train(action, connect, marker_type)

# Reinforcement learning AI
def rl_player(connect, marker_type):
    # Get action based on state
    avail = False
    k = 0
    actions_proposed = rl_algo(state).topk(k=6)[1][0]  # Get actions proposed by network order by largest value
    available_columns = torch.tensor(np.flatnonzero(np.any(connect.board == " ", 0)))  # Get available columns

    # Cycle through every actions proposed by the network and check if column is available
    while avail == False:
        action_proposed = actions_proposed[k].item()
        avail = any(action_proposed == available_columns)
        k += 1
    return connect_4_train(action_proposed, connect, marker_type) # Place Marker


# Perfect AI

def perfect_player(connect, marker_type):
    return

# Main Game function
def connect_4_game():
    print("Welcome to the beautiful game of connect 4.")
    game = True

    # Prompt for single- or multiplayer
    ai_player = second_player_select()

    # If play against AI: Select AI
    if ai_player:
        ai_selected = ai_select()

    # Player_1 select marker type
    player_1_marker, player_2_marker = marker_selection()

    # Instantiate connect 4 class
    connect = connect4()

    if not ai_player:
        return connect_4_manual(player_1_marker, player_2_marker, connect)
    else:
        if ai_selected == "M":
            connect_4_ai(monkey_player, player_1_marker, player_2_marker, connect)
        elif ai_selected == "R":
            ## Load AI and set to eval mode
            # Get image size (will be constant for all iterations)
            _, _, screen_height, screen_width = connect.export_board().shape
            # get numbers of actions
            n_actions = torch.tensor(connect.actions_performable().shape).item()
            # Instatiate network
            global rl_algo
            rl_algo = q_learning_model.DQN(screen_height, screen_width, n_actions).to(device)
            rl_algo.load_state_dict(torch.load("C:/Users/TonyG/Documents/GitHub/connect4/q_learning_mod"))
            rl_algo.eval() # Set to evaluation mode
            connect_4_ai(rl_player, player_1_marker, player_2_marker, connect)
        else:
            connect_4_ai(perfect_player, player_1_marker, player_2_marker, connect)
    return

# Create a version to be used for training the DQN

# Helper function for checking board win
def helper_win(current_board, marker_type):
    vert = win_check_vert(marker_type, current_board)
    hor = win_check_hor(marker_type, current_board)
    diag = win_check_diag(marker_type, current_board)

    return any([vert, hor, diag])

# Helper function for placing marker
def helper_place_marker(player_input, current_board, marker_type):
    # Retrieve row index of placement, i.e. the highest index of the empty entries
    row_select = np.flatnonzero((current_board[:, player_input] == " "))[-1]

    # Alter board and return
    current_board[row_select, player_input] = marker_type
    return current_board

# Function for dropping a marker by the AI
def connect_4_train(action, connect, marker_type):
    # Place marker
    connect.board = helper_place_marker(action, connect.board, marker_type)

    # Check for win
    win = helper_win(connect.board, marker_type)

    # Return winning condition and connect class
    return [connect, win]

if __name__ == "__main__":
    # Start Game
    import q_learning_model
    connect_4_game()





