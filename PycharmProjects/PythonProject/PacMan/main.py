import pygame, random
import numpy as np

# Initialize Pygame
pygame.init()
pygame.mixer.init()

# Set up some constants
GAMESPEED = 4
GHOSTS_DELAY_TRACKING = (2, 5, 10, 15)
WIDTH, HEIGHT = 1000, 400
ROWS, COLS = 10, 25
SQUARE_SIZE = WIDTH // COLS
closed = 0

INITIAL_MATRIX = np.array([
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1],
    [0, 2, 2, 0, 2, 2, 0, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2],
    [0, 2, 4, 0, 4, 2, 0, 2, 2, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1],
    [0, 2, 0, 0, 0, 2, 0, 1, 1, 1, 1, 2, 2, 2, 2, 1, 2, 1, 2, 2, 2, 2, 1, 2, 2],
    [0, 2, 4, 0, 4, 2, 0, 2, 2, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 1],
    [0, 2, 2, 2, 2, 2, 0, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 2, 1],
    [0, 0, 0, 2, 0, 0, 0, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1],
    [1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
    [1, 2, 1, 1, 1, 2, 1, 2, 2, 2, 1, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1],
    [1, 2, 2, 2, 2, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1]
])

# Hyperparameters
TRAIN = True
TRAINING_GAMES = 1000
LIMIT_STEPS = 300
LEARNING_RATE = 0.01  # how fast it learns
DISCOUNT_FACTOR = 0.95  # how much it cares about immediate rewards
PACDOT_REWARD = 2
MOVE_PENALTY = -0.1
WIN_REWARD = 100
LOSE_PENALTY = -40
EPSILON_START = 1  # exploration rate at the beggining (high randomness)
EPSILON_END = 0.01  # exploration rate at the end (low randomness = exploitation)
EPSILON_DECAY = 0.995  # how fast it goes from high to low randomness

# Set up the display
WIN = pygame.display.set_mode((WIDTH, HEIGHT))

# Set up the title and icon
pygame.display.set_caption("Pac-man AI")
icon = pygame.image.load("asset/pacman.png")
pygame.display.set_icon(icon)
pac_man_image_open = pygame.image.load("asset/pac_man_abierto.png")
pac_man_image_open = pygame.transform.scale(pac_man_image_open, (SQUARE_SIZE, SQUARE_SIZE))
pac_man_image_closed = pygame.image.load("asset/pac_man_cerrado.png")
pac_man_image_closed = pygame.transform.scale(pac_man_image_closed, (SQUARE_SIZE, SQUARE_SIZE))
ghost_image1 = pygame.image.load("asset/biene.png")
ghost_image2 = pygame.image.load("asset/biene2.png")
ghost_image3 = pygame.image.load("asset/biene3.png")
ghost_image4 = pygame.image.load("asset/biene4.png")

ghost_image1 = pygame.transform.scale(ghost_image1, (SQUARE_SIZE, SQUARE_SIZE))
ghost_image2 = pygame.transform.scale(ghost_image2, (SQUARE_SIZE, SQUARE_SIZE))
ghost_image3 = pygame.transform.scale(ghost_image3, (SQUARE_SIZE, SQUARE_SIZE))
ghost_image4 = pygame.transform.scale(ghost_image4, (SQUARE_SIZE, SQUARE_SIZE))

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 200, 0)
RED = (200, 0, 0)
YELLOW = (255, 255, 0)


def pos_in_grid(pos):
    # Check if pos is in grid
    return pos[0] >= 0 and pos[0] < ROWS and pos[1] >= 0 and pos[1] < COLS


def get_relative_position(agent_pos, dot_pos):
    return (dot_pos[0] - agent_pos[0], dot_pos[1] - agent_pos[1])


def state_to_pos(state):
    x, y = state
    return (y * SQUARE_SIZE, x * SQUARE_SIZE)


def empty_tiles(matrix):
    # Return a list of empty tiles
    empty_tiles = []
    for i in range(ROWS):
        for j in range(COLS):
            if matrix[i][j] != 2:
                empty_tiles.append((i, j))
    return empty_tiles


def get_next_state(state, action):
    if action == 0:  # Up
        return (state[0] - 1, state[1])
    elif action == 1:  # Down
        return (state[0] + 1, state[1])
    elif action == 2:  # Left
        return (state[0], state[1] - 1)
    elif action == 3:  # Right
        return (state[0], state[1] + 1)


# Grid class
class Grid:
    def __init__(self):
        self.reset()

    def draw_dots_walls(self):
        # Draw pac-dots and walls
        for i in range(ROWS):
            for j in range(COLS):
                if self.matrix[i][j] == 1:  # meaning we will draw a dot
                    # it has to pass through each cell, so every row and column will be multiplied by the square size and add it to the middle
                    x = j * SQUARE_SIZE + SQUARE_SIZE // 2  # X-coordinate of the dot's center
                    y = i * SQUARE_SIZE + SQUARE_SIZE // 2  # Y-coordinate of the dot's center
                    radius = 3  # Radius of the dot
                    pygame.draw.circle(WIN, YELLOW, (x, y), radius)
                elif self.matrix[i][j] == 2:  # meaning we will draw a wall
                    x = j * SQUARE_SIZE + SQUARE_SIZE // 2  # X-coordinate of the top-left corner of the rhomboid
                    y = i * SQUARE_SIZE + SQUARE_SIZE // 2  # Y-coordinate of the top-left corner of the rhomboid
                    half_width = SQUARE_SIZE // 2  # Half the width of the rhombus
                    points = [
                        (x, y - half_width),  # Top point
                        (x + half_width, y),  # Right point
                        (x, y + half_width),  # Bottom point
                        (x - half_width, y)  # Left point
                    ]
                    pygame.draw.polygon(WIN, GREEN, points)

    def draw_pac_man(self, state, action):
        # Draw Pac-man
        # 295, 115, 15, -15
        x, y = state_to_pos(state)
        if action == 0:  # UP
            if closed == 1:
                rotated_image = pygame.transform.rotate(pac_man_image_closed, 90)
                WIN.blit(rotated_image, (x, y))
            else:
                rotated_image = pygame.transform.rotate(pac_man_image_open, 115)
                WIN.blit(rotated_image, (x, y))
        elif action == 1:  # DOWN
            if closed == 1:
                rotated_image = pygame.transform.rotate(pac_man_image_closed, 270)
                WIN.blit(rotated_image, (x, y))
            else:
                rotated_image = pygame.transform.rotate(pac_man_image_open, 295)
                WIN.blit(rotated_image, (x, y))
        elif action == 2:  # LEFT
            if closed == 1:
                flipped_image = pygame.transform.flip(pac_man_image_closed, True, False)
                WIN.blit(flipped_image, (x, y))
            else:
                flipped_image = pygame.transform.flip(pac_man_image_open, True, False)
                rotated_image = pygame.transform.rotate(flipped_image, -15)
                WIN.blit(rotated_image, (x, y))
        elif action == 3:  # RIGHT
            if closed == 1:
                WIN.blit(pac_man_image_closed, (x, y))
            else:
                rotated_image = pygame.transform.rotate(pac_man_image_open, 15)
                WIN.blit(rotated_image, (x, y))

    def draw_ghosts(self, ghosts):
        # Draw ghosts
        ghosts_images = [ghost_image1, ghost_image2, ghost_image3, ghost_image4]
        for i, ghost in enumerate(ghosts):
            x, y = state_to_pos(ghost.state)
            WIN.blit(ghosts_images[i], (x, y))

    def draw_score(self, score):
        # draw the score board
        font = pygame.font.Font(None, 36)
        text = "SCORE: " + str(score)
        text_surface = font.render(text, True, (255, 255, 255))
        # Blit the text surface onto the game window (top-left corner)
        WIN.blit(text_surface, (0, 0))

    def reset(self):
        # Reset the game
        self.matrix = INITIAL_MATRIX.copy()
        # leave the matric without pac-man and ghosts
        self.matrix *= (self.matrix != 3) & (self.matrix != 4)

    def start_state(self):
        # Return the start state
        possible_tiles = empty_tiles(self.matrix)
        return possible_tiles[random.randint(0, len(possible_tiles) - 1)]  # (7, 16)

    def step(self, state, action, closed, is_agent=True):
        # Calculate next state based on action
        # If action leads to a wall, next_state = state
        next_state = get_next_state(state, action)
        closed = not closed
        if not pos_in_grid(next_state):
            print(f"{state} -> {next_state} -> {action}")

        # Check if game is done
        n_pac_dots = np.sum(self.matrix == 1)
        if n_pac_dots == 0:
            # all pac-dots eaten
            done = True
            reward = WIN_REWARD
            code = 0
        elif self.matrix[next_state] == 4:
            # Pac-Man caught by ghost
            done = True
            # reward = LOSE_PENALTY
            code = 1
        else:
            done = False
            if self.matrix[next_state] == 1:
                # collision with pac-dot
                if is_agent:
                    self.matrix[next_state] = 0
                reward = PACDOT_REWARD
                code = 2
            else:
                # move to empty space
                reward = MOVE_PENALTY
                code = 3

        return next_state, reward, done, code, closed


# Agent class (Pac-man)
class Agent:
    def __init__(self, num_actions, alpha=LEARNING_RATE, gamma=DISCOUNT_FACTOR, epsilon_start=EPSILON_START,
                 epsilon_end=EPSILON_END, epsilon_decay=EPSILON_DECAY):
        self.num_actions = num_actions
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon_start  # exploration rate
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.reset()

        # Initialize the Q-table to zeros
        self.Q_table = np.zeros((ROWS, COLS, num_actions))

    def choose_action(self, matrix):
        # Return the action with the highest Q-value for the current state
        # This represents the policy
        if np.random.rand() < self.epsilon:
            # Choose a random action
            possible_actions = np.array([])
            for i in range(self.num_actions):
                next_state = get_next_state(self.state, i)
                if pos_in_grid(next_state) and matrix[next_state] != 2:
                    possible_actions = np.append(possible_actions, 1)
                else:
                    possible_actions = np.append(possible_actions, 0)
            return np.random.choice(self.num_actions, p=possible_actions / np.count_nonzero(possible_actions))
        else:
            # Choose the best known action that is valid
            possible_actions = np.array([])
            for i in range(self.num_actions):
                next_state = get_next_state(self.state, i)
                if pos_in_grid(next_state) and matrix[next_state] != 2:
                    possible_actions = np.append(possible_actions, self.Q_table[self.state][i])
                else:
                    possible_actions = np.append(possible_actions, -np.inf)
            return np.argmax(possible_actions)

    def update_Q_table(self, action, reward, next_state):
        # Calculate the current Q-value
        current_q_value = self.Q_table[self.state][action]

        # Calculate the maximum Q-value for the next state
        next_max_q_value = np.max(self.Q_table[next_state])

        # Update the Q-value for the current state-action pair
        self.Q_table[self.state][action] = (1 - self.alpha) * current_q_value + self.alpha * (
                    reward + self.gamma * next_max_q_value)

        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

    def reset(self):
        # Reset the agent state
        self.state = (START_X, START_Y)


class Ghost:
    def __init__(self, id_, state, goal):
        self.id = id_
        self.initial_state = state
        self.goal = goal
        self.reset()

    def choose_action(self, goal, grid):
        self.visited = np.zeros((ROWS, COLS))
        if self.counter > 0:
            self.path = self.dfs(self.state, goal, grid)
            self.counter -= 1
        if self.path == None or len(self.path) == 0:
            possible_actions = np.array([])
            for i in range(4):
                next_state = get_next_state(self.state, i)
                if pos_in_grid(next_state) and self.grid[next_state] != 2:
                    possible_actions = np.append(possible_actions, 1)
                else:
                    possible_actions = np.append(possible_actions, 0)
            return np.random.choice(4, p=possible_actions / np.count_nonzero(possible_actions))

        return self.path.pop(0)

    def dfs(self, position, goal, grid):
        if position == goal:
            return []

        for direction in range(4):
            new_pos = get_next_state(position, direction)

            if pos_in_grid(new_pos) and grid[new_pos] != 2 and not self.visited[new_pos]:
                self.visited[new_pos] = 1
                path = self.dfs(new_pos, goal, grid)
                if path is not None:
                    return [direction] + path

                self.visited[new_pos] = 0
        return None

    def update(self, matrix, goal):
        self.grid = matrix
        self.goal = goal

    def reset(self):
        self.state = self.initial_state
        self.counter = GHOSTS_DELAY_TRACKING[self.id]
        self.update(INITIAL_MATRIX, self.goal)


# Initialize the clock
clock = pygame.time.Clock()

# Initialize the agent
START_X, START_Y = np.where(INITIAL_MATRIX == 3)
START_X, START_Y = START_X[0], START_Y[0]
agent = Agent(num_actions=4)
code_meaning = {0: "WIN", 1: "LOSE", 2: "IN PROGRESS, EATING PAC-DOT", 3: "IN PROGRESS, MOVING"}

# Initialize the ghosts
positions = np.where(INITIAL_MATRIX == 4)
positions = list(zip(positions[0], positions[1]))
ghosts = [Ghost(i, positions[i], agent.state) for i in range(4)]
saved_vals = [0 for _ in range(4)]

# Initialize the grid
grid = Grid()

if TRAIN:
    # Training loop
    print("training agent...")
    for episode in range(TRAINING_GAMES):  # number of games
        grid.reset()
        agent.state = grid.start_state()  # Start state from the grid
        done = False
        steps = 0
        actions = np.array([0, 0, 0, 0])
        while not done:
            action = agent.choose_action(grid.matrix)
            actions[action] += 1

            # Assume the game environment provides the next_state and reward.
            next_state, reward, done, code, closed = grid.step(agent.state, action, closed)
            if steps > LIMIT_STEPS:
                done = True
                reward = LOSE_PENALTY
            agent.update_Q_table(action, reward, next_state)
            agent.state = next_state
            if done:
                print(f"{episode}: code, {code_meaning[code]}, {steps} steps, {actions}")
            steps += 1
    print("agent trained")
    agent.reset()
    grid.reset()

    # Save the Q-table
    np.save("asset/learned_q_table.npy", agent.Q_table)
else:
    # Load the Q-table
    agent.Q_table = np.load("asset/learned_q_table.npy")


# Main loop
def update_display():
    WIN.fill(BLACK)
    grid.draw_dots_walls()
    grid.draw_score(score)
    grid.draw_pac_man(agent.state, action)
    grid.draw_ghosts(ghosts)
    pygame.display.update()


pygame.mixer.music.load("asset/pacman_beginning.wav")

action = 3
score = 0
code = 1
run = True
done = True
music = True
playing = True
update_display()
pygame.mixer.music.play()
while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    if pygame.mixer.music.get_busy():
        if music:
            continue
    else:
        playing = False

    music = False
    # Game logic
    # move the agent
    if not done:
        # Choose an action
        action = agent.choose_action(grid.matrix)
        # Execute the action and get the new state
        new_state, reward, done, code, closed = grid.step(agent.state, action, closed)
        # print(f"agent state: {agent.state} -> {action} -> {new_state}")
        # if steps > LIMIT_STEPS:
        # 	done = True
        # 	reward = LOSE_PENALTY
        # agent.update_Q_table(action, reward, next_state)

        # Update the current state
        agent.state = new_state
        if code == 2:
            score += 10
    # steps += 1
    if done:
        # Reset the grid and state when game is over
        print(f"game over ({code_meaning[code]}), resetting...")
        grid.reset()
        agent.reset()  # grid.start_state()
        action = 3
        steps = 0
        for ghost in ghosts:
            ghost.reset()
        done = False
        score = 0

    # move the ghosts
    for i, ghost in enumerate(ghosts):
        ghost.update(grid.matrix, agent.state)
        # Choose an action
        g_action = ghost.choose_action(agent.state, grid.matrix)
        if g_action == -1:
            continue
        # Execute the action and get the new state
        new_state, _, _, _, closed = grid.step(ghost.state, g_action, closed, is_agent=False)
        # print(f"ghost state: {ghost.state} -> {g_action} -> {new_state}")

        # Update the current state
        grid.matrix[ghost.state] = saved_vals[i]
        saved_vals[i] = grid.matrix[new_state]
        ghost.state = new_state
        # grid.matrix[ghost.state] = 4
        if ghost.state == agent.state:
            print(f"game over, ghost {i} got you!")
            done = True
            break

    # Update the display
    update_display()
    pygame.mixer.music.load("asset/pacman_chomp.wav")
    if not playing:
        pygame.mixer.music.play()
        playing = True
    clock.tick(GAMESPEED)

pygame.quit()