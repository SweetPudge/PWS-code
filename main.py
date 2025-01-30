import numpy as np
import pygame
import sys
import random
import threading

# Kleuren
BLUE = (0,0,255)
BLACK = (0,0,0)
RED = (255,0,0)
YELLOW = (255,255,0)
WHITE = (255, 255, 255)


ROW_COUNT = 6
COLUMN_COUNT = 7

def create_board():
    return np.zeros((ROW_COUNT, COLUMN_COUNT))

def drop_piece(board, row, col, piece):
    board[row][col] = piece

def is_valid_location(board, col):
    return board[ROW_COUNT-1][col] == 0

def get_next_open_row(board, col):
    for r in range(ROW_COUNT):
        if board[r][col] == 0:
            return r

def print_board(board):
    print(np.flip(board, 0))

def winning_move(board, piece):
    for c in range(COLUMN_COUNT-3):
        for r in range(ROW_COUNT):
            if np.all(board[r, c:c+4] == piece):
                return True
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT-3):
            if np.all(board[r:r+4, c] == piece):
                return True
    for c in range(COLUMN_COUNT-3):
        for r in range(ROW_COUNT-3):
            if np.all([board[r+i][c+i] == piece for i in range(4)]):
                return True
    for c in range(COLUMN_COUNT-3):
        for r in range(3, ROW_COUNT):
            if np.all([board[r-i][c+i] == piece for i in range(4)]):
                return True
    return False

def is_board_full(board):
    return all(not is_valid_location(board, col) for col in range(COLUMN_COUNT))

def count_sequences(board, piece, length):
    count = 0
    for c in range(COLUMN_COUNT - (length - 1)):
        for r in range(ROW_COUNT):
            if np.sum(board[r, c:c+length] == piece) == length and np.sum(board[r, c:c+length] == 0) == 4 - length:
                count += 1
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - (length - 1)):
            if np.sum(board[r:r+length, c] == piece) == length and np.sum(board[r:r+length, c] == 0) == 4 - length:
                count += 1
    return count

def is_about_to_win(board, piece):
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT):
           
            if np.sum(board[r, c:c+4] == piece) == 3 and np.sum(board[r, c:c+4] == 0) == 1:
                empty_col = c + np.where(board[r, c:c+4] == 0)[0][0]
                if r == 0 or board[r-1][empty_col] != 0:  
                    return True
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - 3):
       
            if np.sum(board[r:r+4, c] == piece) == 3 and np.sum(board[r:r+4, c] == 0) == 1:
                empty_row = r + np.where(board[r:r+4, c] == 0)[0][0]
                if empty_row == 0 or board[empty_row-1][c] != 0:  
                    return True
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT - 3):
           
            if np.sum([board[r+i][c+i] == piece for i in range(4)]) == 3 and np.sum([board[r+i][c+i] == 0 for i in range(4)]) == 1:
                empty_index = np.where([board[r+i][c+i] == 0 for i in range(4)])[0][0]
                empty_row = r + empty_index
                empty_col = c + empty_index
                if empty_row == 0 or board[empty_row-1][empty_col] != 0:  
                    return True
    for c in range(COLUMN_COUNT - 3):
        for r in range(3, ROW_COUNT):
           
            if np.sum([board[r-i][c+i] == piece for i in range(4)]) == 3 and np.sum([board[r-i][c+i] == 0 for i in range(4)]) == 1:
                empty_index = np.where([board[r-i][c+i] == 0 for i in range(4)])[0][0]
                empty_row = r - empty_index
                empty_col = c + empty_index
                if empty_row == 0 or board[empty_row-1][empty_col] != 0:  
                    return True
    return False

def get_reward(old_board, new_board, piece):
    old_2_seq = count_sequences(old_board, piece, 2)
    old_3_seq = count_sequences(old_board, piece, 3)
   
    new_2_seq = count_sequences(new_board, piece, 2)
    new_3_seq = count_sequences(new_board, piece, 3)
   
   
    delta_2_seq = new_2_seq - old_2_seq
    delta_3_seq = new_3_seq - old_3_seq
   
    reward = delta_2_seq + delta_3_seq * 4  

   
    opponent_piece = 1 if piece == 2 else 2
    if is_about_to_win(old_board, opponent_piece) and not is_about_to_win(new_board, opponent_piece):
        reward += 10  

    return reward

def draw_board(board):
    screen.fill(BLACK)
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            pygame.draw.rect(screen, BLUE, (c*SQUARESIZE, r*SQUARESIZE+SQUARESIZE, SQUARESIZE, SQUARESIZE))
            pygame.draw.circle(screen, BLACK, (int(c*SQUARESIZE+SQUARESIZE/2), int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS)
   
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            if board[r][c] == 1:
                pygame.draw.circle(screen, RED, (int(c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
            elif board[r][c] == 2:
                pygame.draw.circle(screen, YELLOW, (int(c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
    pygame.display.update()

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon  
        self.epsilon_decay = epsilon_decay  
        self.epsilon_min = epsilon_min  

    def get_state(self, board):
        return tuple(map(tuple, board))

    def choose_action(self, board, valid_moves):
        state = self.get_state(board)
        if random.uniform(0, 1) < self.epsilon:  
            return random.choice(valid_moves)
       
        if state not in self.q_table:
            return random.choice(valid_moves)  
        return max(valid_moves, key=lambda col: self.q_table[state].get(col, 0))

    def update_q_table(self, board, action, reward, next_board):
        state = self.get_state(board)
        next_state = self.get_state(next_board)
        if state not in self.q_table:
            self.q_table[state] = {}
        if next_state not in self.q_table:
            self.q_table[next_state] = {}
        max_future_q = max(self.q_table[next_state].values(), default=0)
        current_q = self.q_table[state].get(action, 0)
        self.q_table[state][action] = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)

    def decay_epsilon(self):
       
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def print_q_table(self, num_entries=5):
        print("Q-table (first {} entries):".format(num_entries))
        for i, (state, actions) in enumerate(self.q_table.items()):
            if i >= num_entries:
                break
            print("State:", state)
            print("Actions:", actions)

agent = QLearningAgent(epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01)

board = create_board()
print_board(board)
game_over = False
turn = 0

# Win counters
random_wins = 0
ai_wins = 0

pygame.init()
SQUARESIZE = 100
width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT+1) * SQUARESIZE
size = (width, height)
RADIUS = int(SQUARESIZE/2 - 5)
screen = pygame.display.set_mode(size)
draw_board(board)
pygame.display.update()
myfont = pygame.font.SysFont("monospace", 75)

def reset_game():
    global board, game_over, turn
    board = create_board()
    game_over = False
    turn = 0
    draw_board(board)
    agent.print_q_table()

def draw_win_counters():
   
    pygame.draw.rect(screen, BLACK, (10, 10, width - 20, 50))  

   
    random_wins_text = myfont.render(f"Random: {random_wins}", 1, WHITE)
    screen.blit(random_wins_text, (10, 10))

   
    ai_wins_text = myfont.render(f"AI: {ai_wins}", 1, WHITE)
    screen.blit(ai_wins_text, (width - ai_wins_text.get_width() - 10, 10))


def draw_text_loop():
    while True:
        draw_win_counters()
        pygame.display.update()
        pygame.time.wait(100)  

# Start de tekst-loop in een aparte thread
text_thread = threading.Thread(target=draw_text_loop)
text_thread.daemon = True  
text_thread.start()

while True:
    if turn == 0 and not game_over:
        pygame.time.wait(100)  
        valid_moves = [c for c in range(COLUMN_COUNT) if is_valid_location(board, c)]
        if valid_moves:
           
            old_board = np.copy(board)
           
           
            col = random.choice(valid_moves)
            row = get_next_open_row(board, col)
            drop_piece(board, row, col, 1)
           
           
            reward = get_reward(old_board, board, 1)
           
            draw_board(board)
            if winning_move(board, 1):
                random_wins += 1  
                print("Random speler wint!")
                pygame.time.wait(1000)
                reset_game()
                continue
            turn = 1
        elif is_board_full(board):
            print("Gelijkspel! Het bord is vol.")
            pygame.time.wait(1000)
            reset_game()
            continue
   
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
       
    if turn == 1 and not game_over:
        pygame.time.wait(100)  
        valid_moves = [c for c in range(COLUMN_COUNT) if is_valid_location(board, c)]
        if valid_moves:
           
            old_board = np.copy(board)
           
           
            col = agent.choose_action(board, valid_moves)
            row = get_next_open_row(board, col)
            drop_piece(board, row, col, 2)
           
         
            reward = get_reward(old_board, board, 2)
            agent.update_q_table(old_board, col, reward, board)
           
            draw_board(board)
            if winning_move(board, 2):
                ai_wins += 1
                print("AI wint!")
                pygame.time.wait(1000)  
                reset_game()
                continue
            turn = 0
        elif is_board_full(board):  
            print("Gelijkspel! Het bord is vol.")
            pygame.time.wait(1000)
            reset_game()
            continue

   
    agent.decay_epsilon()

   
    draw_win_counters()
    pygame.display.update()

