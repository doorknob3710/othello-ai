import numpy as np
from numba import njit
import random
import time

# --- グローバル定数 (Numba内で使うために型を意識) ---
# 空:0, 白:1, 黒:2
EMPTY = 0
WHITE = 1
BLACK = 2

# 評価ボード
CELL_WEIGHT = np.array([
    2714,  147,   69,  -18,  -18,   69,  147, 2714,
     147, -577, -186, -153, -153, -186, -577,  147,
      69, -186, -379, -122, -122, -379, -186,   69,
     -18, -153, -122, -169, -169, -122, -153,  -18,
     -18, -153, -122, -169, -169, -122, -153,  -18,
      69, -186, -379, -122, -122, -379, -186,   69,
     147, -577, -186, -153, -153, -186, -577,  147,
    2714,  147,   69,  -18,  -18,   69,  147, 2714
], dtype=np.int32).reshape(8, 8)

# 方向ベクトル (8方向)
DIRECTIONS = np.array([
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1)
], dtype=np.int8)


# --- Numba 最適化関数群 (@njit) ---

@njit(cache=True)
def get_valid_moves(board, player):
    """
    石を置ける場所を返す。
    """
    moves = np.empty((64, 2), dtype=np.int8) # 最大64箇所
    count = 0
    opponent = 3 - player
    
    rows, cols = board.shape
    
    for r in range(8):
        for c in range(8):
            if board[r][c] != EMPTY:
                continue
            
            # 8方向チェック
            is_valid_move = False
            for d in range(8):
                dr, dc = DIRECTIONS[d]
                nr, nc = r + dr, c + dc
                
                # 隣が相手の石か確認
                if 0 <= nr < 8 and 0 <= nc < 8 and board[nr][nc] == opponent:
                    # さらにその先を調べる
                    nr += dr
                    nc += dc
                    while 0 <= nr < 8 and 0 <= nc < 8:
                        if board[nr][nc] == EMPTY:
                            break
                        if board[nr][nc] == player:
                            is_valid_move = True
                            break
                        # 相手の石なら継続
                        nr += dr
                        nc += dc
                
                if is_valid_move:
                    break
            
            if is_valid_move:
                moves[count][0] = r
                moves[count][1] = c
                count += 1
                
    return moves[:count]

@njit(cache=True)
def flip_stones(board, row, col, player):
    """
    石を置き、挟んだ石を裏返す（盤面を直接書き換える）
    """
    board[row, col] = player
    opponent = 3 - player
    
    for d in range(8):
        dr, dc = DIRECTIONS[d]
        r, c = row + dr, col + dc
        
        flip_count = 0
        
        while 0 <= r < 8 and 0 <= c < 8:
            if board[r, c] == EMPTY:
                flip_count = 0
                break
            if board[r, c] == opponent:
                flip_count += 1
            elif board[r, c] == player:
                break 
            
            r += dr
            c += dc
        else:
            flip_count = 0

        if flip_count > 0:
            r, c = row + dr, col + dc
            for _ in range(flip_count):
                board[r, c] = player
                r += dr
                c += dc

@njit(cache=True)
def evaluate(board, player):
    """
    評価関数
    """
    opponent = 3 - player
    score = 0
    flat_board = board.ravel()
    flat_weights = CELL_WEIGHT.ravel()
    
    for i in range(64):
        if flat_board[i] == player:
            score += flat_weights[i]
        elif flat_board[i] == opponent:
            score -= flat_weights[i]
            
    return score

@njit(cache=True)
def negascout(board, player, depth, alpha, beta):
    """
    NegaScout法
    """
    opponent = 3 - player

    if depth == 0:
        return evaluate(board, player)

    valid_moves = get_valid_moves(board, player)

    if len(valid_moves) == 0:
        if len(get_valid_moves(board, opponent)) == 0:
            return evaluate(board, player)
        return -negascout(board, opponent, depth - 1, -beta, -alpha)

    is_first_move = True
    
    for i in range(len(valid_moves)):
        move = valid_moves[i]
        r, c = move[0], move[1]
        
        temp_board = board.copy()
        flip_stones(temp_board, r, c, player)
        
        score = 0
        if is_first_move:
            score = -negascout(temp_board, opponent, depth - 1, -beta, -alpha)
            is_first_move = False
        else:
            score = -negascout(temp_board, opponent, depth - 1, -alpha - 1, -alpha)
            if alpha < score < beta:
                score = -negascout(temp_board, opponent, depth - 1, -beta, -score)
        
        if score > alpha:
            alpha = score
        
        if alpha >= beta:
            return alpha

    return alpha

# --- Python側 (UI周り) ---

def print_board(board):
    print("\n  a b c d e f g h")
    symbols = {EMPTY: "🟩", WHITE: "⚪️", BLACK: "⚫️"}
    for i in range(8):
        row_str = f"{i} "
        for j in range(8):
            row_str += symbols[board[i, j]] + " "
        print(row_str)
    print()

def board_start():
    board = np.zeros((8, 8), dtype=np.int8)
    board[3, 3] = BLACK
    board[3, 4] = WHITE
    board[4, 3] = WHITE
    board[4, 4] = BLACK
    return board

def board_check(board):
    return np.any(board == EMPTY)

def numeric_to_algebraic(row, col):
    return f"{chr(ord('a') + col)}{row}"

def get_winner(board):
    b_cnt = np.sum(board == BLACK)
    w_cnt = np.sum(board == WHITE)
    if b_cnt > w_cnt: return "黒の勝ち"
    elif w_cnt > b_cnt: return "白の勝ち"
    else: return "引き分け"


def computer_move_numba(board, player, depth):
    valid_moves = get_valid_moves(board, player)
    if len(valid_moves) == 0:
        return None
    
    best_score = -float('inf')
    best_move = None
    alpha = -float('inf')
    beta = float('inf')
    
    # 最初の1手目も並列化はせず、ループで回す
    for i in range(len(valid_moves)):
        move = valid_moves[i]
        r, c = move[0], move[1]
        
        temp_board = board.copy()
        flip_stones(temp_board, r, c, player)
        
        score = -negascout(temp_board, 3-player, depth-1, -beta, -alpha)
        
        if score > best_score:
            best_score = score
            best_move = (r, c)
        
        alpha = max(alpha, best_score)
        
    return best_move


# --- AI対戦メインループ ---

def main():
    print("========================================")
    print("  Numba版リバーシ: AI vs AI 特別対戦")
    print("========================================")
    
    board = board_start()
    print_board(board)
    
    # ここで深さを設定
    DEPTH_BLACK = 11
    DEPTH_WHITE = 12
    
    print(f"【設定】")
    print(f"  先手 (黒): Depth {DEPTH_BLACK}")
    print(f"  後手 (白): Depth {DEPTH_WHITE}")
    print("----------------------------------------")
    print("対戦を開始します...(最初の1手はコンパイルが入るため遅いです)")
    time.sleep(1)

    turn_count = 1
    
    while True:
        # 終了判定
        if not board_check(board) and len(get_valid_moves(board, BLACK)) == 0 and len(get_valid_moves(board, WHITE)) == 0:
            break

        # --- 黒の手番 ---
        current = BLACK
        moves = get_valid_moves(board, current)
        
        print(f"\n[Turn {turn_count}] 黒の手番 (Depth {DEPTH_BLACK})")
        if len(moves) == 0:
            print(">> 黒: パス")
        else:
            start_time = time.time()
            best_move = computer_move_numba(board, current, DEPTH_BLACK)
            end_time = time.time()
            
            if best_move:
                r, c = best_move
                print(f">> 黒は {numeric_to_algebraic(r, c)} に置きました。 (思考時間: {end_time - start_time:.2f}秒)")
                flip_stones(board, r, c, current)
            else:
                # 万が一のランダム
                r, c = moves[random.randint(0, len(moves)-1)]
                print(f">> 黒は {numeric_to_algebraic(r, c)} に置きました (Random Fallback)。")
                flip_stones(board, r, c, current)
                
        print_board(board)
        if not board_check(board): break # 盤面が埋まったら即終了
        
        # --- 白の手番 ---
        current = WHITE
        moves = get_valid_moves(board, current)
        
        print(f"\n[Turn {turn_count}] 白の手番 (Depth {DEPTH_WHITE})")
        if len(moves) == 0:
            print(">> 白: パス")
        else:
            start_time = time.time()
            best_move = computer_move_numba(board, current, DEPTH_WHITE)
            end_time = time.time()
            
            if best_move:
                r, c = best_move
                print(f">> 白は {numeric_to_algebraic(r, c)} に置きました。 (思考時間: {end_time - start_time:.2f}秒)")
                flip_stones(board, r, c, current)
            else:
                r, c = moves[random.randint(0, len(moves)-1)]
                print(f">> 白は {numeric_to_algebraic(r, c)} に置きました (Random Fallback)。")
                flip_stones(board, r, c, current)

        print_board(board)
        turn_count += 1

    # 結果
    print("========================================")
    print("  ゲーム終了")
    print("========================================")
    print(get_winner(board))
    b_cnt = np.sum(board == BLACK)
    w_cnt = np.sum(board == WHITE)
    print(f"黒 (Depth {DEPTH_BLACK}): {b_cnt} 石")
    print(f"白 (Depth {DEPTH_WHITE}): {w_cnt} 石")

if __name__ == "__main__":
    main()