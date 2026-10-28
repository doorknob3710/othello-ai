import numpy as np
from numba import njit, objmode
import random
import time

# --- グローバル定数 ---
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

# 方向ベクトル
DIRECTIONS = np.array([
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1)
], dtype=np.int8)


# --- Numba 最適化関数群 (@njit) ---

@njit(cache=True)
def get_valid_moves(board, player):
    moves = np.empty((64, 2), dtype=np.int8)
    count = 0
    opponent = 3 - player
    
    for r in range(8):
        for c in range(8):
            if board[r][c] != EMPTY:
                continue
            
            is_valid_move = False
            for d in range(8):
                dr, dc = DIRECTIONS[d]
                nr, nc = r + dr, c + dc
                
                if 0 <= nr < 8 and 0 <= nc < 8 and board[nr][nc] == opponent:
                    nr += dr
                    nc += dc
                    while 0 <= nr < 8 and 0 <= nc < 8:
                        if board[nr][nc] == EMPTY:
                            break
                        if board[nr][nc] == player:
                            is_valid_move = True
                            break
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
def evaluate_final(board, player):
    opponent = 3 - player
    score = 0
    flat_board = board.ravel()
    for i in range(64):
        if flat_board[i] == player:
            score += 1
        elif flat_board[i] == opponent:
            score -= 1
    return score

# --- タイムアウト対応版 NegaScout ---
@njit(cache=True)
def negascout_with_timeout(board, player, depth, alpha, beta, deadline, shared_stats):
    if shared_stats[0] == 1:
        return 0
    
    shared_stats[1] += 1
    
    # 1024ノードごとに時間チェック
    if (shared_stats[1] & 1023) == 0:
        current_time = 0.0
        with objmode(current_time='float64'):
            current_time = time.time()
            
        if current_time > deadline:
            shared_stats[0] = 1
            return 0

    opponent = 3 - player

    if depth == 0:
        return evaluate(board, player)

    valid_moves = get_valid_moves(board, player)

    if len(valid_moves) == 0:
        if len(get_valid_moves(board, opponent)) == 0:
            return evaluate(board, player)
        return -negascout_with_timeout(board, opponent, depth - 1, -beta, -alpha, deadline, shared_stats)

    is_first_move = True
    for i in range(len(valid_moves)):
        move = valid_moves[i]
        r, c = move[0], move[1]
        
        temp_board = board.copy()
        flip_stones(temp_board, r, c, player)
        
        score = 0
        if is_first_move:
            score = -negascout_with_timeout(temp_board, opponent, depth - 1, -beta, -alpha, deadline, shared_stats)
            is_first_move = False
        else:
            score = -negascout_with_timeout(temp_board, opponent, depth - 1, -alpha - 1, -alpha, deadline, shared_stats)
            if alpha < score < beta:
                score = -negascout_with_timeout(temp_board, opponent, depth - 1, -beta, -score, deadline, shared_stats)
        
        if shared_stats[0] == 1:
            return 0

        if score > alpha:
            alpha = score
        if alpha >= beta:
            return alpha

    return alpha

@njit(cache=True)
def negascout_final(board, player, depth, alpha, beta):
    opponent = 3 - player

    if depth == 0:
        return evaluate_final(board, player)

    valid_moves = get_valid_moves(board, player)

    if len(valid_moves) == 0:
        if len(get_valid_moves(board, opponent)) == 0:
            return evaluate_final(board, player)
        return -negascout_final(board, opponent, depth - 1, -beta, -alpha)

    is_first_move = True
    for i in range(len(valid_moves)):
        move = valid_moves[i]
        r, c = move[0], move[1]
        
        temp_board = board.copy()
        flip_stones(temp_board, r, c, player)
        
        score = 0
        if is_first_move:
            score = -negascout_final(temp_board, opponent, depth - 1, -beta, -alpha)
            is_first_move = False
        else:
            score = -negascout_final(temp_board, opponent, depth - 1, -alpha - 1, -alpha)
            if alpha < score < beta:
                score = -negascout_final(temp_board, opponent, depth - 1, -beta, -score)
        
        if score > alpha:
            alpha = score
        if alpha >= beta:
            return alpha

    return alpha

# --- Python側 (UI・制御) ---

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

def algebraic_to_numeric(move_str):
    try:
        if len(move_str) != 2: return None
        col = ord(move_str[0].lower()) - ord('a')
        row = int(move_str[1])
        if 0 <= col < 8 and 0 <= row < 8:
            return (row, col)
    except:
        pass
    return None

def get_winner(board):
    b_cnt = np.sum(board == BLACK)
    w_cnt = np.sum(board == WHITE)
    diff = b_cnt - w_cnt
    if diff > 0: return "黒の勝ち"
    elif diff < 0: return "白の勝ち"
    else: return "引き分け"

def search_at_depth(board, player, depth, is_exact, deadline=None):
    valid_moves = get_valid_moves(board, player)
    best_score = -float('inf')
    best_move = None
    alpha = -float('inf')
    beta = float('inf')
    
    shared_stats = np.array([0, 0], dtype=np.int32)

    for i in range(len(valid_moves)):
        if deadline is not None and time.time() > deadline:
            return None, 0

        move = valid_moves[i]
        r, c = move[0], move[1]
        
        temp_board = board.copy()
        flip_stones(temp_board, r, c, player)
        
        if is_exact:
            score = -negascout_final(temp_board, 3-player, depth-1, -beta, -alpha)
        else:
            if deadline is None:
                dl = time.time() + 999999
            else:
                dl = deadline
                
            score = -negascout_with_timeout(temp_board, 3-player, depth-1, -beta, -alpha, dl, shared_stats)
        
        if shared_stats[0] == 1:
            return None, 0

        if score > best_score:
            best_score = score
            best_move = (r, c)
        
        alpha = max(alpha, best_score)
        
    return best_move, best_score

def computer_move_iterative(board, player, max_depth, time_limit=10.0):
    """
    反復深化 (Iterative Deepening)
    """
    valid_moves = get_valid_moves(board, player)
    if len(valid_moves) == 0:
        return None, False, 0
    if len(valid_moves) == 1:
        return (valid_moves[0][0], valid_moves[0][1]), False, 0

    start_time = time.time()
    deadline = start_time + time_limit
    
    empty_count = np.sum(board == EMPTY)
    EXACT_SEARCH_THRESHOLD = 11
    
    if empty_count <= EXACT_SEARCH_THRESHOLD:
        best_move, best_score = search_at_depth(board, player, int(empty_count), True)
        return best_move, True, best_score

    best_move = None
    best_score = 0
    
    # 深さ1から順に (1, 2, 3, ...)
    current_depth = 1
    
    # 初期探索（少し猶予を与える）
    first_move, first_score = search_at_depth(board, player, current_depth, False, deadline=deadline + 5.0)
    best_move = first_move
    best_score = first_score
    
    current_depth += 1

    while current_depth <= max_depth:
        if time.time() >= deadline:
            break
            
        move, score = search_at_depth(board, player, current_depth, False, deadline)
        
        if move is None:
            break
        
        best_move = move
        best_score = score
        
        elapsed = time.time() - start_time
        
        # 次の深さへの見積もり (1ずつ増えるなら5倍程度で見積もる)
        if elapsed * 5 > (time_limit * 0.9) or elapsed > (time_limit * 0.8):
            current_depth += 1
            break
            
        current_depth += 1
        
    return best_move, False, best_score, current_depth - 1

# --- メイン処理 ---

def main():
    print("========================================")
    print("  高速リバーシ: 10手 vs 11手 対戦")
    print("========================================")
    
    print("モードを選択:")
    print("  1. 指定深さ対決: 10手 vs 11手")
    print("  2. 指定深さ対決: AI同士（同じ条件）")
    print("  3. 人間 vs AI")
    
    while True:
        mode = input("番号を入力 (1, 2, 3): ")
        if mode in ["1", "2", "3"]:
            break
        print("1, 2, 3 のいずれかを入力してください。")

    player_black_type = "AI"
    player_white_type = "AI"
    
    # デフォルト設定
    depth_black = 10
    depth_white = 10
    limit_black = 10.0
    limit_white = 10.0

    if mode == "1":
        print("\n>>> 10手 vs 11手 対戦（時間制限なし） <<<")
        print("どっちを11手読みにしますか？")
        print("  1. 黒:10手  vs  白:11手")
        print("  2. 黒:11手  vs  白:10手")
        
        while True:
            sub = input("番号を入力 (1 or 2): ")
            if sub in ["1", "2"]:
                break
            print("1 か 2 を入力してください。")
            
        if sub == "1":
            print("設定: 黒(Depth 10) vs 白(Depth 11)")
            depth_black = 10
            depth_white = 11
        else:
            print("設定: 黒(Depth 11) vs 白(Depth 10)")
            depth_black = 11
            depth_white = 10
            
        # 時間制限なし（実質無限）
        limit_black = 999999.0
        limit_white = 999999.0
        
    elif mode == "2":
        d = input("AIの深さを入力してください (推奨: 8〜10): ")
        try:
            val = int(d)
            depth_black = val
            depth_white = val
        except:
            depth_black = 10
            depth_white = 10
        print(f"AI同士 (Depth {depth_black}) で開始します。")

    elif mode == "3":
        print("\nあなたはどちらを持ちますか？")
        print("  1. 黒 (先手)")
        print("  2. 白 (後手)")
        while True:
            side = input("番号を入力 (1 or 2): ")
            if side == "1":
                player_black_type = "HUMAN"
                player_white_type = "AI"
                break
            elif side == "2":
                player_black_type = "AI"
                player_white_type = "HUMAN"
                break
            print("1 か 2 を入力してください。")

    print("----------------------------------------")
    if mode == "1":
        print(f"黒: Max Depth {depth_black} (Limit 無制限)")
        print(f"白: Max Depth {depth_white} (Limit 無制限)")
    else:
        print(f"制限時間: 10.0秒 / 手")
    print("残り11手以下で必勝読み (完全読み) を行います")
    print("----------------------------------------")
    
    time.sleep(1.5)

    board = board_start()
    print_board(board)
    
    turn_count = 1
    
    while True:
        b_moves = get_valid_moves(board, BLACK)
        w_moves = get_valid_moves(board, WHITE)
        if not board_check(board) and len(b_moves) == 0 and len(w_moves) == 0:
            break

        # --- 黒の手番 ---
        current_player = BLACK
        current_type = player_black_type
        moves = get_valid_moves(board, current_player)
        
        # モード1の場合、個別の設定を使う
        d_max = depth_black
        t_lim = limit_black if mode == "1" else 10.0
        
        print(f"\n[Turn {turn_count}] 黒の手番 ({current_type}) - 目標Depth: {d_max}")
        
        if len(moves) == 0:
            print(">> 黒: パス")
        else:
            if current_type == "HUMAN":
                readable_moves = [numeric_to_algebraic(m[0], m[1]) for m in moves]
                print(f"置ける場所: {readable_moves}")
                while True:
                    user_input = input("場所を入力(例: f5): ")
                    pos = algebraic_to_numeric(user_input)
                    valid = False
                    if pos:
                        for m in moves:
                            if m[0] == pos[0] and m[1] == pos[1]:
                                valid = True
                                break
                    if valid:
                        flip_stones(board, pos[0], pos[1], current_player)
                        print(f">> 黒は {user_input} に置きました。")
                        break
                    else:
                        print("そこには置けません。")
            else:
                # AI
                start_time = time.time()
                result = computer_move_iterative(board, current_player, d_max, t_lim)
                end_time = time.time()
                
                if len(result) == 4:
                    best_move, is_exact, score_est, reached_depth = result
                else:
                    best_move, is_exact, score_est = result
                    reached_depth = "Complete"

                if best_move:
                    r, c = best_move
                    move_str = numeric_to_algebraic(r, c)
                    msg = f">> 黒(AI)は {move_str} に置きました。"
                    
                    if is_exact:
                        msg += f" 【必勝読み: 差 {score_est}】"
                    else:
                        msg += f" (Depth {reached_depth}, 時間: {end_time - start_time:.2f}秒)"
                    
                    print(msg)
                    flip_stones(board, r, c, current_player)
                else:
                    r, c = moves[random.randint(0, len(moves)-1)]
                    flip_stones(board, r, c, current_player)

        print_board(board)
        if not board_check(board): break

        # --- 白の手番 ---
        current_player = WHITE
        current_type = player_white_type
        moves = get_valid_moves(board, current_player)

        # モード1の場合、個別の設定を使う
        d_max = depth_white
        t_lim = limit_white if mode == "1" else 10.0
        
        print(f"\n[Turn {turn_count}] 白の手番 ({current_type}) - 目標Depth: {d_max}")
        
        if len(moves) == 0:
            print(">> 白: パス")
        else:
            if current_type == "HUMAN":
                readable_moves = [numeric_to_algebraic(m[0], m[1]) for m in moves]
                print(f"置ける場所: {readable_moves}")
                while True:
                    user_input = input("場所を入力(例: f5): ")
                    pos = algebraic_to_numeric(user_input)
                    valid = False
                    if pos:
                        for m in moves:
                            if m[0] == pos[0] and m[1] == pos[1]:
                                valid = True
                                break
                    if valid:
                        flip_stones(board, pos[0], pos[1], current_player)
                        print(f">> 白は {user_input} に置きました。")
                        break
                    else:
                        print("そこには置けません。")
            else:
                # AI
                start_time = time.time()
                result = computer_move_iterative(board, current_player, d_max, t_lim)
                end_time = time.time()
                
                if len(result) == 4:
                    best_move, is_exact, score_est, reached_depth = result
                else:
                    best_move, is_exact, score_est = result
                    reached_depth = "Complete"

                if best_move:
                    r, c = best_move
                    move_str = numeric_to_algebraic(r, c)
                    msg = f">> 白(AI)は {move_str} に置きました。"
                    
                    if is_exact:
                        msg += f" 【必勝読み: 差 {score_est}】"
                    else:
                        msg += f" (Depth {reached_depth}, 時間: {end_time - start_time:.2f}秒)"
                        
                    print(msg)
                    flip_stones(board, r, c, current_player)
                else:
                    r, c = moves[random.randint(0, len(moves)-1)]
                    flip_stones(board, r, c, current_player)

        print_board(board)
        turn_count += 1

    # 結果
    print("========================================")
    print("  ゲーム終了")
    print("========================================")
    print(get_winner(board))
    b_cnt = np.sum(board == BLACK)
    w_cnt = np.sum(board == WHITE)
    print(f"黒: {b_cnt} 石")
    print(f"白: {w_cnt} 石")

if __name__ == "__main__":
    main()