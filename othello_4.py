import numpy as np
from numba import njit
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
    """中盤までの評価関数（重み付け）"""
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
    """終盤の評価関数（純粋な石の数差）"""
    opponent = 3 - player
    score = 0
    flat_board = board.ravel()
    for i in range(64):
        if flat_board[i] == player:
            score += 1
        elif flat_board[i] == opponent:
            score -= 1
    return score

@njit(cache=True)
def negascout(board, player, depth, alpha, beta):
    """通常探索（重み付け評価）"""
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

@njit(cache=True)
def negascout_final(board, player, depth, alpha, beta):
    """完全読み用探索（石数評価）"""
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

# --- Python側 (UI・制御・時間管理) ---

def print_board(board):
    print("\n  a　b　c　d　e　f　g　h")
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

def search_at_depth(board, player, depth, is_exact):
    """
    指定された深さで1回探索を行うヘルパー関数
    """
    valid_moves = get_valid_moves(board, player)
    best_score = -float('inf')
    best_move = None
    alpha = -float('inf')
    beta = float('inf')

    for i in range(len(valid_moves)):
        move = valid_moves[i]
        r, c = move[0], move[1]
        
        temp_board = board.copy()
        flip_stones(temp_board, r, c, player)
        
        if is_exact:
            # 完全読み
            score = -negascout_final(temp_board, 3-player, depth-1, -beta, -alpha)
        else:
            # 通常読み
            score = -negascout(temp_board, 3-player, depth-1, -beta, -alpha)
        
        if score > best_score:
            best_score = score
            best_move = (r, c)
        
        alpha = max(alpha, best_score)
        
    return best_move, best_score

def computer_move_iterative(board, player, max_depth, time_limit=10.0):
    """
    反復深化 (Iterative Deepening)
    時間が許す限り深く読む。
    """
    valid_moves = get_valid_moves(board, player)
    if len(valid_moves) == 0:
        return None, False, 0
    
    # 1手しかない場合は探索不要で即返し（時間節約）
    if len(valid_moves) == 1:
        return (valid_moves[0][0], valid_moves[0][1]), False, 0

    start_time = time.time()
    
    # 空きマス数チェック
    empty_count = np.sum(board == EMPTY)
    
    # 必勝読みの閾値 (13だと重いことがあるので11に調整)
    EXACT_SEARCH_THRESHOLD = 18
    
    # 終盤の必勝読みモード
    if empty_count <= EXACT_SEARCH_THRESHOLD:
        # 時間制限は気にせず、完全読みを実行する (11手なら数秒以内)
        # ただしあまりに時間がかかりそうならここも制限を入れるべきだが、
        # Numbaなら11手は十分速いと仮定。
        best_move, best_score = search_at_depth(board, player, int(empty_count), True)
        return best_move, True, best_score

    # 通常探索（反復深化）
    best_move = None
    best_score = 0
    current_depth = 1
    reached_max = False

    while current_depth <= max_depth:
        # 探索実行
        move, score = search_at_depth(board, player, current_depth, False)
        
        # 結果を保存
        best_move = move
        best_score = score
        
        elapsed = time.time() - start_time
        
        # 次の深さに進むか判定
        # 一般に深さが1増えると計算時間は3~5倍になる。
        # 次の深さの予想時間 = elapsed * 5 くらいと見積もる。
        # 余裕を持って判定 (制限時間の90%くらいを目安に)
        if elapsed * 5 > (time_limit * 0.9) or elapsed > (time_limit * 0.8):
            break
            
        current_depth += 1
        
    return best_move, False, best_score, current_depth

# --- メイン処理 ---

def main():
    
    print("モードを選択:")
    print("  1. AI(MAX 10手) vs AI(MAX 10手)")
    print("  2. 人間 vs AI(MAX 10手)")
    
    while True:
        mode = input("番号を入力 (1 or 2): ")
        if mode in ["1", "2"]:
            break
        print("1 か 2 を入力してください。")

    player_black_type = "AI"
    player_white_type = "AI"
    
    # 時間制限（秒）
    TIME_LIMIT = 10.0
    MAX_DEPTH = 10

    if mode == "2":
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
    else:
        print("\nAI vs AI (MAX 10手, 10秒制限) を開始します。")

    print("----------------------------------------")
    print(f"制限時間: {TIME_LIMIT}秒 / 手")
    print("時間内で可能な限り深く読みます (反復深化)")
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
        
        print(f"\n[Turn {turn_count}] 黒の手番 ({current_type})")
        
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
                # computer_move_iterative を呼び出す
                # 戻り値が可変長なので注意して受け取る
                result = computer_move_iterative(board, current_player, MAX_DEPTH, TIME_LIMIT)
                end_time = time.time()
                
                if len(result) == 4:
                    best_move, is_exact, score_est, reached_depth = result
                else:
                    # 必勝読みの場合は depth が返ってこない
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
        
        print(f"\n[Turn {turn_count}] 白の手番 ({current_type})")
        
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
                result = computer_move_iterative(board, current_player, MAX_DEPTH, TIME_LIMIT)
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