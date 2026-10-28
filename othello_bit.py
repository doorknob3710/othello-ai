import numpy as np
from numba import njit, uint64, int8, int32, int64, float64, boolean, objmode
import random
import time

# --- グローバル定数 ---
EMPTY = 0
WHITE = 1
BLACK = 2

# ビットボード定数
MASK_X = uint64(0x7e7e7e7e7e7e7e7e)

# 評価ボード
WEIGHTS = np.array([
    2714,  147,   69,  -18,  -18,   69,  147, 2714,
     147, -577, -186, -153, -153, -186, -577,  147,
      69, -186, -379, -122, -122, -379, -186,   69,
     -18, -153, -122, -169, -169, -122, -153,  -18,
     -18, -153, -122, -169, -169, -122, -153,  -18,
      69, -186, -379, -122, -122, -379, -186,   69,
     147, -577, -186, -153, -153, -186, -577,  147,
    2714,  147,   69,  -18,  -18,   69,  147, 2714,
], dtype=np.int32)

# --- Numba ビット演算関数群 ---

@njit("int32(uint64)", cache=False)
def popcount(x):
    x = x - ((x >> uint64(1)) & uint64(0x5555555555555555))
    x = (x & uint64(0x3333333333333333)) + ((x >> uint64(2)) & uint64(0x3333333333333333))
    x = (x + (x >> uint64(4))) & uint64(0x0f0f0f0f0f0f0f0f)
    x = x + (x >> uint64(8))
    x = x + (x >> uint64(16))
    x = x + (x >> uint64(32))
    return int32(x & uint64(0x7f))

@njit("uint64(uint64, uint64)", cache=False)
def get_legal_moves_bits(player_bb, opponent_bb):
    blank = ~(player_bb | opponent_bb)
    moves = uint64(0)
    
    mask = opponent_bb & MASK_X
    t = mask & (player_bb >> uint64(1))
    for _ in range(5): t |= mask & (t >> uint64(1))
    moves |= blank & (t >> uint64(1))
    
    mask = opponent_bb & MASK_X
    t = mask & (player_bb << uint64(1))
    for _ in range(5): t |= mask & (t << uint64(1))
    moves |= blank & (t << uint64(1))

    mask = opponent_bb
    t = mask & (player_bb << uint64(8))
    for _ in range(5): t |= mask & (t << uint64(8))
    moves |= blank & (t << uint64(8))
    
    mask = opponent_bb
    t = mask & (player_bb >> uint64(8))
    for _ in range(5): t |= mask & (t >> uint64(8))
    moves |= blank & (t >> uint64(8))
    
    mask = opponent_bb & MASK_X
    t = mask & (player_bb << uint64(7))
    for _ in range(5): t |= mask & (t << uint64(7))
    moves |= blank & (t << uint64(7))
    
    mask = opponent_bb & MASK_X
    t = mask & (player_bb << uint64(9))
    for _ in range(5): t |= mask & (t << uint64(9))
    moves |= blank & (t << uint64(9))
    
    mask = opponent_bb & MASK_X
    t = mask & (player_bb >> uint64(9))
    for _ in range(5): t |= mask & (t >> uint64(9))
    moves |= blank & (t >> uint64(9))
    
    mask = opponent_bb & MASK_X
    t = mask & (player_bb >> uint64(7))
    for _ in range(5): t |= mask & (t >> uint64(7))
    moves |= blank & (t >> uint64(7))
    
    return moves

@njit("UniTuple(uint64, 2)(uint64, uint64, uint64)", cache=False)
def make_move_bit(player_bb, opponent_bb, move_bit):
    rev = uint64(0)
    
    mask = uint64(move_bit >> uint64(1))
    processed = uint64(0)
    while (mask & MASK_X) and (mask & opponent_bb):
        processed |= mask
        mask >>= uint64(1)
    if mask & player_bb: rev |= processed
    
    mask = uint64(move_bit << uint64(1))
    processed = uint64(0)
    while (mask & MASK_X) and (mask & opponent_bb):
        processed |= mask
        mask <<= uint64(1)
    if mask & player_bb: rev |= processed

    mask = uint64(move_bit << uint64(8))
    processed = uint64(0)
    while (mask) and (mask & opponent_bb):
        processed |= mask
        mask <<= uint64(8)
    if mask & player_bb: rev |= processed

    mask = uint64(move_bit >> uint64(8))
    processed = uint64(0)
    while (mask) and (mask & opponent_bb):
        processed |= mask
        mask >>= uint64(8)
    if mask & player_bb: rev |= processed
    
    mask = uint64(move_bit << uint64(7))
    processed = uint64(0)
    while (mask & MASK_X) and (mask & opponent_bb):
        processed |= mask
        mask <<= uint64(7)
    if mask & player_bb: rev |= processed

    mask = uint64(move_bit << uint64(9))
    processed = uint64(0)
    while (mask & MASK_X) and (mask & opponent_bb):
        processed |= mask
        mask <<= uint64(9)
    if mask & player_bb: rev |= processed
    
    mask = uint64(move_bit >> uint64(9))
    processed = uint64(0)
    while (mask & MASK_X) and (mask & opponent_bb):
        processed |= mask
        mask >>= uint64(9)
    if mask & player_bb: rev |= processed

    mask = uint64(move_bit >> uint64(7))
    processed = uint64(0)
    while (mask & MASK_X) and (mask & opponent_bb):
        processed |= mask
        mask >>= uint64(7)
    if mask & player_bb: rev |= processed

    return (player_bb ^ move_bit ^ rev), (opponent_bb ^ rev)

@njit("int64(uint64, uint64)", cache=False)
def evaluate_score(player_bb, opponent_bb):
    score = 0
    for i in range(64):
        mask = uint64(1) << uint64(63 - i)
        if player_bb & mask:
            score += WEIGHTS[i]
        elif opponent_bb & mask:
            score -= WEIGHTS[i]
            
    p_moves = get_legal_moves_bits(player_bb, opponent_bb)
    o_moves = get_legal_moves_bits(opponent_bb, player_bb)
    score += (popcount(p_moves) - popcount(o_moves)) * 100
    return score

@njit("int64(uint64, uint64)", cache=False)
def evaluate_exact(player_bb, opponent_bb):
    return int64(popcount(player_bb)) - int64(popcount(opponent_bb))

@njit("int8[:](uint64)", cache=False)
def get_move_list(legal_bits):
    moves = np.empty(64, dtype=np.int8)
    count = 0
    if legal_bits == 0:
        return moves[:0]
    for i in range(64):
        mask = uint64(1) << uint64(63 - i)
        if legal_bits & mask:
            moves[count] = i
            count += 1
    return moves[:count]

@njit("int8[:,:](uint64)", cache=False)
def get_move_list_as_coords(legal_bits):
    moves = np.empty((64, 2), dtype=np.int8)
    count = 0
    if legal_bits == 0:
        return moves[:0]
    for i in range(64):
        mask = uint64(1) << uint64(63 - i)
        if legal_bits & mask:
            moves[count][0] = i // 8
            moves[count][1] = i % 8
            count += 1
    return moves[:count]

@njit("int64(uint64, uint64, int64, int64, int64, float64, int32[:], boolean)", cache=False)
def negascout_bb(player_bb, opponent_bb, depth, alpha, beta, deadline, shared_stats, is_exact):
    if shared_stats[0] == 1:
        return 0
    
    shared_stats[1] += 1
    if (shared_stats[1] & 2047) == 0: 
        current_time = 0.0
        with objmode(current_time='float64'):
            current_time = time.time()
        if deadline > 0 and current_time > deadline:
            shared_stats[0] = 1
            return 0

    if depth == 0:
        if is_exact:
            return int64(evaluate_exact(player_bb, opponent_bb))
        else:
            return int64(evaluate_score(player_bb, opponent_bb))
            
    # --- MPC (Multi Prob Cut) ---
    if not is_exact and depth >= 4:
        if depth >= 6:
            mpc_depth = depth - 4
            mpc_margin = 3000 
        else:
            mpc_depth = depth - 2
            mpc_margin = 1500

        bound_high = beta + mpc_margin
        score_high = negascout_bb(player_bb, opponent_bb, mpc_depth, bound_high - 1, bound_high, deadline, shared_stats, False)
        if score_high >= bound_high:
            return score_high 
            
        bound_low = alpha - mpc_margin
        score_low = negascout_bb(player_bb, opponent_bb, mpc_depth, bound_low, bound_low + 1, deadline, shared_stats, False)
        if score_low <= bound_low:
            return score_low 
    # ---------------------------

    legal_bits = get_legal_moves_bits(player_bb, opponent_bb)
    
    if legal_bits == 0:
        opp_legal = get_legal_moves_bits(opponent_bb, player_bb)
        if opp_legal == 0:
            return int64(evaluate_exact(player_bb, opponent_bb))
        return -negascout_bb(opponent_bb, player_bb, depth - 1, -beta, -alpha, deadline, shared_stats, is_exact)

    move_indices = get_move_list(legal_bits)
    
    is_first = True
    for i in range(len(move_indices)):
        idx = move_indices[i]
        move_bit = uint64(1) << uint64(63 - idx)
        new_p, new_o = make_move_bit(player_bb, opponent_bb, move_bit)
        
        score = 0
        if is_first:
            score = -negascout_bb(new_o, new_p, depth - 1, -beta, -alpha, deadline, shared_stats, is_exact)
            is_first = False
        else:
            score = -negascout_bb(new_o, new_p, depth - 1, -alpha - 1, -alpha, deadline, shared_stats, is_exact)
            if alpha < score < beta:
                score = -negascout_bb(new_o, new_p, depth - 1, -beta, -score, deadline, shared_stats, is_exact)
        
        if shared_stats[0] == 1:
            return 0

        if score > alpha:
            alpha = score
        if alpha >= beta:
            return alpha
            
    return alpha

# --- 変換関数 & ラッパー (順序変更) ---

@njit("UniTuple(uint64, 2)(int8[:,:])", cache=False)
def array_to_bitboard(board):
    black_bb = uint64(0)
    white_bb = uint64(0)
    flat = board.ravel()
    for i in range(64):
        if flat[i] == 2: # BLACK
            black_bb |= (uint64(1) << uint64(63 - i))
        elif flat[i] == 1: # WHITE
            white_bb |= (uint64(1) << uint64(63 - i))
    return black_bb, white_bb

@njit("int8[:,:](uint64, uint64)", cache=False)
def bitboard_to_array(black_bb, white_bb):
    board = np.zeros((8, 8), dtype=np.int8)
    flat = board.ravel()
    for i in range(64):
        mask = uint64(1) << uint64(63 - i)
        if black_bb & mask:
            flat[i] = 2 # BLACK
        elif white_bb & mask:
            flat[i] = 1 # WHITE
    return board

# --- 定義順序を変更したコアロジック ---

# 1. _solve_bitboard_core を先に定義する
@njit("Tuple((int8[:], boolean, int64, int64))(uint64, uint64, int64, int64, float64, int64)", cache=False)
def _solve_bitboard_core(black_bb, white_bb, turn_color, max_depth, time_limit, exact_threshold):
    if turn_color == 2: # BLACK
        my_bb, op_bb = black_bb, white_bb
    else:
        my_bb, op_bb = white_bb, black_bb

    legal_bits = get_legal_moves_bits(my_bb, op_bb)
    
    no_move = np.array([-1, -1], dtype=np.int8)
    if legal_bits == 0:
        return no_move, False, 0, 0

    move_indices = get_move_list(legal_bits)
    if len(move_indices) == 1:
        idx = move_indices[0]
        ret_move = np.array([idx // 8, idx % 8], dtype=np.int8)
        return ret_move, False, 0, 0

    stone_count = popcount(my_bb | op_bb)
    empty_count = 64 - stone_count
    
    is_exact = False
    current_max_depth = max_depth
    
    start_time = 0.0
    with objmode(start_time='float64'):
        start_time = time.time()
        
    deadline = 0.0
    if empty_count <= exact_threshold:
        is_exact = True
        current_max_depth = empty_count
        deadline = 0.0
    else:
        deadline = start_time + time_limit

    best_move_idx = -1
    best_score = -99999999
    reached_depth = 0
    
    start_depth = 1
    if is_exact:
        start_depth = current_max_depth
        
    for d in range(start_depth, current_max_depth + 1):
        if not is_exact:
            curr = 0.0
            with objmode(curr='float64'):
                curr = time.time()
            if curr > deadline:
                break
        
        alpha = -99999999
        beta = 99999999
        shared_stats = np.array([0, 0], dtype=np.int32)
        
        temp_best_idx = -1
        temp_best_score = -99999999
        completed_depth = True
        
        for i in range(len(move_indices)):
            idx = move_indices[i]
            move_bit = uint64(1) << uint64(63 - idx)
            new_p, new_o = make_move_bit(my_bb, op_bb, move_bit)
            
            score = -negascout_bb(new_o, new_p, d - 1, -beta, -alpha, deadline, shared_stats, is_exact)
            
            if shared_stats[0] == 1:
                completed_depth = False
                break
            
            if score > temp_best_score:
                temp_best_score = score
                temp_best_idx = idx
            
            alpha = max(alpha, score)
        
        if completed_depth:
            best_move_idx = temp_best_idx
            best_score = temp_best_score
            reached_depth = d
            if is_exact: break
        else:
            break
            
    if best_move_idx != -1:
        ret_move = np.array([best_move_idx // 8, best_move_idx % 8], dtype=np.int8)
        return ret_move, is_exact, best_score, reached_depth
    else:
        rand_idx = 0
        with objmode(rand_idx='int64'):
            rand_idx = random.randint(0, len(move_indices) - 1)
        idx = move_indices[rand_idx]
        ret_move = np.array([idx // 8, idx % 8], dtype=np.int8)
        return ret_move, False, 0, 0

# 2. solve_bitboard_wrapper を _solve_bitboard_core の後に定義する
@njit("Tuple((int8[:], boolean, int64, int64))(int8[:,:], int64, int64, float64, int64)", cache=False)
def solve_bitboard_wrapper(board, player, max_depth, time_limit, exact_threshold):
    black_bb, white_bb = array_to_bitboard(board)
    return _solve_bitboard_core(black_bb, white_bb, player, max_depth, time_limit, exact_threshold)

# 3. _get_valid_moves_numba もラッパーより前に定義する（念のため）
@njit("int8[:,:](int8[:,:], int64)", cache=False)
def _get_valid_moves_numba(board, player):
    black_bb, white_bb = array_to_bitboard(board)
    if player == 2: # BLACK
        moves_bits = get_legal_moves_bits(black_bb, white_bb)
    else:
        moves_bits = get_legal_moves_bits(white_bb, black_bb)
    return get_move_list_as_coords(moves_bits)

# 4. _flip_stones_numba も同様
@njit("int8[:,:](int8[:,:], int64, int64, int64)", cache=False)
def _flip_stones_numba(board, row, col, player):
    black_bb, white_bb = array_to_bitboard(board)
    move_bit = uint64(1) << uint64(63 - (row * 8 + col))
    
    if player == 2: # BLACK
        new_black, new_white = make_move_bit(black_bb, white_bb, move_bit)
    else:
        new_white, new_black = make_move_bit(white_bb, black_bb, move_bit)
        
    return bitboard_to_array(new_black, new_white)

# --- Pythonラッパー関数 ---

def get_valid_moves(board, player):
    return _get_valid_moves_numba(board, player)

def flip_stones(board, row, col, player):
    new_board = _flip_stones_numba(board, row, col, player)
    np.copyto(board, new_board)

# --- UI部分 ---

def board_start():
    board = np.zeros((8, 8), dtype=np.int8)
    board[3, 3] = BLACK
    board[3, 4] = WHITE
    board[4, 3] = WHITE
    board[4, 4] = BLACK
    return board

def board_check(board):
    return np.any(board == EMPTY)

def print_board(board):
    print("\n  a　b　c　d　e　f　g　h")
    symbols = {EMPTY: "🟩", WHITE: "⚪️", BLACK: "⚫️"}
    for i in range(8):
        row_str = f"{i} "
        for j in range(8):
            row_str += symbols[board[i, j]] + " "
        print(row_str)
    print()

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

# --- メイン処理 ---

def main():
    print("========================================")
    print("  超高速リバーシ: BitBoard + MPC (NameError Fix)")
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
            if sub in ["1", "2"]: break
            
        if sub == "1":
            depth_black, depth_white = 10, 11
        else:
            depth_black, depth_white = 11, 10
        limit_black = 999999.0
        limit_white = 999999.0
        
    elif mode == "2":
        d = input("AIの深さを入力してください (推奨: 10〜14): ")
        try:
            val = int(d)
            depth_black, depth_white = val, val
        except:
            depth_black, depth_white = 10, 10
        print(f"AI同士 (Depth {depth_black}) で開始します。")

    elif mode == "3":
        print("\nあなたはどちらを持ちますか？")
        print("  1. 黒 (先手)")
        print("  2. 白 (後手)")
        while True:
            side = input("番号を入力 (1 or 2): ")
            if side == "1":
                player_black_type = "HUMAN"
                break
            elif side == "2":
                player_white_type = "HUMAN"
                break

    print("----------------------------------------")
    print("BitBoardエンジン + Multi Prob Cut (MPC) 搭載。")
    print("定義順序を修正し、NameErrorを回避しました。")
    print("残り17手以下で必勝読み (完全読み) を行います。")
    print("----------------------------------------")
    time.sleep(1.5)

    board = board_start()
    print_board(board)
    
    turn_count = 1
    
    while True:
        # UIから呼ぶときは get_valid_moves (内部でNumbaを呼ぶ)
        b_moves = get_valid_moves(board, BLACK)
        w_moves = get_valid_moves(board, WHITE)
        if not board_check(board) and len(b_moves) == 0 and len(w_moves) == 0:
            break

        # --- 黒の手番 ---
        current_player = BLACK
        current_type = player_black_type
        moves = b_moves
        
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
                        for i in range(len(moves)):
                            if moves[i][0] == pos[0] and moves[i][1] == pos[1]:
                                valid = True
                                break
                    if valid:
                        flip_stones(board, pos[0], pos[1], current_player)
                        print(f">> 黒は {user_input} に置きました。")
                        break
                    else:
                        print("そこには置けません。")
            else:
                start_time = time.time()
                
                result = solve_bitboard_wrapper(board, current_player, d_max, t_lim, 17)
                end_time = time.time()
                
                best_move, is_exact, score, reached_depth = result
                
                if best_move[0] != -1:
                    r, c = best_move[0], best_move[1]
                    move_str = numeric_to_algebraic(r, c)
                    msg = f">> 黒(AI)は {move_str} に置きました。"
                    if is_exact:
                        msg += f" 【必勝読み: 差 {score}】"
                    else:
                        msg += f" (Depth {reached_depth}, 評価値 {score}, 時間: {end_time - start_time:.2f}秒)"
                    print(msg)
                    flip_stones(board, r, c, current_player)
                else:
                    print("Error: No move found")
                    break

        print_board(board)
        if not board_check(board): break

        # --- 白の手番 ---
        current_player = WHITE
        current_type = player_white_type
        moves = w_moves
        
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
                        for i in range(len(moves)):
                            if moves[i][0] == pos[0] and moves[i][1] == pos[1]:
                                valid = True
                                break
                    if valid:
                        flip_stones(board, pos[0], pos[1], current_player)
                        print(f">> 白は {user_input} に置きました。")
                        break
                    else:
                        print("そこには置けません。")
            else:
                start_time = time.time()
                result = solve_bitboard_wrapper(board, current_player, d_max, t_lim, 17)
                end_time = time.time()
                
                best_move, is_exact, score, reached_depth = result
                
                if best_move[0] != -1:
                    r, c = best_move[0], best_move[1]
                    move_str = numeric_to_algebraic(r, c)
                    msg = f">> 白(AI)は {move_str} に置きました。"
                    if is_exact:
                        msg += f" 【必勝読み: 差 {score}】"
                    else:
                        msg += f" (Depth {reached_depth}, 評価値 {score}, 時間: {end_time - start_time:.2f}秒)"
                    print(msg)
                    flip_stones(board, r, c, current_player)
                else:
                    print("Error: No move found")
                    break

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