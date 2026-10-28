import json
import random
import copy
import numpy as np
import time
import numba # Numbaをインポート
from tqdm import tqdm # tqdmもインポート

# --- othello_with_ml_ai.py から AIロジックを全て移植 ---

# --- ステップ1: 学習済みモデル(v1)の読み込み ---
try:
    # ★ v1モデル (原型) を読み込む
    G_PATTERN_TABLE = np.load('pattern_weights.npy') 
    print("AI(v1)の評価テーブル (pattern_weights.npy) を読み込みました。")
except FileNotFoundError:
    print("エラー: 'pattern_weights.npy' (v1モデル) が見つかりません。")
    print("       AIが動作しません。先に learn.py を実行して v1モデルを作成してください。")
    exit()
except Exception as e:
    print(f"エラー: pattern_weights.npy の読み込みに失敗しました。 {e}")
    exit()

# --- ステップ2: パターン定義 (learn.py / othello_with_ml_ai.py と共通) ---
BASE_3_POWERS = np.array([3**i for i in range(10)], dtype=np.float64)

NUM_HOR_PATTERNS = 3**8; NUM_HOR_LINES = 8
NUM_VER_PATTERNS = 3**8; NUM_VER_LINES = 8
NUM_DIAG8_PATTERNS = 3**8; NUM_DIAG8_LINES = 2
NUM_DIAG7_PATTERNS = 3**7; NUM_DIAG7_LINES = 4
NUM_DIAG6_PATTERNS = 3**6; NUM_DIAG6_LINES = 4
NUM_DIAG5_PATTERNS = 3**5; NUM_DIAG5_LINES = 4
NUM_DIAG4_PATTERNS = 3**4; NUM_DIAG4_LINES = 4
NUM_PAT_EDGE_X_PATTERNS = 3**10; NUM_PAT_EDGE_X_LINES = 4
NUM_PAT_CORNER_TRI_PATTERNS = 3**10; NUM_PAT_CORNER_TRI_LINES = 4

TOTAL_FEATURES = (NUM_HOR_PATTERNS * NUM_HOR_LINES) + \
                 (NUM_VER_PATTERNS * NUM_VER_LINES) + \
                 (NUM_DIAG8_PATTERNS * NUM_DIAG8_LINES) + \
                 (NUM_DIAG7_PATTERNS * NUM_DIAG7_LINES) + \
                 (NUM_DIAG6_PATTERNS * NUM_DIAG6_LINES) + \
                 (NUM_DIAG5_PATTERNS * NUM_DIAG5_LINES) + \
                 (NUM_DIAG4_PATTERNS * NUM_DIAG4_LINES) + \
                 (NUM_PAT_EDGE_X_PATTERNS * NUM_PAT_EDGE_X_LINES) + \
                 (NUM_PAT_CORNER_TRI_PATTERNS * NUM_PAT_CORNER_TRI_LINES) # 603450

if G_PATTERN_TABLE is not None and len(G_PATTERN_TABLE) != TOTAL_FEATURES:
    print("エラー: 読み込んだ pattern_weights_v1.npy のサイズが、")
    print(f"       プログラムの定義 ({TOTAL_FEATURES}) と一致しません。")
    print(f"       (読み込んだサイズ: {len(G_PATTERN_TABLE)})")
    print("       v1モデルと、このプログラムのパターン定義が不一致です。")
    exit()

# --- Othelloの基本関数 (ほぼNumba化) ---

def board_start():
    board = [[0 for _ in range(8)] for _ in range(8)]
    board[3][3] = 2; board[3][4] = 1; board[4][3] = 1; board[4][4] = 2
    return board

@numba.jit(nopython=True)
def count_empty_squares_numba(board_np):
    empty_count = 0
    for i in range(8):
        for j in range(8):
            if board_np[i, j] == 0: empty_count += 1
    return empty_count

# ★★★ Numba 高速化 (can_place) ★★★
# set() と list.append() を排除し、固定長Numpy配列のみを使用
@numba.jit(nopython=True)
def can_place_numba(board_np, player):
    """ Numba化された can_place (超高速版) """
    
    # 戻り値: (moves_array, move_count)
    # moves_array は (60, 2) のNumpy配列 (row, col)
    # オセロの最大合法手は60手未満
    moves_array = np.zeros((60, 2), dtype=np.int64)
    move_count = 0
    
    opponent = 3 - player
    directions = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    
    for i in range(8): # row
        for j in range(8): # col
            if board_np[i, j] != 0: continue
            
            # このマス(i, j)が合法手か
            is_valid_move_for_this_square = False 
            
            for dr, dc in directions:
                x, y = i + dr, j + dc
                if not (0 <= x < 8 and 0 <= y < 8 and board_np[x, y] == opponent):
                    continue
                
                found_opponent = False
                while 0 <= x < 8 and 0 <= y < 8:
                    if board_np[x, y] == 0: break
                    if board_np[x, y] == opponent:
                        found_opponent = True
                        x += dr; y += dc
                    elif board_np[x, y] == player:
                        if found_opponent:
                            is_valid_move_for_this_square = True
                        break
                    else: break
                
                if is_valid_move_for_this_square:
                    # 合法手が見つかった
                    if move_count < 60: # バッファオーバーラン防止
                        moves_array[move_count, 0] = i
                        moves_array[move_count, 1] = j
                        move_count += 1
                    
                    # このマス(i,j)のチェックは終了 (次のマスへ)
                    break 
                    
    return moves_array, move_count

# ★★★ Numba 高速化 (flip_stones) ★★★
# list of tuples を排除し、固定長Numpy配列のみを使用
@numba.jit(nopython=True)
def flip_stones_numba(board_np, row, col, player):
    """ Numba化された flip_stones (超高速版) """
    opponent = 3 - player
    directions = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    board_np[row, col] = player
    
    # ひっくり返す石を一時的に保持 (固定長配列)
    stones_to_flip_np = np.zeros((8, 2), dtype=np.int64)
    flip_count = 0
    
    for dr, dc in directions:
        x, y = row + dr, col + dc
        flip_count = 0 # リセット
        
        while 0 <= x < 8 and 0 <= y < 8:
            if board_np[x, y] == 0:
                flip_count = 0 # リセット
                break
            if board_np[x, y] == opponent:
                if flip_count < 8: # 配列オーバーラン防止
                    stones_to_flip_np[flip_count, 0] = x
                    stones_to_flip_np[flip_count, 1] = y
                    flip_count += 1
            elif board_np[x, y] == player:
                # flip_count > 0 ならひっくり返す
                for i in range(flip_count):
                    flip_x = stones_to_flip_np[i, 0]
                    flip_y = stones_to_flip_np[i, 1]
                    board_np[flip_x, flip_y] = player
                break # この方向は終了
            x += dr; y += dc
            
    # board_np は参照渡しなので、呼び出し元で変更が反映される

def winner(board_list):
    """ 勝敗判定 (Pythonリストを受け取る) """
    black_count = 0
    white_count = 0
    for r in range(8):
        for c in range(8):
            if board_list[r][c] == 2:
                black_count += 1
            elif board_list[r][c] == 1:
                white_count += 1

    if black_count > white_count: return 1.0
    elif white_count > black_count: return -1.0
    else: return 0.0

# --- ★AI評価関数 (A): 機械学習AI (Numba) ---
@numba.jit(nopython=True) 
def calculate_pattern_score(board_np, pattern_table, base_powers):
    """
    Numbaで高速化される評価計算の「中身」
    """
    total_score = 0.0
    base_offset = 0
    
    # --- 1. 横パターンのスコア (8ライン) ---
    powers_8 = base_powers[:8]
    for r in range(NUM_HOR_LINES):
        # ★ Numba dot エラー修正
        line = board_np[r, :].astype(np.float64) 
        index_float = np.dot(line, powers_8)
        index_int = np.int64(np.round(index_float))
        total_score += pattern_table[index_int + base_offset]
        base_offset += NUM_HOR_PATTERNS

    # --- 2. 縦パターンのスコア (8ライン) ---
    for c in range(NUM_VER_LINES):
        # ★ Numba dot エラー修正
        line = board_np[:, c].astype(np.float64)
        index_float = np.dot(line, powers_8)
        index_int = np.int64(np.round(index_float))
        total_score += pattern_table[index_int + base_offset]
        base_offset += NUM_VER_PATTERNS
    
    # --- 3. 斜めパターンのスコア ---
    # 3-1. DIAG8 (長さ8 x 2ライン)
    powers = powers_8
    L = 8
    line = np.zeros(L, dtype=np.float64)
    # (k=0)
    for i in range(L): line[i] = board_np[i, i]
    index_float = np.dot(line, powers)
    total_score += pattern_table[np.int64(np.round(index_float)) + base_offset]
    base_offset += NUM_DIAG8_PATTERNS
    # (k=0, flipped)
    for i in range(L): line[i] = board_np[i, 7-i]
    index_float = np.dot(line, powers)
    total_score += pattern_table[np.int64(np.round(index_float)) + base_offset]
    base_offset += NUM_DIAG8_PATTERNS

    # 3-2. DIAG7 (長さ7 x 4ライン)
    powers = base_powers[:7]
    L = 7
    line = np.zeros(L, dtype=np.float64)
    # (k=1)
    for i in range(L): line[i] = board_np[i, i+1]
    index_float = np.dot(line, powers)
    total_score += pattern_table[np.int64(np.round(index_float)) + base_offset]
    base_offset += NUM_DIAG7_PATTERNS
    # (k=1, flipped)
    for i in range(L): line[i] = board_np[i, 7-(i+1)]
    index_float = np.dot(line, powers)
    total_score += pattern_table[np.int64(np.round(index_float)) + base_offset]
    base_offset += NUM_DIAG7_PATTERNS
    # (k=-1)
    for i in range(L): line[i] = board_np[i+1, i]
    index_float = np.dot(line, powers)
    total_score += pattern_table[np.int64(np.round(index_float)) + base_offset]
    base_offset += NUM_DIAG7_PATTERNS
    # (k=-1, flipped)
    for i in range(L): line[i] = board_np[i+1, 7-i]
    index_float = np.dot(line, powers)
    total_score += pattern_table[np.int64(np.round(index_float)) + base_offset]
    base_offset += NUM_DIAG7_PATTERNS

    # 3-3. DIAG6 (長さ6 x 4ライン)
    powers = base_powers[:6]
    L = 6
    line = np.zeros(L, dtype=np.float64)
    # (k=2)
    for i in range(L): line[i] = board_np[i, i+2]
    index_float = np.dot(line, powers)
    total_score += pattern_table[np.int64(np.round(index_float)) + base_offset]
    base_offset += NUM_DIAG6_PATTERNS
    # (k=2, flipped)
    for i in range(L): line[i] = board_np[i, 7-(i+2)]
    index_float = np.dot(line, powers)
    total_score += pattern_table[np.int64(np.round(index_float)) + base_offset]
    base_offset += NUM_DIAG6_PATTERNS
    # (k=-2)
    for i in range(L): line[i] = board_np[i+2, i]
    index_float = np.dot(line, powers)
    total_score += pattern_table[np.int64(np.round(index_float)) + base_offset]
    base_offset += NUM_DIAG6_PATTERNS
    # (k=-2, flipped)
    for i in range(L): line[i] = board_np[i+2, 7-i]
    index_float = np.dot(line, powers)
    total_score += pattern_table[np.int64(np.round(index_float)) + base_offset]
    base_offset += NUM_DIAG6_PATTERNS

    # 3-4. DIAG5 (長さ5 x 4ライン)
    powers = base_powers[:5]
    L = 5
    line = np.zeros(L, dtype=np.float64)
    # (k=3)
    for i in range(L): line[i] = board_np[i, i+3]
    index_float = np.dot(line, powers)
    total_score += pattern_table[np.int64(np.round(index_float)) + base_offset]
    base_offset += NUM_DIAG5_PATTERNS
    # (k=3, flipped)
    for i in range(L): line[i] = board_np[i, 7-(i+3)]
    index_float = np.dot(line, powers)
    total_score += pattern_table[np.int64(np.round(index_float)) + base_offset]
    base_offset += NUM_DIAG5_PATTERNS
    # (k=-3)
    for i in range(L): line[i] = board_np[i+3, i]
    index_float = np.dot(line, powers)
    total_score += pattern_table[np.int64(np.round(index_float)) + base_offset]
    base_offset += NUM_DIAG5_PATTERNS
    # (k=-3, flipped)
    for i in range(L): line[i] = board_np[i+3, 7-i]
    index_float = np.dot(line, powers)
    total_score += pattern_table[np.int64(np.round(index_float)) + base_offset]
    base_offset += NUM_DIAG5_PATTERNS

    # 3-5. DIAG4 (長さ4 x 4ライン)
    powers = base_powers[:4]
    L = 4
    line = np.zeros(L, dtype=np.float64)
    # (k=4)
    for i in range(L): line[i] = board_np[i, i+4]
    index_float = np.dot(line, powers)
    total_score += pattern_table[np.int64(np.round(index_float)) + base_offset]
    base_offset += NUM_DIAG4_PATTERNS
    # (k=4, flipped)
    for i in range(L): line[i] = board_np[i, 7-(i+4)]
    index_float = np.dot(line, powers)
    total_score += pattern_table[np.int64(np.round(index_float)) + base_offset]
    base_offset += NUM_DIAG4_PATTERNS
    # (k=-4)
    for i in range(L): line[i] = board_np[i+4, i]
    index_float = np.dot(line, powers)
    total_score += pattern_table[np.int64(np.round(index_float)) + base_offset]
    base_offset += NUM_DIAG4_PATTERNS
    # (k=-4, flipped)
    for i in range(L): line[i] = board_np[i+4, 7-i]
    index_float = np.dot(line, powers)
    total_score += pattern_table[np.int64(np.round(index_float)) + base_offset]
    base_offset += NUM_DIAG4_PATTERNS

    
    # --- 4. EDGE_X パターン (4辺) ---
    powers = base_powers[:10] # 10マス
    line = np.zeros(10, dtype=np.float64)
    
    # Top (rot 0)
    for c in range(8): line[c] = board_np[0, c]
    line[8] = board_np[1, 1]; line[9] = board_np[1, 6]
    index_float = np.dot(line, powers)
    total_score += pattern_table[np.int64(np.round(index_float)) + base_offset]
    base_offset += NUM_PAT_EDGE_X_PATTERNS
    
    # Right (rot 90)
    for r in range(8): line[r] = board_np[r, 7]
    line[8] = board_np[1, 6]; line[9] = board_np[6, 6]
    index_float = np.dot(line, powers)
    total_score += pattern_table[np.int64(np.round(index_float)) + base_offset]
    base_offset += NUM_PAT_EDGE_X_PATTERNS
    
    # Bottom (rot 180)
    for c in range(8): line[c] = board_np[7, 7-c]
    line[8] = board_np[6, 6]; line[9] = board_np[6, 1]
    index_float = np.dot(line, powers)
    total_score += pattern_table[np.int64(np.round(index_float)) + base_offset]
    base_offset += NUM_PAT_EDGE_X_PATTERNS
    
    # Left (rot 270)
    for r in range(8): line[r] = board_np[7-r, 0]
    line[8] = board_np[6, 1]; line[9] = board_np[1, 1]
    index_float = np.dot(line, powers)
    total_score += pattern_table[np.int64(np.round(index_float)) + base_offset]
    base_offset += NUM_PAT_EDGE_X_PATTERNS
    
    
    # --- 5. CORNER_TRI パターン (4コーナー) ---
    powers = base_powers[:10] # 10マス
    line = np.zeros(10, dtype=np.float64)

    # Top-Left (rot 0)
    line[0] = board_np[0, 0]; line[1] = board_np[0, 1]; line[2] = board_np[0, 2]; line[3] = board_np[0, 3]
    line[4] = board_np[1, 0]; line[5] = board_np[1, 1]; line[6] = board_np[1, 2]
    line[7] = board_np[2, 0]; line[8] = board_np[2, 1]; line[9] = board_np[3, 0]
    index_float = np.dot(line, powers)
    total_score += pattern_table[np.int64(np.round(index_float)) + base_offset]
    base_offset += NUM_PAT_CORNER_TRI_PATTERNS
    
    # Top-Right (rot 90)
    line[0] = board_np[0, 7]; line[1] = board_np[0, 6]; line[2] = board_np[0, 5]; line[3] = board_np[0, 4]
    line[4] = board_np[1, 7]; line[5] = board_np[1, 6]; line[6] = board_np[1, 5]
    line[7] = board_np[2, 7]; line[8] = board_np[2, 6]; line[9] = board_np[3, 7]
    index_float = np.dot(line, powers)
    total_score += pattern_table[np.int64(np.round(index_float)) + base_offset]
    base_offset += NUM_PAT_CORNER_TRI_PATTERNS

    # Bottom-Right (rot 180)
    line[0] = board_np[7, 7]; line[1] = board_np[7, 6]; line[2] = board_np[7, 5]; line[3] = board_np[7, 4]
    line[4] = board_np[6, 7]; line[5] = board_np[6, 6]; line[6] = board_np[6, 5]
    line[7] = board_np[5, 7]; line[8] = board_np[5, 6]; line[9] = board_np[4, 7]
    index_float = np.dot(line, powers)
    total_score += pattern_table[np.int64(np.round(index_float)) + base_offset]
    base_offset += NUM_PAT_CORNER_TRI_PATTERNS
    
    # Bottom-Left (rot 270)
    line[0] = board_np[7, 0]; line[1] = board_np[6, 0]; line[2] = board_np[5, 0]; line[3] = board_np[4, 0]
    line[4] = board_np[7, 1]; line[5] = board_np[6, 1]; line[6] = board_np[5, 1]
    line[7] = board_np[7, 2]; line[8] = board_np[6, 2]; line[9] = board_np[7, 3]
    index_float = np.dot(line, powers)
    total_score += pattern_table[np.int64(np.round(index_float)) + base_offset]
    base_offset += NUM_PAT_CORNER_TRI_PATTERNS

    return total_score

@numba.jit(nopython=True)
def evaluate_numba(board_np, player, pattern_table, base_powers):
    """ Numba化された評価関数の「入り口」 """
    # board_np は int64 で渡ってくる。
    # calculate_pattern_score を呼ぶ前に float64 に変換する
    board_float_np = board_np.astype(np.float64)
    
    score = calculate_pattern_score(board_float_np, pattern_table, base_powers)
    if player == 2: return score
    else: return -score

# --- ★AI探索アルゴリズム (Numba版) ---
@numba.jit(nopython=True)
def negascout_numba(board_np, player, depth, alpha, beta, pattern_table, base_powers):
    """
    NegaScout法（完全Numba版）
    board_np は int64 を期待
    """
    opponent = 3 - player
    
    # 1. ベースケース: 深さ0
    if depth == 0:
        return evaluate_numba(board_np, player, pattern_table, base_powers)
    
    # 2. 合法手の取得 (Numba版・高速)
    legal_moves_array, move_count = can_place_numba(board_np, player)

    # 3. パスの処理
    if move_count == 0:
        # 相手もパスか (Numba版)
        if can_place_numba(board_np, opponent)[1] == 0: # move_count == 0
            return evaluate_numba(board_np, player, pattern_table, base_powers)
        # 自分だけパス
        return -negascout_numba(board_np, opponent, depth - 1, -beta, -alpha, pattern_table, base_powers)
    
    # 4. NegaScout探索
    is_first_move = True
    for k in range(move_count): # ★ 高速化: Numbaフレンドリーなループ
        row = legal_moves_array[k, 0]
        col = legal_moves_array[k, 1]
        
        # ★ Numba 高速化: 高速なNumpyコピーを使う
        temp_board_np = board_np.copy() # int64 のコピー
        
        # ★ Numba 高速化: Numba版 flip_stones を使う
        flip_stones_numba(temp_board_np, row, col, player)
        
        if is_first_move:
            score = -negascout_numba(temp_board_np, opponent, depth - 1, -beta, -alpha, pattern_table, base_powers)
            is_first_move = False
        else:
            score = -negascout_numba(temp_board_np, opponent, depth - 1, -alpha - 1, -alpha, pattern_table, base_powers)
            if alpha < score < beta:
                score = -negascout_numba(temp_board_np, opponent, depth - 1, -beta, -score, pattern_table, base_powers)
        
        alpha = max(alpha, score)
        if alpha >= beta:
            return alpha
    return alpha

# --- AIの「Python側」ラッパー関数 ---
# (Numba化されたAIエンジンを呼び出す)
def computer_scout(board_list, player, eval_func, depth_ply=7):
    """
    NegaScout 汎用 (Python側ラッパー)
    """
    opponent = 3 - player
    best_score = -np.inf; best_move = None
    
    # 1. Pythonリスト -> Numba用Numpy配列 (int64) に変換
    board_np = np.array(board_list, dtype=np.int64)
    
    # 2. 合法手の取得 (Numba版)
    legal_moves_array, move_count = can_place_numba(board_np, player)
    if move_count == 0: return None
    
    alpha = -np.inf; beta = np.inf
    is_first_move = True
    
    # ★ TODO: legal_moves を並び替える (Move Ordering)
    
    for k in range(move_count): # ★ 高速化
        row = legal_moves_array[k, 0]
        col = legal_moves_array[k, 1]
        
        # 3. Numbaの世界で探索実行
        temp_board_np = board_np.copy() # int64 のコピー
        flip_stones_numba(temp_board_np, row, col, player)
        
        # 4. AIエンジン(Numba)を呼び出す
        if is_first_move:
            score = -negascout_numba(temp_board_np, opponent, depth_ply, -beta, -alpha, G_PATTERN_TABLE, BASE_3_POWERS)
            is_first_move = False
        else:
            score = -negascout_numba(temp_board_np, opponent, depth_ply, -alpha - 1, -alpha, G_PATTERN_TABLE, BASE_3_POWERS)
            if alpha < score < beta:
                score = -negascout_numba(temp_board_np, opponent, depth_ply, -beta, -score, G_PATTERN_TABLE, BASE_3_POWERS)
        
        if score > best_score:
            best_score = score
            best_move = (row, col) # ★ move を (row, col) で保存
        
        alpha = max(alpha, best_score)
        
    if best_move is None: 
        # 合法手があるのに best_move が None はありえないはずだが、
        # 万が一の保険
        idx = random.randint(0, move_count - 1)
        best_move = (legal_moves_array[idx, 0], legal_moves_array[idx, 1])

    return best_move

def computer_ai_commander(board_list, player, eval_func):
    """
    AI司令塔 (Python側)
    """
    board_np = np.array(board_list, dtype=np.int64)
    empty_squares = count_empty_squares_numba(board_np)
    
    depth_ply = 7
    if empty_squares <= 16: depth_ply = 11
    elif empty_squares <= 24: depth_ply = 9
    else: 
        if empty_squares >= 50: # 開始〜10手目
            depth_ply = 5 # 6手読み (高速化のため)
        else:
            depth_ply = 7 # 8手読み
            
    # Python側の computer_scout を呼び出す
    return computer_scout(board_list, player, eval_func, depth_ply)

# --- 自己対戦メインロジック ---

def play_game(ai_black, ai_white):
    """
    AI同士で1局対戦し、盤面の履歴と最終スコアを返す
    """
    board = board_start()
    game_history = []
    turn_player = 2
    
    # ★ Numba化 (高速化) のため、board は Numpy 配列(int64)で管理
    board_np = np.array(board, dtype=np.int64)
    
    while True:
        # 盤面をPythonリストとして保存
        game_history.append(board_np.tolist())
        
        current_ai_func = ai_black if turn_player == 2 else ai_white
        
        # ★ AIを呼び出す (AIはPythonリストの board を期待)
        board_list = board_np.tolist()
        move = current_ai_func(board_list, turn_player) # (row, col) を期待
        
        if move is not None:
            row, col = move
            if board_np[row, col] != 0:
                print(f"AIが不正な手 {move} を選びました。")
                return [], 0.0
            
            # ★ Numba版 flip_stones を使用 (board_np を直接変更)
            flip_stones_numba(board_np, row, col, turn_player)
            
        else:
            # パスの場合
            # 相手もパスか (Numba版)
            if can_place_numba(board_np, 3 - turn_player)[1] == 0: # move_count == 0
                break # 両者パス = ゲーム終了
                
        turn_player = 3 - turn_player
        
    # Pythonリストに変換してwinnerに渡す
    final_score = winner(board_np.tolist())
    return game_history, final_score

# --- ★AIラッパー (self_play.py専用) ---

def ai_ml_player(board, player):
    """
    v1モデル (pattern_weights.npy) を使うAI
    """
    # 評価関数 `evaluate` (Python版) を渡す
    # (computer_scout は内部でNumba版 negascout を呼ぶので eval_func は実際には使われない)
    return computer_ai_commander(board, player, None) # eval_funcはNoneでOK

def random_ai(board, player):
    """ ランダムAI (Numba版 can_place を使う) """
    board_np = np.array(board, dtype=np.int64)
    
    # ★ 修正: Numba版 can_place を正しく呼び出す
    legal_moves_array, move_count = can_place_numba(board_np, player)
    
    if move_count == 0:
        return None
        
    # ★ 修正: Numba配列からランダムに選ぶ
    idx = random.randint(0, move_count - 1)
    return (legal_moves_array[idx, 0], legal_moves_array[idx, 1])


# --- ★ MAIN ★ ---
if __name__ == "__main__":
    
    # ★★★ フェーズ2の設定 ★★★
    NUM_GAMES = 50000 # 5万試合
    OUTPUT_FILE = "training_data_v2.jsonl" # ★ v2 (jsonl形式)
    
    # ★自己対戦に使うAIを選択★
    # ----------------------------------------------------
    # フェーズ1 (v1モデル作成用)
    # AI_BLACK = random_ai 
    # AI_WHITE = random_ai
    
    # ★ フェーズ2: v1モデル AI 同士を対戦させる
    AI_BLACK = ai_ml_player 
    AI_WHITE = ai_ml_player
    # ----------------------------------------------------
    
    print(f"--- AI自己対戦 (フェーズ2) を開始します ---")
    print(f"使用AI (黒): {AI_BLACK.__name__} (v1モデル)")
    print(f"使用AI (白): {AI_WHITE.__name__} (v1モデル)")
    print(f"対局回数: {NUM_GAMES} 局")
    print(f"出力ファイル: {OUTPUT_FILE}")
    print("-" * 30)
    
    # Numbaのコンパイルを先に行う
    print("Numbaコンパイル中... (数分かかる場合があります)")
    start_compile_time = time.time()
    # (Pythonラッパー -> Numbaエンジン の順でコンパイル)
    try:
        _ = ai_ml_player(board_start(), 2) 
    except Exception as e:
        print("--- Numba コンパイル中にエラーが発生しました ---")
        print(e)
        print("Numba の型推論に失敗した可能性があります。")
        exit()
        
    print(f"Numbaコンパイル完了。 (所要時間: {time.time() - start_compile_time:.2f} 秒)")
    
    start_time = time.time()

    # 既存のv2ファイルがあれば削除
    import os
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        print(f"既存の {OUTPUT_FILE} を削除しました。")

    # tqdm (プログレスバー) のセットアップ
    with tqdm(total=NUM_GAMES, desc="自己対戦中") as pbar:
        for i in range(NUM_GAMES):
            game_boards, final_score = play_game(AI_BLACK, AI_WHITE)
            
            if not game_boards:
                print(f"ゲーム {i+1}/{NUM_GAMES} はエラーによりスキップされました。")
                continue

            # ★ 1行1JSON (jsonl) 形式で追記
            try:
                with open(OUTPUT_FILE, 'a') as f: # 'a' = 追記モード
                    for board_state in game_boards:
                        line_data = json.dumps({
                            'board': board_state,
                            'score': final_score
                        })
                        f.write(line_data + '\n')
            except Exception as e:
                print(f"エラー: {OUTPUT_FILE} への書き込みに失敗しました。 {e}")
                break
            
            pbar.update(1) # プログレスバーを1進める
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                if elapsed > 0:
                    games_per_sec = (i + 1) / elapsed
                    pbar.set_description(f"自己対戦中 ({games_per_sec:.2f} 局/秒)")
                
    end_time = time.time()
    
    print("-" * 30)
    print(f"全 {NUM_GAMES} 局の自己対戦が完了しました。")
    print(f"所要時間: {end_time - start_time:.2f} 秒")
    if (end_time - start_time) > 0:
        print(f"平均速度: {NUM_GAMES / (end_time - start_time):.2f} 局/秒")
    print(f"データを {OUTPUT_FILE} に保存しました。")