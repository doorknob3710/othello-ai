#リバーシゲーム
#8x8の盤面を作成する
#2を黒,1を白とする
#初期配置は中央に黒2つ,白2つ
#初めは8*8の盤面に0を入れる
import IPython
from IPython.display import clear_output
import random
import numpy as np
import numba # ★ Numbaライブラリをインポート
import time # ★ AI対戦の表示用にインポート

# --- ステップ3: 学習済みモデルの読み込み ---
try:
    # learn.py が出力した評価テーブル（重み）を読み込む
    G_PATTERN_TABLE = np.load('pattern_weights.npy')
    print("AI: パターン評価テーブル (pattern_weights.npy) を読み込みました。")
except FileNotFoundError:
    print("警告: pattern_weights.npy が見つかりません。")
    print("警告: AIは動作しません。先に learn.py を実行してください。")
    G_PATTERN_TABLE = None
except Exception as e:
    print(f"エラー: pattern_weights.npy の読み込みに失敗しました。 {e}")
    G_PATTERN_TABLE = None

# --- ステップ3: パターン定義 (learn.py と共通) ---
# Numba(othello.py)のnp.dotとの互換性のため float64 にする
BASE_3_POWERS = np.array([3**i for i in range(10)], dtype=np.float64) # ★ 10に変更

# 1. 横8マスのパターン (8ライン)
NUM_HOR_PATTERNS = 3**8 # 6561
NUM_HOR_LINES = 8
# 2. 縦8マスのパターン (8ライン)
NUM_VER_PATTERNS = 3**8 # 6561
NUM_VER_LINES = 8

# 3. ★新規★ 斜めパターン (DIAG)
# (長さ8 x 2ライン)
NUM_DIAG8_PATTERNS = 3**8 # 6561
NUM_DIAG8_LINES = 2
# (長さ7 x 4ライン)
NUM_DIAG7_PATTERNS = 3**7 # 2187
NUM_DIAG7_LINES = 4
# (長さ6 x 4ライン)
NUM_DIAG6_PATTERNS = 3**6 # 729
NUM_DIAG6_LINES = 4
# (長さ5 x 4ライン)
NUM_DIAG5_PATTERNS = 3**5 # 243
NUM_DIAG5_LINES = 4
# (長さ4 x 4ライン)
NUM_DIAG4_PATTERNS = 3**4 # 81
NUM_DIAG4_LINES = 4

# 4. ★新規★ EDGE_X パターン (辺 + X打ち) (10マス)
# (長さ10 x 4辺)
NUM_PAT_EDGE_X_PATTERNS = 3**10 # 59049
NUM_PAT_EDGE_X_LINES = 4

# 5. ★新規★ CORNER_TRI パターン (角の三角領域) (10マス)
# (長さ10 x 4コーナー)
NUM_PAT_CORNER_TRI_PATTERNS = 3**10 # 59049
NUM_PAT_CORNER_TRI_LINES = 4


# 特徴量（評価テーブル）の総数
TOTAL_FEATURES = (NUM_HOR_PATTERNS * NUM_HOR_LINES) + \
                 (NUM_VER_PATTERNS * NUM_VER_LINES) + \
                 (NUM_DIAG8_PATTERNS * NUM_DIAG8_LINES) + \
                 (NUM_DIAG7_PATTERNS * NUM_DIAG7_LINES) + \
                 (NUM_DIAG6_PATTERNS * NUM_DIAG6_LINES) + \
                 (NUM_DIAG5_PATTERNS * NUM_DIAG5_LINES) + \
                 (NUM_DIAG4_PATTERNS * NUM_DIAG4_LINES) + \
                 (NUM_PAT_EDGE_X_PATTERNS * NUM_PAT_EDGE_X_LINES) + \
                 (NUM_PAT_CORNER_TRI_PATTERNS * NUM_PAT_CORNER_TRI_LINES) # ★ 612096 になるはず
                 

if G_PATTERN_TABLE is not None and len(G_PATTERN_TABLE) != TOTAL_FEATURES:
    print("警告: 読み込んだ pattern_weights.npy のサイズが、")
    print(f"       プログラムの定義 ({TOTAL_FEATURES}) と一致しません。")
    print(f"       (読み込んだサイズ: {len(G_PATTERN_TABLE)})")
    print("       古いモデルを読み込んでいる可能性があります。learn.py を再実行してください。")
    G_PATTERN_TABLE = None # エラーを防ぐために無効化

# -------------------------------------------------

def print_board(board):
    clear_output()
    print("a　b　c　d　e　f　g　h") # 列の表示 (人間用)
    for i in range(8):
        for j in range(8):
            if board[i][j] == 0:#緑
                print("🟩", end=" ")
            elif board[i][j] == 1:#白
                print("⚪️", end=" ")
            elif board[i][j] == 2:#黒
                print("⚫️", end=" ")
        print(i, end=" ") # 行の表示 (人間用)
        print()

def board_start():
    board = [[0 for _ in range(8)] for _ in range(8)]
    board[3][3] = 2
    board[3][4] = 1
    board[4][3] = 1
    board[4][4] = 2
    return board

def count_empty_squares(board):
    """
    ★新規★
    盤面の空きマスの数を数える (終盤判定用)
    """
    empty_count = 0
    for i in range(8):
        for j in range(8):
            if board[i][j] == 0:
                empty_count += 1
    return empty_count


def board_check(board):
    for i in range(8):
        for j in range(8):
            if board[i][j] == 0:
                return True
    return False

def can_place(board, player):
    """
    石を置ける場所のリストを (row, col) の数値タプルで返す
    """
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),          (0, 1),
                  (1, -1), (1, 0), (1, 1)]
    can_place_positions = []
    opponent = 3 - player # 1なら2, 2なら1

    for i in range(8): # row
        for j in range(8): # col
            if board[i][j] != 0:
                continue # 既に石がある

            found_valid_direction = False
            for direction in directions:
                x, y = i + direction[0], j + direction[1]
                
                # 1. 隣が相手の石か
                if not (0 <= x < 8 and 0 <= y < 8 and board[x][y] == opponent):
                    continue
                
                # 2. さらにその先に自分の石があるか
                found_opponent = False
                while 0 <= x < 8 and 0 <= y < 8:
                    if board[x][y] == 0:
                        break # 空白
                    if board[x][y] == opponent:
                        found_opponent = True
                        x += direction[0]
                        y += direction[1]
                    elif board[x][y] == player:
                        if found_opponent:
                            found_valid_direction = True
                        break # 自分の石
                    else:
                        break # (ありえないはず)
                
                if found_valid_direction:
                    # (i, j) = (row, col)
                    can_place_positions.append((i, j))
                    break # このマスは置けるとわかったので、次のマスへ
                    
    # 重複を削除 (setにしてからlistに戻す)
    return list(set(can_place_positions))

def flip_stones(board, row, col, player):
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),          (0, 1),
                  (1, -1), (1, 0), (1, 1)]
    opponent = 3 - player
    
    board[row][col] = player # まず石を置く

    for direction in directions:
        x, y = row + direction[0], col + direction[1]
        stones_to_flip = []
        while 0 <= x < 8 and 0 <= y < 8:
            if board[x][y] == 0:
                break # 空白
            if board[x][y] == opponent:
                stones_to_flip.append((x, y))
            elif board[x][y] == player:
                # 自分の石に到達したら、挟んだ石をひっくり返す
                for flip_x, flip_y in stones_to_flip:
                    board[flip_x][flip_y] = player
                break # この方向の探索は終了
            x += direction[0]
            y += direction[1]


# --- ★AI評価関数 (A): 機械学習AI (Numba) ---

@numba.jit(nopython=True) # Numbaのおまじない (C++並みに高速化)
def calculate_pattern_score(board_np, pattern_table, base_powers):
    """
    Numbaで高速化される評価計算の「中身」
    引数: board_np (Numpy配列), pattern_table (学習済みの重み)
    """
    total_score = 0.0
    base_offset = 0
    
    # --- 1. 横パターンのスコア (8ライン) ---
    powers_8 = base_powers[:8]
    for r in range(NUM_HOR_LINES):
        line = board_np[r, :] # 横1ライン (8マス)
        index_float = np.dot(line, powers_8)
        index_int = np.int64(np.round(index_float))
        total_score += pattern_table[index_int + base_offset]
        base_offset += NUM_HOR_PATTERNS

    # --- 2. 縦パターンのスコア (8ライン) ---
    for c in range(NUM_VER_LINES):
        line = board_np[:, c] # 縦1ライン (8マス)
        index_float = np.dot(line, powers_8)
        index_int = np.int64(np.round(index_float))
        total_score += pattern_table[index_int + base_offset]
        base_offset += NUM_VER_PATTERNS
    
    # --- 3. ★新規★ 斜めパターンのスコア ---
    # Numbaは np.diag, np.fliplr が使えないため、
    # learn.py と同じパターンを "手動" で抽出する
    
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

    
    # --- 4. ★新規★ EDGE_X パターン (4辺) ---
    powers = base_powers[:10] # 10マス
    line = np.zeros(10, dtype=np.float64)
    
    # Top (rot 0)
    # (0,0)...(0,7)
    for c in range(8): line[c] = board_np[0, c]
    line[8] = board_np[1, 1]
    line[9] = board_np[1, 6]
    index_float = np.dot(line, powers)
    total_score += pattern_table[np.int64(np.round(index_float)) + base_offset]
    base_offset += NUM_PAT_EDGE_X_PATTERNS
    
    # Right (rot 90)
    # (0,7)...(7,7)
    for r in range(8): line[r] = board_np[r, 7]
    line[8] = board_np[1, 6]
    line[9] = board_np[6, 6]
    index_float = np.dot(line, powers)
    total_score += pattern_table[np.int64(np.round(index_float)) + base_offset]
    base_offset += NUM_PAT_EDGE_X_PATTERNS
    
    # Bottom (rot 180)
    # (7,7)...(7,0)
    for c in range(8): line[c] = board_np[7, 7-c]
    line[8] = board_np[6, 6]
    line[9] = board_np[6, 1]
    index_float = np.dot(line, powers)
    total_score += pattern_table[np.int64(np.round(index_float)) + base_offset]
    base_offset += NUM_PAT_EDGE_X_PATTERNS
    
    # Left (rot 270)
    # (7,0)...(0,0)
    for r in range(8): line[r] = board_np[7-r, 0]
    line[8] = board_np[6, 1]
    line[9] = board_np[1, 1]
    index_float = np.dot(line, powers)
    total_score += pattern_table[np.int64(np.round(index_float)) + base_offset]
    base_offset += NUM_PAT_EDGE_X_PATTERNS
    
    
    # --- 5. ★新規★ CORNER_TRI パターン (4コーナー) ---
    powers = base_powers[:10] # 10マス
    line = np.zeros(10, dtype=np.float64)

    # Top-Left (rot 0)
    line[0] = board_np[0, 0]; line[1] = board_np[0, 1]; line[2] = board_np[0, 2]; line[3] = board_np[0, 3]
    line[4] = board_np[1, 0]; line[5] = board_np[1, 1]; line[6] = board_np[1, 2]
    line[7] = board_np[2, 0]; line[8] = board_np[2, 1]
    line[9] = board_np[3, 0]
    index_float = np.dot(line, powers)
    total_score += pattern_table[np.int64(np.round(index_float)) + base_offset]
    base_offset += NUM_PAT_CORNER_TRI_PATTERNS
    
    # Top-Right (rot 90)
    line[0] = board_np[0, 7]; line[1] = board_np[0, 6]; line[2] = board_np[0, 5]; line[3] = board_np[0, 4]
    line[4] = board_np[1, 7]; line[5] = board_np[1, 6]; line[6] = board_np[1, 5]
    line[7] = board_np[2, 7]; line[8] = board_np[2, 6]
    line[9] = board_np[3, 7]
    index_float = np.dot(line, powers)
    total_score += pattern_table[np.int64(np.round(index_float)) + base_offset]
    base_offset += NUM_PAT_CORNER_TRI_PATTERNS

    # Bottom-Right (rot 180)
    line[0] = board_np[7, 7]; line[1] = board_np[7, 6]; line[2] = board_np[7, 5]; line[3] = board_np[7, 4]
    line[4] = board_np[6, 7]; line[5] = board_np[6, 6]; line[6] = board_np[6, 5]
    line[7] = board_np[5, 7]; line[8] = board_np[5, 6]
    line[9] = board_np[4, 7]
    index_float = np.dot(line, powers)
    total_score += pattern_table[np.int64(np.round(index_float)) + base_offset]
    base_offset += NUM_PAT_CORNER_TRI_PATTERNS
    
    # Bottom-Left (rot 270)
    line[0] = board_np[7, 0]; line[1] = board_np[6, 0]; line[2] = board_np[5, 0]; line[3] = board_np[4, 0]
    line[4] = board_np[7, 1]; line[5] = board_np[6, 1]; line[6] = board_np[5, 1]
    line[7] = board_np[7, 2]; line[8] = board_np[6, 2]
    line[9] = board_np[7, 3]
    index_float = np.dot(line, powers)
    total_score += pattern_table[np.int64(np.round(index_float)) + base_offset]
    base_offset += NUM_PAT_CORNER_TRI_PATTERNS

    return total_score


def evaluate(board, player):
    """
    NegaScoutから呼ばれる評価関数の「入り口」 (機械学習AI用)
    """
    if G_PATTERN_TABLE is None:
        print("エラー: 評価テーブルが読み込まれていないため、評価できません。")
        return 0.0 # 評価値0を返す

    # 1. Pythonのリスト(board)を、Numbaが扱える
    #    Numpy配列 (np.float64) に変換する
    board_np = np.array(board, dtype=np.float64) 

    # 2. Numbaでコンパイルされた超高速な関数を呼び出す
    #    注: G_PATTERN_TABLE と BASE_3_POWERS も引数として渡す
    score = calculate_pattern_score(board_np, G_PATTERN_TABLE, BASE_3_POWERS)
    
    # 3. プレイヤーに応じてスコアを調整
    # score は「黒にとっての評価値」
    if player == 2: # 黒番
        return score # 黒にとっての評価値をそのまま返す
    else: # 白番 (player == 1)
        return -score # 黒にとっての評価値を反転 (白にとっての評価値) にする

# --- ★AI評価関数 (B): マス評価AI (cell_weight) ---

cell_weight = np.array([
    2714,  147,   69,  -18,  -18,   69,  147, 2714,
     147, -577, -186, -153, -153, -186, -577,  147,
      69, -186, -379, -122, -122, -379, -186,   69,
     -18, -153, -122, -169, -169, -122, -153,  -18,
     -18, -153, -122, -169, -169, -122, -153,  -18,
      69, -186, -379, -122, -122, -379, -186,   69,
     147, -577, -186, -153, -153, -186, -577,  147,
    2714,  147,   69,  -18,  -18,   69,  147, 2714
])

def evaluate_cell_weight(board, player):
    """
    古いAI (マス評価) のための評価関数
    """
    score = 0
    opponent = 3 - player
    for i in range(8):
        for j in range(8):
            idx = i * 8 + j
            if board[i][j] == player:
                score += cell_weight[idx]
            elif board[i][j] == opponent:
                score -= cell_weight[idx]
    return score

# -------------------------------------------------


# --- NegaMax (AIの探索アルゴリズム) ---
# (NegaScoutを使うので、NegaMaxは必須ではないが、
# 浅い読みのAI（computer_4など）のために残しておく)
# ★変更点: eval_func を引数で受け取る
def negamax(board, player, depth, eval_func):
    """
    NegaMax法（eval_func を指定可能）
    """
    opponent = 3 - player
    
    # 1. 深さ制限に達したら、指定された評価関数で評価
    if depth == 0:
        return eval_func(board, player) # ★変更点
    
    # 2. 合法手を取得
    legal_moves = can_place(board, player)

    # 3. パスの処理
    if not legal_moves:
        # 3a. 相手もパス（ゲーム終了）
        if not can_place(board, opponent):
            return eval_func(board, player) # ★変更点
        
        # 3b. 自分だけパス
        score = -negamax(board, opponent, depth - 1, eval_func) # ★変更点
        return score
    
    # 4. 通常の探索
    best_score = -float('inf')
    for move in legal_moves: # move は (row, col)
        temp_board = [row[:] for row in board]
        row, col = move
        temp_board[row][col] = player
        flip_stones(temp_board, row, col, player)
        
        score = -negamax(temp_board, opponent, depth - 1, eval_func) # ★変更点
        
        if score > best_score:
            best_score = score
    
    return best_score


# --- NegaScout (NegaMaxの高速版) ---
# ★変更点: eval_func を引数で受け取る
def negascout(board, player, depth, alpha, beta, eval_func):
    """
    NegaScout法（eval_func を指定可能）
    """
    opponent = 3 - player
    
    # 1. ベースケース: 深さ0
    if depth == 0:
        return eval_func(board, player) # ★変更点
    
    # 2. 合法手の取得
    legal_moves = can_place(board, player)

    # 3. パスの処理
    if not legal_moves:
        # 3a. 相手もパス（ゲーム終了）
        if not can_place(board, opponent):
            return eval_func(board, player) # ★変更点
        # 3b. 自分だけパス
        return -negascout(board, opponent, depth - 1, -beta, -alpha, eval_func) # ★変更点

    # 4. NegaScout探索
    is_first_move = True
    for move in legal_moves: # move は (row, col)
        temp_board = [row[:] for row in board]
        row, col = move
        temp_board[row][col] = player
        flip_stones(temp_board, row, col, player)

        if is_first_move:
            # 4a. 最初の1手目 (フルウィンドウ)
            score = -negascout(temp_board, opponent, depth - 1, -beta, -alpha, eval_func) # ★変更点
            is_first_move = False
        else:
            # 4b. 2手目以降 (ヌルウィンドウ)
            score = -negascout(temp_board, opponent, depth - 1, -alpha - 1, -alpha, eval_func) # ★変更点
            
            # 4c. 再探索
            if alpha < score < beta:
                score = -negascout(temp_board, opponent, depth - 1, -beta, -score, eval_func) # ★変更点

        # 5. alpha値の更新
        alpha = max(alpha, score)

        # 6. ベータカット (枝刈り)
        if alpha >= beta:
            return alpha  # 探索終了

    return alpha # 最終的に見つかった最善スコア(alpha)を返す


# --- AIの呼び出し関数 (AIの「手」を選ぶ) ---
# ★変更点: 深度(ply)を引数で受け取る 汎用的な関数
def computer_scout(board, player, eval_func, depth_ply=7):
    """
    NegaScout 汎用 (depth_ply で深さを指定)
    eval_func: evaluate または evaluate_cell_weight を渡す
    depth_ply: 探索の深さ (NegaScoutに渡す値。7なら計8手読み)
    """
    opponent = 3 - player
    best_score = -float('inf') 
    best_move = None
    legal_moves = can_place(board, player)

    if not legal_moves:
        return None 

    alpha = -float('inf')
    beta = float('inf')
    
    is_first_move = True
    # ★ TODO: ここで legal_moves を「浅い読み」などで並び替えると
    # NegaScout (アルファベータ法) はさらに高速になる
    
    for move in legal_moves:
        temp_board = [row[:] for row in board]
        row, col = move
        temp_board[row][col] = player
        flip_stones(temp_board, row, col, player)

        if is_first_move:
            score = -negascout(temp_board, opponent, depth_ply, -beta, -alpha, eval_func) # ★ depth_ply
            is_first_move = False
        else:
            score = -negascout(temp_board, opponent, depth_ply, -alpha - 1, -alpha, eval_func) # ★ depth_ply
            if alpha < score < beta:
                score = -negascout(temp_board, opponent, depth_ply, -beta, -score, eval_func) # ★ depth_ply

        if score > best_score:
            best_score = score
            best_move = move
        
        alpha = max(alpha, best_score) # alpha を更新

    if best_move is None:
        return random.choice(legal_moves) 
    return best_move


# --- ★新規★ AIの司令塔 (動的深度) ---
def computer_ai_commander(board, player, eval_func):
    """
    AIの司令塔: 盤面の状況に応じて「探索深度」を変える
    """
    
    # 1. 盤面の空きマスの数を数える
    empty_squares = count_empty_squares(board)
    
    # 2. 探索深度 (ply) を決める
    #    (NegaScoutに渡す値。NegaScoutのdepth=7 で 計8手読み)
    depth_ply = 7 # デフォルト (8手読み)

    if empty_squares <= 16:
        # 最終盤 (残り16マス): 12手読み
        depth_ply = 11 
    elif empty_squares <= 24:
        # 終盤 (残り24マス): 10手読み
        depth_ply = 9
    else:
        # 序盤・中盤 (残り25マス以上): 8手読み
        depth_ply = 7 
    
    # デバッグ用
    # print(f"DEBUG: 空きマス {empty_squares}, 探索深度 {depth_ply+1}手読み")
    
    # 3. 決定した深度で NegaScout を呼び出す
    return computer_scout(board, player, eval_func, depth_ply)


# --- 勝敗判定 ---
def winner(board):
    black_count = sum(row.count(2) for row in board)
    white_count = sum(row.count(1) for row in board)
    if black_count > white_count:
        return "黒の勝ち"
    elif white_count > black_count:
        return "白の勝ち"
    else:
        return "引き分け"

# --- ヘルパー関数 (人間用 <-> AI用) ---

def numeric_to_algebraic(row, col):
    """ (0, 0) -> 'a0' """
    try:
        col_char = chr(ord('a') + col)
        row_str = str(row)
        return col_char + row_str
    except Exception:
        return "???"

def algebraic_to_numeric(move_str):
    """ 'a0' -> (0, 0) """
    try:
        col_char = move_str[0].lower()
        row_str = move_str[1]
        
        col = ord(col_char) - ord('a')
        row = int(row_str)
        
        if 0 <= col <= 7 and 0 <= row <= 7:
            return (row, col) # (row, col) タプル
        else:
            return None # 盤面外
    except Exception:
        return None # 不正な入力


# --- メインループ ---
# (★ AI対戦モードを追加)

board = board_start()
print_board(board)
blackwhite = 2 # 2:黒(人間/AI), 1:白(AI/人間)
two_p =input("1人(AI戦), 2人(対人戦), 3(AI vs AI) を入力してください:") # ★変更点

# AIが動作可能かチェック
# ★変更点: 1P と 3P の時は pattern_weights.npy が必要
if (two_p == "1" or two_p == "3") and G_PATTERN_TABLE is None:
    print("AIの評価テーブル(pattern_weights.npy)が読み込めなかったため、")
    print("機械学習AIを使ったモードを開始できません。")
    print("learn.py を実行して pattern_weights.npy を作成してください。")
    # (ここで終了させる)
    exit()


while board_check(board) == True:
    
    # 現在の手番のプレイヤー
    current_player = blackwhite
    
    # 1. 合法手を取得 (AI内部の (row, col) 形式)
    legal_moves_numeric = can_place(board, current_player)

    # 2. パス判定
    if not legal_moves_numeric:
        player_name = "黒" if current_player == 2 else "白"
        print(f"{player_name} は置ける場所がありません。")
        
        # 相手の手番に切り替える
        blackwhite = 3 - current_player
        
        # 相手も置けるかチェック
        if not can_place(board, blackwhite):
            print(f"{'白' if blackwhite == 1 else '黒'} も置ける場所がありません。")
            print("--- ゲーム終了 ---")
            break # 両者パス
        
        continue # 自分だけパス

    # 3. 手の入力 (人間かAIか)
    
    row, col = None, None # これから決定する (row, col)

    if (current_player == 2 and two_p == "2") or (current_player == 1 and two_p == "2") or (current_player == 2 and two_p == "1"):
        # --- 人間の手番 (1Pの黒 または 2Pの白黒) ---
        # (★ 1Pモードの黒番 (player 2) も人間に変更)
        player_name = "黒" if current_player == 2 else "白"
        
        # 人間に見せるため、代数表記に変換
        legal_moves_algebraic = [numeric_to_algebraic(r, c) for r, c in legal_moves_numeric]
        print(f"{player_name} が置ける場所は以下の通りです。")   
        print(legal_moves_algebraic)
        
        while True: # 正しい入力があるまでループ
            user_input = input(f"{player_name} の番です。置きたい場所を入力してください(例:a0):")
            
            # 代数表記 -> 数値 (row, col) に変換
            move_numeric = algebraic_to_numeric(user_input)
            
            if move_numeric is None:
                print("不正な入力です。(例: a0)")
                continue
            
            # (row, col) が合法手リストにあるかチェック
            if move_numeric not in legal_moves_numeric:
                print("その場所には置けません。もう一度入力してください。")
                continue
            
            # 正しい入力
            row, col = move_numeric
            break # 入力ループを抜ける

    else:
        # --- AIの手番 ---
        
        if two_p == "1":
            # --- 1Pモード (人間 vs AI) ---
            # player 1 (白) が AI
            print("白 (機械学習AI) が思考中です...")
            
            # ★ AI司令塔を呼び出す (機械学習AI)
            move = computer_ai_commander(board, 1, evaluate) 
            
            if move is None:
                move = random.choice(legal_moves_numeric)
            row, col = move
            print(f"白 (機械学習AI) は {numeric_to_algebraic(row, col)} に置きました。")

        elif two_p == "3":
            # --- 3Pモード (AI vs AI) ---
            if current_player == 2:
                # player 2 (黒) が 機械学習AI
                print("黒 (機械学習AI) が思考中です...")
                
                # ★ AI司令塔を呼び出す (機械学習AI)
                move = computer_ai_commander(board, 2, evaluate) 
                
                if move is None:
                    print("AIが手を選べませんでした。ランダムな手を選びます。")
                    move = random.choice(legal_moves_numeric)
                row, col = move
                print(f"黒 (機械学習AI) は {numeric_to_algebraic(row, col)} に置きました。")
            else:
                # player 1 (白) が マス評価AI
                print("白 (マス評価AI) が思考中です...")
                
                # ★ AI司令塔を呼び出す (マス評価AI)
                move = computer_ai_commander(board, 1, evaluate_cell_weight) 
                
                if move is None:
                    print("AIが手を選べませんでした。ランダムな手を選びます。")
                    move = random.choice(legal_moves_numeric)
                row, col = move
                print(f"白 (マス評価AI) は {numeric_to_algebraic(row, col)} に置きました。")
            
            time.sleep(0.5) # ★AI対戦が早すぎないように 0.5秒待つ

    
    # 4. 盤面に石を置き、ひっくり返す
    # (人間が選んだ (row, col) も、AIが選んだ (row, col) も、ここで処理される)
    if row is not None and col is not None:
        board[row][col] = current_player
        flip_stones(board, row, col, current_player)
        print_board(board)
    
    # 5. 手番を交代
    blackwhite = 3 - current_player


# ゲーム終了
print("--- 最終結果 ---")
win = winner(board)
print(win)
black_count = sum(row.count(2) for row in board)
white_count = sum(row.count(1) for row in board)
print(f"黒: {black_count} 石, 白: {white_count} 石")