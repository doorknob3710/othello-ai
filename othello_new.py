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
# パターンを (3^N) 進数として一意のインデックスに変換するための「基数」
BASE_3_POWERS = np.array([3**i for i in range(8)], dtype=np.int64)

# 1. 横8マスのパターン
NUM_HOR_PATTERNS = 3**8 # 6561
NUM_HOR_LINES = 8
# 2. 縦8マスのパターン
NUM_VER_PATTERNS = 3**8 # 6561
NUM_VER_LINES = 8

# 特徴量の総数 (learn.py と一致している必要がある)
TOTAL_FEATURES = (NUM_HOR_PATTERNS * NUM_HOR_LINES) + \
                 (NUM_VER_PATTERNS * NUM_VER_LINES)

if G_PATTERN_TABLE is not None and len(G_PATTERN_TABLE) != TOTAL_FEATURES:
    print("警告: 読み込んだ pattern_weights.npy のサイズが、")
    print(f"       プログラムの定義 ({TOTAL_FEATURES}) と一致しません。")
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


# --- ★ステップ3: 新しいAI評価関数 (Numbaで高速化) ---

@numba.jit(nopython=True) # Numbaのおまじない (C++並みに高速化)
def calculate_pattern_score(board_np, pattern_table):
    """
    Numbaで高速化される評価計算の「中身」
    引数: board_np (Numpy配列), pattern_table (学習済みの重み)
    """
    total_score = 0.0
    
    # 1. 横パターンのスコアを合計
    base_offset = 0
    powers = BASE_3_POWERS
    
    for r in range(NUM_HOR_LINES):
        line = board_np[r, :] # 横1ライン (8マス)
        # 3進数の配列を10進数のインデックスに変換
        index = np.dot(line, powers)
        # 評価テーブルからスコアを引いて加算
        total_score += pattern_table[index + base_offset]
        base_offset += NUM_HOR_PATTERNS # 次のラインのオフセットにずらす

    # 2. 縦パターンのスコアを合計
    # base_offset は横パターンの最後(6561 * 7)を引き継いでいる
    for c in range(NUM_VER_LINES):
        line = board_np[:, c] # 縦1ライン (8マス)
        # 3進数の配列を10進数のインデックスに変換
        index = np.dot(line, powers)
        # 評価テーブルからスコアを引いて加算
        total_score += pattern_table[index + base_offset]
        base_offset += NUM_VER_PATTERNS # 次のラインのオフセットにずらす
    
    # (斜めパターンなどもここに追加していく)

    return total_score


def evaluate(board, player):
    """
    NegaScoutから呼ばれる評価関数の「入り口」
    (古い cell_weight の evaluate と差し替え)
    """
    if G_PATTERN_TABLE is None:
        print("エラー: 評価テーブルが読み込まれていないため、評価できません。")
        return 0.0 # 評価値0を返す

    # 1. Pythonのリスト(board)を、Numbaが扱える
    #    Numpy配列 (np.int64) に変換する
    #    (learn.py は 0, 1, 2 で学習したので、そのまま使う)
    board_np = np.array(board, dtype=np.int64) 

    # 2. Numbaでコンパイルされた超高速な関数を呼び出す
    #    注: 学習データ(training_data.json)は、
    #        「黒(2)視点での最終的な勝敗」を学習しています。
    #        そのため、この関数が返すスコアは「黒(2)にとっての評価値」です。
    score = calculate_pattern_score(board_np, G_PATTERN_TABLE)
    
    # 3. プレイヤーに応じてスコアを調整
    if player == 2: # 黒番
        return score # 黒にとっての評価値をそのまま返す
    else: # 白番 (player == 1)
        return -score # 黒にとっての評価値を反転 (白にとっての評価値) にする

# -------------------------------------------------


# --- NegaMax (AIの探索アルゴリズム) ---
# (変更なし: evaluate()を呼び出すだけ)
def negamax(board, player, depth):
    # ... (othello_numeric_unified.py と同じ)
    opponent = 3 - player
    if depth == 0:
        return evaluate(board, player)
    legal_moves = can_place(board, player)
    if not legal_moves:
        if not can_place(board, opponent):
            return evaluate(board, player)
        score = -negamax(board, opponent, depth - 1)
        return score
    best_score = -float('inf')
    for move in legal_moves:
        temp_board = [row[:] for row in board]
        row, col = move
        temp_board[row][col] = player
        flip_stones(temp_board, row, col, player)
        score = -negamax(temp_board, opponent, depth - 1)
        if score > best_score:
            best_score = score
    return best_score


# --- NegaScout (NegaMaxの高速版) ---
# (変更なし: evaluate()を呼び出すだけ)
def negascout(board, player, depth, alpha, beta):
    # ... (othello_numeric_unified.py と同じ)
    opponent = 3 - player
    if depth == 0:
        return evaluate(board, player)
    legal_moves = can_place(board, player)
    if not legal_moves:
        if not can_place(board, opponent):
            return evaluate(board, player)
        return -negascout(board, opponent, depth - 1, -beta, -alpha)
    is_first_move = True
    for move in legal_moves:
        temp_board = [row[:] for row in board]
        row, col = move
        temp_board[row][col] = player
        flip_stones(temp_board, row, col, player)
        if is_first_move:
            score = -negascout(temp_board, opponent, depth - 1, -beta, -alpha)
            is_first_move = False
        else:
            score = -negascout(temp_board, opponent, depth - 1, -alpha - 1, -alpha)
            if alpha < score < beta:
                score = -negascout(temp_board, opponent, depth - 1, -beta, -score)
        alpha = max(alpha, score)
        if alpha >= beta:
            return alpha
    return alpha


# --- AIの呼び出し関数 (AIの「手」を選ぶ) ---
# (変更なし: negascout()を呼び出すだけ)

def computer_scout_6(board, player=1):
    # ... (othello_numeric_unified.py と同じ)
    opponent = 3 - player
    best_score = -float('inf') 
    best_move = None
    legal_moves = can_place(board, player)
    if not legal_moves:
        return None 
    alpha = -float('inf')
    beta = float('inf')
    is_first_move = True
    for move in legal_moves:
        temp_board = [row[:] for row in board]
        row, col = move
        temp_board[row][col] = player
        flip_stones(temp_board, row, col, player)
        if is_first_move:
            score = -negascout(temp_board, opponent, 5, -beta, -alpha) # depth 5 (計6手)
            is_first_move = False
        else:
            score = -negascout(temp_board, opponent, 5, -alpha - 1, -alpha)
            if alpha < score < beta:
                score = -negascout(temp_board, opponent, 5, -beta, -score)
        if score > best_score:
            best_score = score
            best_move = move
        alpha = max(alpha, best_score)
    if best_move is None:
        return random.choice(legal_moves) 
    return best_move


def computer_scout_8(board, player=1):
    # ... (othello_numeric_unified.py と同じ)
    opponent = 3 - player
    best_score = -float('inf')
    best_move = None
    legal_moves = can_place(board, player)
    if not legal_moves:
        return None 
    alpha = -float('inf')
    beta = float('inf')
    is_first_move = True
    for move in legal_moves:
        temp_board = [row[:] for row in board]
        row, col = move
        temp_board[row][col] = player
        flip_stones(temp_board, row, col, player)
        if is_first_move:
            score = -negascout(temp_board, opponent, 7, -beta, -alpha) # depth 7 (計8手)
            is_first_move = False
        else:
            score = -negascout(temp_board, opponent, 7, -alpha - 1, -alpha)
            if alpha < score < beta:
                score = -negascout(temp_board, opponent, 7, -beta, -score)
        if score > best_score:
            best_score = score
            best_move = move
        alpha = max(alpha, best_score)
    if best_move is None:
        return random.choice(legal_moves)
    return best_move

# (古いAI (computer, computer_4, computer_6) は
#  新しい evaluate() を使うので、そのままでも動きますが、
#  混乱を避けるためコメントアウトしても良いでしょう)
# def computer(...):
# def computer_4(...):
# def computer_6(...):


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
# (変更なし)

board = board_start()
print_board(board)
blackwhite = 2 # 2:黒(人間), 1:白(AI/人間)
two_p =input("1人プレイなら1,2人プレイなら2を入力してください:")

# AIが動作可能かチェック
if two_p == "1" and G_PATTERN_TABLE is None:
    print("AIの評価テーブルが読み込めなかったため、1Pモードを開始できません。")
    print("learn.py を実行して pattern_weights.npy を作成してください。")
    # (ここで終了させる)
    exit()


while board_check(board) == True:
    
    # ... (othello_numeric_unified.py と同じ)
    current_player = blackwhite
    legal_moves_numeric = can_place(board, current_player)

    if not legal_moves_numeric:
        player_name = "黒" if current_player == 2 else "白"
        print(f"{player_name} は置ける場所がありません。")
        blackwhite = 3 - current_player
        if not can_place(board, blackwhite):
            print(f"{'白' if blackwhite == 1 else '黒'} も置ける場所がありません。")
            print("--- ゲーム終了 ---")
            break
        continue

    row, col = None, None

    if (current_player == 2) or (current_player == 1 and two_p == "2"):
        player_name = "黒" if current_player == 2 else "白"
        legal_moves_algebraic = [numeric_to_algebraic(r, c) for r, c in legal_moves_numeric]
        print(f"{player_name} が置ける場所は以下の通りです。")   
        print(legal_moves_algebraic)
        
        while True:
            user_input = input(f"{player_name} の番です。置きたい場所を入力してください(例:a0):")
            move_numeric = algebraic_to_numeric(user_input)
            if move_numeric is None:
                print("不正な入力です。(例: a0)")
                continue
            if move_numeric not in legal_moves_numeric:
                print("その場所には置けません。もう一度入力してください。")
                continue
            row, col = move_numeric
            break

    else:
        # --- AIの手番 (1Pの白) ---
        print("白 (AI) が思考中です...")
        
        # ★AIを選択★
        # move = computer_scout_6(board, 1) # 6手読み
        move = computer_scout_8(board, 1) # 8手読み
        
        if move is None:
            move = random.choice(legal_moves_numeric)

        row, col = move
        print(f"白 (AI) は {numeric_to_algebraic(row, col)} に置きました。")

    
    if row is not None and col is not None:
        board[row][col] = current_player
        flip_stones(board, row, col, current_player)
        print_board(board)
    
    blackwhite = 3 - current_player


# ゲーム終了
print("--- 最終結果 ---")
win = winner(board)
print(win)
black_count = sum(row.count(2) for row in board)
white_count = sum(row.count(1) for row in board)
print(f"黒: {black_count} 石, 白: {white_count} 石")