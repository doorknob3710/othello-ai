#リバーシゲーム
#8x8の盤面を作成する
#2を黒,1を白とする
#初期配置は中央に黒2つ,白2つ
#初めは8*8の盤面に0を入れる
import IPython
from IPython.display import clear_output
import random
import numpy as np


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


# --- 評価関数 ---
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

def evaluate(board, player):
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

# --- NegaMax (AIの探索アルゴリズム) ---
def negamax(board, player, depth):
    opponent = 3 - player

    if depth == 0:
        return evaluate(board, player)

    legal_moves = can_place(board, player) # (row, col) のリスト

    if not legal_moves:
        if not can_place(board, opponent):
            return evaluate(board, player) # ゲーム終了
        score = -negamax(board, opponent, depth - 1) # パス
        return score

    best_score = -float('inf')
    for move in legal_moves: # move は (row, col) タプル
        temp_board = [row[:] for row in board]
        
        # ★変更点: (row, col) をそのまま代入
        row, col = move
        temp_board[row][col] = player
        
        flip_stones(temp_board, row, col, player)
        
        score = -negamax(temp_board, opponent, depth - 1)
        
        if score > best_score:
            best_score = score
    
    return best_score

# --- NegaScout (NegaMaxの高速版) ---
def negascout(board, player, depth, alpha, beta):
    opponent = 3 - player

    if depth == 0:
        return evaluate(board, player)

    legal_moves = can_place(board, player) # (row, col) のリスト

    if not legal_moves:
        if not can_place(board, opponent):
            return evaluate(board, player) # ゲーム終了
        return -negascout(board, opponent, depth - 1, -beta, -alpha) # パス

    is_first_move = True
    for move in legal_moves: # move は (row, col) タプル
        temp_board = [row[:] for row in board]
        
        # ★変更点: (row, col) をそのまま代入
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
            return alpha  # 枝刈り

    return alpha


# --- AIの呼び出し関数 (AIの「手」を選ぶ) ---

def computer_scout_6(board, player=1):
    opponent = 3 - player
    best_score = -float('inf') 
    best_move = None
    legal_moves = can_place(board, player) # (row, col) のリスト

    if not legal_moves:
        return None 

    alpha = -float('inf')
    beta = float('inf')
    
    is_first_move = True # is_first_move に修正 (元のコードのtypo修正)
    for move in legal_moves: # move は (row, col) タプル
        temp_board = [row[:] for row in board]
        
        # ★変更点: (row, col) をそのまま代入
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
            best_move = move # move (row, col) をそのまま保存
        
        alpha = max(alpha, best_score)

    if best_move is None:
        return random.choice(legal_moves) 
    return best_move # (row, col) タプルを返す


def computer_scout_8(board, player=1):
    opponent = 3 - player
    best_score = -float('inf')
    best_move = None
    legal_moves = can_place(board, player) # (row, col) のリスト

    if not legal_moves:
        return None 

    alpha = -float('inf')
    beta = float('inf')
    
    is_first_move = True
    for move in legal_moves: # move は (row, col) タプル
        temp_board = [row[:] for row in board]
        
        # ★変更点: (row, col) をそのまま代入
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
            best_move = move # move (row, col) をそのまま保存
        
        alpha = max(alpha, best_score)

    if best_move is None:
        return random.choice(legal_moves)
    return best_move # (row, col) タプルを返す


def computer(board, player=1):
    """ 1手読みAI """
    legal_moves = can_place(board, player) # (row, col) のリスト
    if not legal_moves:
        return random.choice(legal_moves) if legal_moves else None

    best_score = -float('inf')
    best_move = None
    for move in legal_moves: # move は (row, col) タプル
        temp_board = [row[:] for row in board]
        
        # ★変更点: (row, col) をそのまま代入
        row, col = move
        temp_board[row][col] = player
        
        flip_stones(temp_board, row, col, player)
        score = evaluate(temp_board, player)
        
        if score > best_score:
            best_score = score
            best_move = move # move (row, col) をそのまま保存

    return best_move


def computer_4(board, player=1):
    """ 4手読みAI (NegaMax depth=3) """
    opponent = 3 - player
    best_score = -float('inf')
    best_move = None
    legal_moves = can_place(board, player) # (row, col) のリスト

    if not legal_moves:
        return None

    for move in legal_moves: # move は (row, col) タプル
        temp_board = [row[:] for row in board]
        
        # ★変更点: (row, col) をそのまま代入
        row, col = move
        temp_board[row][col] = player
        
        flip_stones(temp_board, row, col, player)

        score = -negamax(temp_board, opponent, 3) # depth=3 (計4手読み)
        
        if score > best_score:
            best_score = score
            best_move = move # move (row, col) をそのまま保存

    if best_move is None:
        return random.choice(legal_moves)
    return best_move # (row, col) タプルを返す


def computer_6(board, player=1):
    """ 6手読みAI (NegaMax depth=5) """
    opponent = 3 - player
    best_score = -float('inf')
    best_move = None
    legal_moves = can_place(board, player) # (row, col) のリスト

    if not legal_moves:
        return None

    for move in legal_moves: # move は (row, col) タプル
        temp_board = [row[:] for row in board]
        
        # ★変更点: (row, col) をそのまま代入
        row, col = move
        temp_board[row][col] = player

        flip_stones(temp_board, row, col, player)

        score = -negamax(temp_board, opponent, 5) # depth=5 (計6手読み)
        
        if score > best_score:
            best_score = score
            best_move = move # move (row, col) をそのまま保存

    if best_move is None:
        return random.choice(legal_moves)
    return best_move # (row, col) タプルを返す

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
board = board_start()
print_board(board)
blackwhite = 2 # 2:黒(人間), 1:白(AI/人間)
two_p =input("1人プレイなら1,2人プレイなら2を入力してください:")


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

    if (current_player == 2) or (current_player == 1 and two_p == "2"):
        # --- 人間の手番 (黒 または 2Pの白) ---
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
        # --- AIの手番 (1Pの白) ---
        print("白 (AI) が思考中です...")
        
        # AIを選択 (★ここでAIを切り替える)
        # move = computer(board, 1)
        # move = computer_4(board, 1)
        # move = computer_6(board, 1)
        # move = computer_scout_6(board, 1)
        move = computer_scout_8(board, 1) # 8手読み
        
        if move is None:
            # (AIがバグなどでNoneを返した場合の保険)
            print("AIが手を選べませんでした。ランダムな手を選びます。")
            move = random.choice(legal_moves_numeric)

        # AIが返した (row, col) を代入
        row, col = move
        print(f"白 (AI) は {numeric_to_algebraic(row, col)} に置きました。")

    
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