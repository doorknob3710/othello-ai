import json
import random
import copy
import numpy as np
import time

# --- Othelloの基本関数 (othello_numeric_unified.py と共通) ---

def board_start():
    """ 盤面の初期化 """
    board = [[0 for _ in range(8)] for _ in range(8)]
    board[3][3] = 2 # 2: 黒
    board[3][4] = 1 # 1: 白
    board[4][3] = 1
    board[4][4] = 2
    return board

def board_check(board):
    """ 盤面に空きマスがあるか (ゲーム続行可能か) """
    for i in range(8):
        for j in range(8):
            if board[i][j] == 0:
                return True
    return False

def can_place(board, player):
    """
    ★変更点★
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
    """ 石を置いて、ひっくり返す (board自体を書き換える) """
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

def winner(board):
    """
    勝敗を判定してスコアを返す
    (1.0: 黒の勝ち, -1.0: 白の勝ち, 0.0: 引き分け)
    """
    black_count = sum(row.count(2) for row in board)
    white_count = sum(row.count(1) for row in board)
    
    if black_count > white_count:
        return 1.0 # 黒の勝ち
    elif white_count > black_count:
        return -1.0 # 白の勝ち
    else:
        return 0.0 # 引き分け

# --- 自己対戦で使うAI ---

def random_ai(board, player):
    """ ランダムに手を選ぶAI """
    legal_moves = can_place(board, player) # (row, col)のリスト
    if not legal_moves:
        return None # パス
    return random.choice(legal_moves) # (row, col) を返す

# --- 1手読みAI ---
# (othello.py の evaluate と computer をコピー)
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

def evaluate_1ply(board, player):
    """ 1手読みAI用の評価関数 (cell_weight) """
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

def computer_1ply(board, player):
    """ 1手読みAI (同じスコアの手はランダムに選ぶ) """
    legal_moves = can_place(board, player) # (row, col)のリスト
    if not legal_moves:
        return None # パス

    best_score = -float('inf')
    best_moves = []
    
    for move in legal_moves: # move は (row, col) タプル
        temp_board = [row[:] for row in board]
        
        # ★変更点: (row, col) をそのまま代入
        row, col = move
        
        flip_stones(temp_board, row, col, player) 
        score = evaluate_1ply(temp_board, player)
        
        if score > best_score:
            best_score = score
            best_moves = [move]
        elif score == best_score:
            best_moves.append(move)
            
    if not best_moves:
        return random.choice(legal_moves) # 万が一

    # ベストな手（同点含む）からランダムに1つ選ぶ
    return random.choice(best_moves) # (row, col) を返す


# --- 自己対戦メインロジック ---

def play_game(ai_black, ai_white):
    """
    AI同士で1局対戦し、盤面の履歴と最終スコアを返す
    """
    board = board_start()
    game_history = [] 
    
    turn_player = 2 # 2: 黒番から
    
    while board_check(board):
        game_history.append(copy.deepcopy(board))
        
        current_ai = ai_black if turn_player == 2 else ai_white
        
        # ★変更点: move は (row, col) タプル
        move = current_ai(board, turn_player)
        
        if move is not None:
            # ★変更点: (row, col) をそのまま代入
            row, col = move
            
            if board[row][col] != 0:
                print(f"AIが不正な手 {move} を選びました。")
                return [], 0.0 
                
            flip_stones(board, row, col, turn_player)
            
        else:
            # パスの場合
            opponent_ai = ai_white if turn_player == 2 else ai_black
            if not opponent_ai(board, 3 - turn_player):
                break # 両者パス = ゲーム終了
                
        turn_player = 3 - turn_player
        
    final_score = winner(board)
    
    return game_history, final_score


if __name__ == "__main__":
    
    NUM_GAMES = 100000 # ★学習用の対局回数
    OUTPUT_FILE = "training_data.json"
    
    # ★自己対戦に使うAIを選択★
    # ----------------------------------------------------
    AI_BLACK = random_ai # 黒番AI
    AI_WHITE = random_ai # 白番AI
    # (例: 1手読みAI同士にする場合)
    # AI_BLACK = computer_1ply
    # AI_WHITE = computer_1ply
    # ----------------------------------------------------
    
    
    print(f"自己対戦を開始します (計{NUM_GAMES}局)")
    print(f"使用AI (黒): {AI_BLACK.__name__}")
    print(f"使用AI (白): {AI_WHITE.__name__}")
    
    start_time = time.time()
    all_game_data = []

    for i in range(NUM_GAMES):
        game_boards, final_score = play_game(AI_BLACK, AI_WHITE)
        
        if not game_boards:
            print(f"ゲーム {i+1}/{NUM_GAMES} はエラーによりスキップされました。")
            continue

        for board_state in game_boards:
            all_game_data.append({
                'board': board_state,
                'score': final_score
            })
            
        if (i + 1) % 100 == 0:
            print(f"... {i+1}/{NUM_GAMES} 局 完了。 (現在 {len(all_game_data)} 盤面)")
            
    end_time = time.time()
    
    print("-" * 30)
    print(f"全 {NUM_GAMES} 局の自己対戦が完了しました。")
    print(f"合計 {len(all_game_data)} の盤面データを収集。")
    print(f"所要時間: {end_time - start_time:.2f} 秒")
    
    try:
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(all_game_data, f)
        print(f"データを {OUTPUT_FILE} に保存しました。")
    except Exception as e:
        print(f"エラー: ファイルの保存に失敗しました。 {e}")