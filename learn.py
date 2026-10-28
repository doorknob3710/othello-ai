import json
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError
import time
from tqdm import tqdm

# --- 学習設定 ---
INPUT_FILE = 'training_data.json'
OUTPUT_FILE = 'pattern_weights.npy' # 最終的に othello.py が読み込むファイル

# --- パターン評価の定義 ---
# オセロの盤面は 0:空, 1:白, 2:黒 の3値で表現される。
# 各パターンを3進数として扱い、一意のインデックスに変換する。

# 3進数変換用の基数 (長さ10。最長の CORNER_TRI と EDGE_X が 10 マス)
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
                 (NUM_PAT_CORNER_TRI_PATTERNS * NUM_PAT_CORNER_TRI_LINES) # ★変更★

def get_pattern_indices(line_array, base_powers, base_offset, num_patterns):
    """
    (この関数は othello_with_ml_ai.py とは無関係)
    盤面のライン(1D-array)から特徴量インデックスを計算する
    """
    line_len = len(line_array)
    # 3進数の配列を10進数のインデックスに変換
    index_float = np.dot(line_array, base_powers[:line_len])
    # インデックスとして使うため、整数に丸めて変換
    index_int = int(np.round(index_float))
    
    # オフセットを追加
    return index_int + base_offset


def board_to_features(board_list):
    """
    8x8の盤面リストから、学習に使う「特徴量インデックス」のリストを抽出する
    """
    
    # 高速化のため、Numpy配列に変換 (float64)
    board_np = np.array(board_list, dtype=np.float64)
    # 右上/左下方向の斜めパターン抽出用に、左右反転した盤面も用意
    board_flipped = np.fliplr(board_np)
    
    feature_indices = []
    base_offset = 0

    # 1. 横パターンのインデックスを計算 (8ライン)
    powers = BASE_3_POWERS[:8]
    for r in range(NUM_HOR_LINES):
        line = board_np[r, :]
        index = np.dot(line, powers)
        feature_indices.append(int(np.round(index)) + base_offset)
    base_offset += (NUM_HOR_PATTERNS * NUM_HOR_LINES) # ★オフセットのバグ修正

    # 2. 縦パターンのインデックスを計算 (8ライン)
    powers = BASE_3_POWERS[:8]
    for c in range(NUM_VER_LINES):
        line = board_np[:, c]
        index = np.dot(line, powers)
        feature_indices.append(int(np.round(index)) + base_offset)
    base_offset += (NUM_VER_PATTERNS * NUM_VER_LINES) # ★オフセットのバグ修正

    # 3. ★新規★ 斜めパターンのインデックスを計算
    diag_patterns = [
        (NUM_DIAG8_PATTERNS, 8, [0]),           # k=0 (1ライン)
        (NUM_DIAG7_PATTERNS, 7, [1, -1]),       # k=1, -1 (2ライン)
        (NUM_DIAG6_PATTERNS, 6, [2, -2]),       # k=2, -2 (2ライン)
        (NUM_DIAG5_PATTERNS, 5, [3, -3]),       # k=3, -3 (2ライン)
        (NUM_DIAG4_PATTERNS, 4, [4, -4]),       # k=4, -4 (2ライン)
    ]

    for num_patterns, L, k_values in diag_patterns:
        powers = BASE_3_POWERS[:L]
        
        for k in k_values:
            # 1. 左上->右下 方向の斜め
            line_diag = np.diag(board_np, k=k)
            index_diag = np.dot(line_diag, powers)
            feature_indices.append(int(np.round(index_diag)) + base_offset)
            base_offset += num_patterns
            
            # 2. 右上->左下 方向の斜め (左右反転盤を使う)
            line_flipped = np.diag(board_flipped, k=k)
            index_flipped = np.dot(line_flipped, powers)
            feature_indices.append(int(np.round(index_flipped)) + base_offset)
            base_offset += num_patterns # ★オフセットのバグ修正

    # 4. ★新規★ EDGE_X パターンのインデックスを計算 (4辺)
    powers = BASE_3_POWERS[:10] # 長さ10
    
    # Top (rot 0)
    coords_top = [(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7), (1,1),(1,6)]
    line = np.array([board_np[r, c] for r, c in coords_top], dtype=np.float64)
    index = np.dot(line, powers)
    feature_indices.append(int(np.round(index)) + base_offset)
    base_offset += NUM_PAT_EDGE_X_PATTERNS
    
    # Right (rot 90)
    coords_right = [(0,7),(1,7),(2,7),(3,7),(4,7),(5,7),(6,7),(7,7), (1,6),(6,6)]
    line = np.array([board_np[r, c] for r, c in coords_right], dtype=np.float64)
    index = np.dot(line, powers)
    feature_indices.append(int(np.round(index)) + base_offset)
    base_offset += NUM_PAT_EDGE_X_PATTERNS
    
    # Bottom (rot 180)
    coords_bottom = [(7,7),(7,6),(7,5),(7,4),(7,3),(7,2),(7,1),(7,0), (6,6),(6,1)]
    line = np.array([board_np[r, c] for r, c in coords_bottom], dtype=np.float64)
    index = np.dot(line, powers)
    feature_indices.append(int(np.round(index)) + base_offset)
    base_offset += NUM_PAT_EDGE_X_PATTERNS
    
    # Left (rot 270)
    coords_left = [(7,0),(6,0),(5,0),(4,0),(3,0),(2,0),(1,0),(0,0), (6,1),(1,1)]
    line = np.array([board_np[r, c] for r, c in coords_left], dtype=np.float64)
    index = np.dot(line, powers)
    feature_indices.append(int(np.round(index)) + base_offset)
    base_offset += NUM_PAT_EDGE_X_PATTERNS
    
    # 5. ★新規★ CORNER_TRI パターンのインデックスを計算 (4コーナー)
    powers = BASE_3_POWERS[:10] # 長さ10

    # Top-Left (rot 0)
    coords_tl = [(0,0),(0,1),(0,2),(0,3), (1,0),(1,1),(1,2), (2,0),(2,1), (3,0)]
    line = np.array([board_np[r, c] for r, c in coords_tl], dtype=np.float64)
    index = np.dot(line, powers)
    feature_indices.append(int(np.round(index)) + base_offset)
    base_offset += NUM_PAT_CORNER_TRI_PATTERNS

    # Top-Right (rot 90)
    coords_tr = [(0,7),(0,6),(0,5),(0,4), (1,7),(1,6),(1,5), (2,7),(2,6), (3,7)]
    line = np.array([board_np[r, c] for r, c in coords_tr], dtype=np.float64)
    index = np.dot(line, powers)
    feature_indices.append(int(np.round(index)) + base_offset)
    base_offset += NUM_PAT_CORNER_TRI_PATTERNS
    
    # Bottom-Right (rot 180)
    coords_br = [(7,7),(7,6),(7,5),(7,4), (6,7),(6,6),(6,5), (5,7),(5,6), (4,7)]
    line = np.array([board_np[r, c] for r, c in coords_br], dtype=np.float64)
    index = np.dot(line, powers)
    feature_indices.append(int(np.round(index)) + base_offset)
    base_offset += NUM_PAT_CORNER_TRI_PATTERNS
    
    # Bottom-Left (rot 270)
    coords_bl = [(7,0),(6,0),(5,0),(4,0), (7,1),(6,1),(5,1), (7,2),(6,2), (7,3)]
    line = np.array([board_np[r, c] for r, c in coords_bl], dtype=np.float64)
    index = np.dot(line, powers)
    feature_indices.append(int(np.round(index)) + base_offset)
    base_offset += NUM_PAT_CORNER_TRI_PATTERNS
    
    return feature_indices


def build_sparse_matrix(X_indices, y_scores, total_features):
    """
    (この関数は othello_with_ml_ai.py とは無関係)
    scikit-learn に渡すためのスパースな訓練データ (X, y) を構築する
    """
    
    # ターゲット（y）: 各盤面の最終スコア (黒の勝ち=1.0, 白の勝ち=-1.0)
    y = np.array(y_scores, dtype=np.float64)
    
    # 特徴量（X）:
    # 盤面 N 個 × 特徴量 M 個 の巨大な行列（ほぼ0）の代わりに、
    # (i, j) = 1.0 という形式のスパース行列データを作成する
    
    # data: 非ゼロ要素の値 (今回はすべて 1.0)
    # row_ind: 非ゼロ要素の「行」インデックス (盤面ID)
    # col_ind: 非ゼロ要素の「列」インデックス (特徴量ID)
    
    data = []
    row_ind = []
    col_ind = []
    
    num_boards = len(X_indices)
    num_features_per_board = len(X_indices[0]) # 1盤面あたりの特徴量数 (例: 34個)
    
    # データを1次元のリストにフラット化
    data = np.ones(num_boards * num_features_per_board, dtype=np.float64)
    
    # 各特徴量がどの盤面(行)に属するか
    # [0, 0, ..., 1, 1, ..., 2, 2, ...]
    row_ind = np.repeat(np.arange(num_boards), num_features_per_board)
    
    # 各特徴量のインデックス (列)
    # X_indices を [idx1, idx2, idx3, ...] の形にフラット化
    col_ind = np.concatenate(X_indices)

    # CSR (Compressed Sparse Row) 形式のスパース行列を構築
    from scipy.sparse import csr_matrix
    X = csr_matrix((data, (row_ind, col_ind)), shape=(num_boards, total_features))
    
    return X, y


if __name__ == "__main__":
    
    print(f"特徴量（評価テーブル）の総数: {TOTAL_FEATURES}") # 612096 になっているはず
    
    start_time_all = time.time()
    
    # --- ステップ1: 教師データの読み込み ---
    print(f"ステップ1: {INPUT_FILE} から教師データを読み込んでいます...")
    try:
        with open(INPUT_FILE, 'r') as f:
            # (変更) 巨大なJSONリストを一度に正しく読み込む
            all_game_data = json.load(f)
        
        num_data = len(all_game_data)
        print(f"{num_data} 件の盤面データを読み込みました。")
        
        if num_data == 0:
            raise Exception(f"{INPUT_FILE} が空です。")
            
    except json.JSONDecodeError as e:
        print(f"エラー: {INPUT_FILE} の読み込みに失敗しました。 {e}")
        print("ファイルが壊れているか、self_play.py が中断された可能性があります。")
        print("self_play.py を再実行して、training_data.json を作り直してください。")
        exit()
    except FileNotFoundError:
        print(f"エラー: {INPUT_FILE} が見つかりません。")
        print("self_play.py を実行して、training_data.json を作成してください。")
        exit()
    except Exception as e:
        print(f"予期せぬエラー: {e}")
        exit()

    # --- ステップ2: 特徴量 (X) とターゲット (y) の準備 ---
    
    # all_game_data から X (特徴量インデックス) と y (スコア) を抽出
    X_indices = []
    y_scores = []
    
    # (変更) board_to_features が重いので、tqdmで進捗を表示
    print("ステップ2: 盤面データから特徴量を抽出しています...")
    for data in tqdm(all_game_data, desc="特徴量抽出中"):
        # board_to_features を呼び出し
        features = board_to_features(data['board'])
        X_indices.append(features)
        y_scores.append(data['score'])
        
    # (メモリ解放) 巨大なJSONデータはもう不要
    del all_game_data

    # --- ステップ3: スパース行列の構築 ---
    print("ステップ3: 機械学習用のスパース行列を構築しています...")
    try:
        X_sparse, y_sparse = build_sparse_matrix(X_indices, y_scores, TOTAL_FEATURES)
    except Exception as e:
        print(f"エラー: スパース行列の構築に失敗しました。 {e}")
        print("TOTAL_FEATURES の定義や base_offset の計算が間違っている可能性があります。")
        exit()
        
    # (メモリ解放)
    del X_indices
    del y_scores

    # --- ステップ4: 機械学習の実行 ---
    print(f"ステップ4: 機械学習を開始します (全 {num_data} 件)")
    
    # 1. スケーラーの準備 (標準化)
    #    スパース行列の場合、'with_mean=False' が必須
    scaler = StandardScaler(with_mean=False) 

    # 2. モデルの準備 (線形回帰)
    #    SGDRegressor は大規模データ（スパース行列）を効率的に学習できる
    model = SGDRegressor(
        loss='squared_error', # 損失関数: 最小二乗法
        penalty='l2',         # 正則化: L2 (過学習を防ぐ)
        alpha=0.0001,         # 正則化の強さ
        max_iter=1000,        # 最大反復回数
        tol=1e-3,             # 収束の許容誤差
        shuffle=True,         # 各反復でデータをシャッフル
        random_state=42,
        eta0=0.01,            # 初期学習率
        learning_rate='invscaling', # 学習率を徐々に下げる
        verbose=0             # ログの出力レベル
    )

    start_learn_time = time.time()
    
    try:
        # データを標準化
        # X_scaled = scaler.fit_transform(X_sparse)
        # (標準化はSGDRegressorが内部で行う or しない方が良い場合もある)
        # 今回は、入力が (0, 1) がメインなのでスケーリングなしで試す
        
        # モデルの学習 (ミニバッチ学習)
        # 巨大なデータを一度にメモリにロードせず、部分的に学習させる
        # (tqdm を使って進捗を表示)
        batch_size = 10000 # 1万件ずつ学習
        n_batches = int(np.ceil(num_data / batch_size))
        
        classes = np.array([1.0, -1.0]) # スコアのクラス (SGDRegressorでは不要だが...)
                                        # partial_fit のために必要？
                                        # -> SGDRegressorには不要
        
        # tqdm (プログレスバー) のセットアップ
        with tqdm(total=n_batches, desc="学習中 (SGD)") as pbar:
            for i in range(n_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, num_data)
                
                # スパース行列からバッチをスライス
                X_batch = X_sparse[start:end]
                y_batch = y_sparse[start:end]
                
                # partial_fit でバッチ学習
                model.partial_fit(X_batch, y_batch)
                pbar.update(1)

        print(f"\n学習が完了しました。 (所要時間: {time.time() - start_learn_time:.2f} 秒)")

    except MemoryError:
        print("\nエラー: メモリが不足しています。")
        print("マシンを再起動するか、NUM_GAMES を減らして training_data.json を作り直してください。")
        exit()
    except Exception as e:
        print(f"\nエラー: 学習中にエラーが発生しました。 {e}")
        exit()

    # --- ステップ5: 学習結果の保存 ---
    print(f"ステップ5: 学習結果を {OUTPUT_FILE} に保存しました。")
    try:
        # 学習結果（重み）は model.coef_ に格納されている
        learned_weights = model.coef_.astype(np.float64) # float64で保存
        np.save(OUTPUT_FILE, learned_weights)
        
    except NotFittedError:
        print("エラー: モデルが学習されていません。")
    except Exception as e:
        print(f"エラー: モデルの保存に失敗しました。 {e}")

    print("-" * 30)
    print(f"--- 全工程完了 --- (総所要時間: {time.time() - start_time_all:.2f} 秒)")