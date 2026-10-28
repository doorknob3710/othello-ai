# Othello AI

Pythonで制作したオセロAI。NegaScout探索、反復深化、Numbaによる高速化、ビットボード、自己対戦による評価関数の学習を試した開発記録です。

## 実行方法

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python othello_bit.py
```

メニューからAI同士の対戦または人間対AIを選択します。初回はNumbaのコンパイルに時間がかかります。動作確認済みのPython・依存ライブラリのバージョンは今後追記します。

## 主なファイル

- `othello_bit.py`: ビットボード版。NegaScout、反復深化、Multi Prob Cut、終盤の完全読みを実装。
- `othello_bit8.py`: ビットボード版の別バージョン。
- `othello_5.py`: 1手10秒を設定した反復深化版。
- `othello_10vs11.py`: 探索深さを比較する版。
- `othello_numba*.py`, `othello_4.py`: 高速化・探索の試作。
- `othello.py`, `othello_new.py`: 初期の探索・評価関数の試作。
- `self_play*.py`, `learn.py`, `ai_vs_ai.py`: 自己対戦データの生成、評価関数学習、比較用のコード。

## 時間制限について

通常探索には10秒の設定があります。ただし、`othello_bit.py` は終盤の完全読みに入ると時間制限を解除します。探索深さ比較モードも時間制限なしです。そのため、現状のコードは「すべての局面で10秒以内」を保証しません。Numbaのコンパイル時間も探索時間とは別に発生します。
