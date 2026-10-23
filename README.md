# JAXON3 Eval Only

学習済み JAXON3 ランニングポリシーを Genesis の平面上で実行する評価専用リポジトリです。
配布用 `policy.pt` と対応する `eval_config.json`、URDF・メッシュを使って実行します。
報酬関数、報酬係数、学習処理、critic、optimizer、学習ログ、ハードル・模倣データは含みません。

## 環境構築

Linux x86_64 / Python 3.10 / NVIDIA GPU を想定しています。ビューアにはデスクトップ環境が必要です。
既存の conda を利用して、次の環境を作成します。

```bash
conda create -n jaxon3_evalonly python=3.10 pip -y
conda activate jaxon3_evalonly
python -m pip install --upgrade pip
nvidia-smi
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
cd ~/jaxon3_evalonly
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

主要バージョンは Genesis `1.3.3`、PyTorch `2.8.0`（CUDA 12.8）、NumPy `2.2.6`、SciPy `1.15.3` です。
PyTorch の CUDA 12.8 パッケージは [公式の過去バージョン一覧](https://pytorch.org/get-started/previous-versions/#v280)、
Genesis は [1.3.3 の配布ページ](https://pypi.org/project/genesis-world/1.3.3/) を参照してください。
評価する側で `rsl-rl-lib`、`keyboard`、TensorBoard をインストールする必要はありません。
Genesis 本体の clone や元の学習リポジトリも不要です。

## モデルの配置

```text
jaxon3_evalonly/
├── jaxon3_eval.py
├── jaxon3_env_evalonly.py
├── requirements.txt
├── policy/
│   ├── policy.pt                 # 別途受け取る推論用モデル（Git 管理対象外）
│   ├── eval_config.json          # この policy.pt と対になる評価設定
│   └── models/
│       ├── urdf/                # 選択された 1 個の URDF
│       └── meshes/              # URDF が参照するメッシュのみ
└── tools/export_policy.py       # モデル提供者向け
```

`policy.pt` は TorchScript 形式です。学習時の `model_2500.pt` 等を名前だけ変更しても使えません。
提供者が後述のコマンドで推論用に変換します。観測の正規化統計もこのファイルに含まれます。
Git から取得した直後は `.pt` がないため、提供者から受け取ったファイルを `policy/policy.pt` に置いてください。
評価設定には関節順序・PD ゲイン・観測スケールなど、推論とシミュレーションに必要な項目だけを保存します。
起動時に重みとモデルファイルの SHA-256 を確認し、異なる組み合わせを検出します。

## 実行

```bash
conda activate jaxon3_evalonly
cd ~/jaxon3_evalonly
python jaxon3_eval.py --command 3.0 0.0 0.0
```

`--command VX VY YAW` は前進速度 [m/s]、横方向速度 [m/s]、旋回角速度 [rad/s] です。
省略時は `3.0 0.0 0.0`。ビューアのカメラはロボットを追従します。
終了するには Ctrl+C またはビューアのウィンドウを閉じます。初回は物理カーネルのコンパイルに時間がかかります。

画面を表示せずに 1,000 ステップ（同梱設定では 5 秒）実行する場合:

```bash
python jaxon3_eval.py --headless --max-steps 1000 --command 3.0 0.0 0.0
```

別の配布用モデル一式を使う場合:

```bash
python jaxon3_eval.py --bundle /path/to/another_policy --command 5.0 0.0 0.0
```

| オプション | 内容 |
| --- | --- |
| `--bundle DIR` | `policy.pt`、JSON、`models/` を含むフォルダ。既定はスクリプトの隣の `policy/` |
| `--command VX VY YAW` | 速度指令 |
| `--headless` | ビューアなし。実時間への待ち合わせも行わない |
| `--max-steps N` | 実行ステップ数。既定の `0` は無制限 |
| `--num-envs N` | 並列ロボット数。既定は `1`、描画は先頭のみ |
| `--device cuda` / `--device cpu` | バックエンド。既定は CUDA、CPU は低速な動作確認用 |
| `--show-collision` | 衝突形状を表示 |
| `--fixed-camera` | カメラ追従を無効化 |
| `--seed N` | 乱数シード。既定は `1` |

どの作業ディレクトリからでもスクリプトの絶対パスで実行できます。

## 評価環境の動作

- 元の関節順序、観測順序、スケール、PD 制御、トルク上限、armature を維持します。
- 行動は 1 ステップ遅延して適用します。頭部／ベース速度、body／heading 座標系、ローパスフィルタ、頭部速度の遅延に対応します。
- 平面、固定の初期姿勢・向き、一定の速度目標を使います。観測内の平滑化された速度指令は元コードと同じ加速度上限で更新します。
- 摩擦係数は床・ロボットとも `1.0`、質量・重心・ゲイン等は公称値です。物理ランダム化と観測ノイズは適用しません。
- roll / pitch の閾値超過またはエピソード時間超過で自動リセットします。元コードの速度追従誤差による打ち切りやランダム指令更新はありません。

この構成は共有したポリシーの走行を再生・観察するためのものです。学習環境でのスコア集計や、その終了条件を含む性能評価は行いません。

## モデル提供者: 別のチェックポイントを書き出す

この操作は、元のチェックポイントと対応する `cfgs.pkl` を保持する側だけで行います。
自分が作成した信頼できる `cfgs.pkl` を指定してください（pickle の読込はコードを実行できる形式です）。
評価設定を手で推測せず、必ずそのチェックポイントの学習設定を使います。

既存の学習用環境で `rsl-rl-lib==5.5.1` を使用してください。
書き出しに必要な実装は [RSL-RL 5.5.1](https://pypi.org/project/rsl-rl-lib/5.5.1/) に含まれます。

```bash
python tools/export_policy.py \
  --checkpoint /path/to/running/model_2500.pt \
  --config /path/to/running/cfgs.pkl \
  --output /path/to/new_policy_bundle
```

`--config` を省略するとチェックポイントと同じフォルダの `cfgs.pkl` を使います。
出力先には新しいフォルダ名を指定してください。既存フォルダは上書きしません。
URDF の元の保存場所が変わった場合は `--model-dir /path/to/models/jaxon3/urdf` を指定します。
複数 URDF で学習したモデルは `--urdf-index 0` 等で 1 個を選択します。
対応するのは 5.5.1 の MLP actor と通常走行の観測形式です。旧 rsl-rl のチェックポイントや別タスクは対象外です。

出力先に推論用 `.pt`、評価設定、必要なモデルがまとめて作成されます。
書き出し時には元の actor と推論用モデルの出力をバッチ数 1・4 で比較します。
共有するのは出力フォルダの内容です。元の `cfgs.pkl` と学習チェックポイントは共有用リポジトリに置きません。
新しいモデルへ差し替える際は、JSON・URDF・メッシュも同じ出力一式に更新してください。

## GitHub への登録

`.gitignore` で `.pt`、`.pkl`、ログ、キャッシュを除外しています。
コード・評価用 JSON・モデルデータを Git に登録し、`policy.pt` は別途受け渡します。

```bash
git add README.md requirements.txt .gitignore jaxon3_eval.py jaxon3_env_evalonly.py tools tests policy
git diff --cached --stat
git commit -m "Add standalone JAXON3 running evaluation"
git remote add origin <共有用リポジトリのURL>
git push -u origin main
```

## トラブル対応

- `policy.pt does not match ...`: 提供者から同時に出力された `.pt` と JSON を使ってください。
- `Missing or modified model asset`: `models/` の相対配置・内容を元の配布一式に戻してください。
- `CUDA is unavailable`: `nvidia-smi`、使用している Python 環境、PyTorch の CUDA 版を確認してください。
- GUI が表示できないサーバーでは `--headless` を使用してください。
- CPU で動作を切り分ける場合は `--device cpu --headless --max-steps 10` を指定します。
- DAE 読込時の legacy parser / convex hull に関する警告は元モデルでも生じます。物理特性の再現性のため Genesis のバージョンを固定してください。

設定書き出しの検査は `python -m unittest discover -s tests`、シミュレーションの確認は上記の headless コマンドで実行します。

## 動作確認記録

2026-10-23、上記バージョンの既存 Linux / CUDA 環境で確認しました。

- 推論用モデルと元 actor の出力一致（バッチ数 1・4）。
- GPU headless 1,000 ステップ、指令 `3 0 0`、5 秒間をリセットなしで完走。
- GUI と追従カメラで 100 ステップを完走。
- 元環境のランダム化を無効化し、同じ指令で 2 体・100 ステップを比較。観測の最大絶対差は約 `3e-6`、トルクは比較許容誤差内で一致。
- 速度観測の各座標系・遅延と部分リセット、設定の許可リストのテストに成功。

CPU 実行と、新規 conda 環境へのインストールからの通し実行は未検証です。
