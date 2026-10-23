# モデル提供者向け README

評価環境の構築と実行方法は [README](README.md) を参照してください。

## 別のチェックポイントを書き出す

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

## 設定書き出しの検査

リポジトリのルートで次のコマンドを実行します。

```bash
python -m unittest discover -s tests
```
