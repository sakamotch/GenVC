# Voice Editing Feature

GenVCに統計的潜在空間編集機能を追加しました。この機能により、自分の声を「ピッチを高く」「声を明るく」などの方向に調整できます。

## 概要

PCA/ICAを用いて潜在空間の主要な変動軸を発見し、それらの軸に沿って声質を編集します。

### 基本的な仕組み

```
[あなたの声] → 潜在ベクトル抽出 → 統計的変換 → 編集された潜在ベクトル
                                          ↓
[ソース音声] ← 音声生成 ← 編集された声質
```

## セットアップ

### 1. 追加の依存関係

```bash
pip install scikit-learn matplotlib seaborn gradio
```

### 2. 必要なデータ

- LibriTTSデータセット（またはその他の多様な話者を含むデータセット）
- 学習済みGenVCモデル

## 使用方法

### ステップ1: 潜在ベクトルの抽出

LibriTTSから話者ごとの代表的な潜在ベクトルを抽出します。

```bash
python extract_latents.py \
    --model_path pre_trained/GenVC_small.pth \
    --libritts_path /path/to/LibriTTS \
    --split train-clean-100 \
    --output_dir latent_data \
    --max_audios_per_speaker 5
```

**出力:**
- `latent_data/speaker_vectors.npy` - 話者ベクトルの行列 (num_speakers, 1024)
- `latent_data/speaker_ids.json` - 話者IDリスト
- `latent_data/metadata.json` - メタデータ

**オプション:**
- `--max_speakers N` - テスト用に最大N人の話者のみ処理
- `--save_all_latents` - 各発話の潜在ベクトルも保存（大容量）

### ステップ2: PCA/ICA分析

抽出した潜在ベクトルに対してPCA/ICA分析を実行します。

```bash
# PCA分析
python analyze_latent_space.py \
    --data_dir latent_data \
    --output_dir analysis_results \
    --n_pca_components 50 \
    --standardize

# ICA分析も実行する場合
python analyze_latent_space.py \
    --data_dir latent_data \
    --output_dir analysis_results \
    --n_pca_components 50 \
    --n_ica_components 20 \
    --standardize
```

**出力:**
- `analysis_results/pca/` - PCA結果
  - `components.npy` - 主成分ベクトル (50, 1024)
  - `explained_variance_ratio.npy` - 寄与率
  - `metadata.json` - メタデータ
- `analysis_results/ica/` - ICA結果（同様の構造）
- `analysis_results/*.png` - 可視化グラフ

**可視化される情報:**
- `pca_variance.png` - 各主成分の寄与率
- `pca_distributions.png` - 主成分の分布
- `pca_scatter_matrix.png` - 主成分間の散布図

### ステップ3: 声質編集

#### 方法A: コマンドラインで推論

```bash
python infer_with_editing.py \
    --model_path pre_trained/GenVC_small.pth \
    --src_wav samples/source.wav \
    --ref_audio samples/my_voice.wav \
    --output_path output/edited.wav \
    --method pca \
    --pc1 0.5 \
    --pc2 -0.3 \
    --pc3 0.2
```

**パラメータ説明:**
- `--src_wav` - 変換したい音声（内容）
- `--ref_audio` - あなたの声（ベースとなる声質）
- `--pc1`, `--pc2`, ... - 主成分の係数（±3σの範囲を推奨）
- `--method` - `pca` または `ica`
- `--preserve_norm` - 編集後のノルムを保存（オプション）

#### 方法B: インタラクティブ探索（推奨）

Webインターフェースで対話的に編集を試すことができます。

```bash
python explore_components.py \
    --model_path pre_trained/GenVC_small.pth \
    --method pca \
    --n_components 10 \
    --device cuda
```

ブラウザで http://127.0.0.1:7860 を開き、スライダーで主成分を調整しながらリアルタイムに音声を生成できます。

**インターフェースの使い方:**
1. ソース音声（変換したい内容）をアップロード
2. リファレンス音声（あなたの声）をアップロード
3. スライダーで各主成分を調整
4. "Generate"ボタンをクリック
5. 生成された音声を聴いて確認

## 主成分の意味を理解する

### 典型的な主成分の例

PCA分析では、通常以下のような主成分が発見されます：

- **PC1** (寄与率 ~15-20%): 性別・ピッチの高低
  - 正の方向: 高い声、女性的
  - 負の方向: 低い声、男性的

- **PC2** (寄与率 ~8-12%): 声の明るさ・暗さ
  - 正の方向: 明るい、軽い声
  - 負の方向: 暗い、重い声

- **PC3** (寄与率 ~5-8%): 年齢または声質の特徴
  - 話者データによって異なる

### 主成分の意味を発見する方法

1. **可視化を確認**
   - `pca_distributions.png` で各成分の分布を確認
   - `pca_scatter_matrix.png` で成分間の関係を確認

2. **試聴して確認**
   - `explore_components.py` で1つずつ成分を調整
   - 各成分が知覚的にどう変化するか確認

3. **メモを取る**
   - 各成分の知覚的な意味をメモ
   - 最も有用な5~10個の成分を特定

## プリセットの作成

よく使う編集パターンをプリセットとして保存できます。

```python
from layers.voice_editor import VoiceLatentEditor

editor = VoiceLatentEditor(method="pca")

# プリセット作成
editor.save_editing_preset(
    preset_name="deeper_voice",
    coefficients={"PC1": -2.0, "PC2": -0.5},
    description="Make voice deeper and darker"
)

editor.save_editing_preset(
    preset_name="brighter_voice",
    coefficients={"PC1": 1.0, "PC2": 1.5},
    description="Make voice brighter and lighter"
)
```

プリセットを使用:

```bash
python infer_with_editing.py \
    --model_path pre_trained/GenVC_small.pth \
    --src_wav source.wav \
    --ref_audio my_voice.wav \
    --output_path output.wav \
    --preset deeper_voice
```

## トラブルシューティング

### Q: 分析に時間がかかりすぎる
A: `--max_speakers` オプションで処理する話者数を制限してください。50-100人でも十分な結果が得られます。

### Q: 編集しても変化が少ない
A: 係数を大きくしてみてください（例: ±3.0まで）。または `--preserve_norm` フラグを外してください。

### Q: 編集後の音声が不自然
A: 係数が大きすぎる可能性があります。±3σの範囲内（通常±2.0程度）に抑えてください。

### Q: どの主成分が重要か分からない
A: `explore_components.py` を使って、1つずつ成分を動かして聴いて確認してください。寄与率が高い（PC1, PC2, PC3など）ものから試すのがおすすめです。

## 高度な使い方

### Pythonスクリプトから直接使用

```python
import torch
from inference.model_init import model_init
from inference.inference_utils import synthesize_utt
from layers.voice_editor import VoiceLatentEditor
from utils import load_audio

# モデルとエディタの初期化
model, config = model_init("pre_trained/GenVC_small.pth", "cuda")
editor = VoiceLatentEditor(method="pca", analysis_dir="analysis_results")

# 音声読み込み
src_wav = load_audio("source.wav", model.content_sample_rate)
ref_audio = load_audio("my_voice.wav", config.audio.sample_rate)

# 潜在ベクトル抽出
ref_audio = ref_audio.to(model.device)
cond_latent = model.get_gpt_cond_latents(ref_audio, config.audio.sample_rate)

# 編集
edited_latent = editor.edit(cond_latent, {"PC1": 1.0, "PC2": -0.5})

# 合成
synthesized = synthesize_utt(model, src_wav, cond_latent=edited_latent)

# 保存
import torchaudio
torchaudio.save("output.wav", synthesized.unsqueeze(0).cpu(), config.audio.sample_rate)
```

### バッチ処理

複数の音声を一度に処理:

```python
import glob
from pathlib import Path

src_files = glob.glob("inputs/*.wav")
coefficients = {"PC1": 0.5, "PC2": -0.3}

for src_file in src_files:
    src_wav = load_audio(src_file, model.content_sample_rate)
    synthesized = synthesize_utt(model, src_wav, cond_latent=edited_latent)

    output_path = Path("outputs") / Path(src_file).name
    torchaudio.save(str(output_path), synthesized.unsqueeze(0).cpu(), config.audio.sample_rate)
```

## 実装の詳細

### ファイル構成

```
GenVC/
├── extract_latents.py          # ステップ1: 潜在ベクトル抽出
├── analyze_latent_space.py     # ステップ2: PCA/ICA分析
├── infer_with_editing.py       # ステップ3: 編集付き推論
├── explore_components.py       # インタラクティブ探索UI
├── layers/
│   └── voice_editor.py         # 声質編集モジュール
└── inference/
    └── inference_utils.py      # 修正: cond_latent受け取り対応
```

### データフロー

```
LibriTTS → extract_latents.py → speaker_vectors.npy (N, 1024)
                                        ↓
                            analyze_latent_space.py
                                        ↓
                        PCA/ICA components (50, 1024)
                                        ↓
                                voice_editor.py
                                        ↓
自分の声 → cond_latent → 編集 → edited_latent → GenVC → 出力音声
```

## 今後の拡張案

- [ ] より多様なデータセットでの分析
- [ ] 意味のある成分への自動ラベリング
- [ ] ユーザーフィードバックを用いた半教師あり学習
- [ ] リアルタイム処理の最適化
- [ ] 感情や話し方のスタイル編集

## 参考文献

- PCA: Jolliffe, I. T. (2002). Principal component analysis.
- ICA: Hyvärinen, A., & Oja, E. (2000). Independent component analysis.
- GenVC: [Original GenVC paper](https://arxiv.org/abs/2502.04519)

## ライセンス

このコードは元のGenVCプロジェクトと同じライセンスに従います。
