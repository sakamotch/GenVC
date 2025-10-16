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

### 3. LibriTTSデータセットの選択

| データセット | サイズ | 話者数 | 推奨度 | 用途 |
|------------|--------|--------|--------|------|
| dev-clean | 1.2GB | ~40人 | ⭐⭐⭐⭐ | **クイックテスト・動作確認** |
| train-clean-100 | 7.7GB | ~251人 | ⭐⭐⭐⭐⭐ | **実用レベル（推奨）** |
| train-clean-360 | 27GB | ~921人 | ⭐⭐⭐⭐ | 高品質・研究用 |

**推奨:** まず **dev-clean** で動作確認してから **train-clean-100** で本番実行

## クイックスタート: dev-cleanで動作確認

初めて使う場合は、まず小規模なdev-cleanデータセットで動作確認することを推奨します。

### 準備: ディレクトリ構造

```bash
cd /path/to/GenVC

# データ用ディレクトリを作成
mkdir -p data/LibriTTS
mkdir -p latent_data
mkdir -p analysis_results
mkdir -p outputs
```

### ステップ1: dev-cleanのダウンロードと展開

```bash
# データディレクトリに移動
cd data/LibriTTS

# ダウンロード（約1.2GB、5-10分）
wget https://www.openslr.org/resources/60/dev-clean.tar.gz

# 展開
tar -xzf dev-clean.tar.gz

# 構造確認
ls -la dev-clean/
# dev-clean/1272/, dev-clean/1673/, ... (話者IDディレクトリ)
```

### ステップ2: 潜在ベクトル抽出（30分-1時間）

```bash
# プロジェクトルートに戻る
cd ../..

# dev-cleanから潜在ベクトルを抽出
python extract_latents.py \
    --model_path pre_trained/GenVC_small.pth \
    --libritts_path data/LibriTTS \
    --split dev-clean \
    --output_dir latent_data \
    --device cuda \
    --max_audios_per_speaker 5 \
    --min_utterances 3
```

**期待される結果:**
- 処理時間: 30分-1時間（GPU使用時）
- 話者数: 約30-40人
- 出力ファイル:
  - `latent_data/speaker_vectors.npy` (40, 1024)
  - `latent_data/speaker_ids.json`
  - `latent_data/metadata.json`

### ステップ3: PCA/ICA分析（5-10分）

```bash
python analyze_latent_space.py \
    --data_dir latent_data \
    --output_dir analysis_results \
    --n_pca_components 20 \
    --n_ica_components 10 \
    --standardize
```

**出力ファイル:**
- `analysis_results/pca/components.npy`
- `analysis_results/pca_variance.png` - 寄与率グラフ
- `analysis_results/pca_distributions.png` - 主成分の分布
- `analysis_results/pca_scatter_matrix.png` - 散布図

**可視化を確認:**
```bash
# Windowsの場合
explorer.exe analysis_results/

# 重要なグラフ:
# - pca_variance.png: PC1-5で50%以上の分散を説明できていればOK
# - pca_distributions.png: 各主成分の分布形状
```

### ステップ4: インタラクティブ探索

```bash
python explore_components.py \
    --model_path pre_trained/GenVC_small.pth \
    --method pca \
    --n_components 10 \
    --device cuda
```

ブラウザで http://127.0.0.1:7860 を開く

**試すこと:**
1. サンプル音声をアップロード
2. PC1を -3 → +3 に動かす（通常：性別・ピッチ）
3. PC2を -3 → +3 に動かす（通常：明るさ）
4. 各主成分が何を制御しているかメモを取る

### ステップ5: コマンドラインでテスト

```bash
# PC1のみ調整（高い声）
python infer_with_editing.py \
    --model_path pre_trained/GenVC_small.pth \
    --src_wav samples/EF4_ENG_0112_1.wav \
    --ref_audio samples/EM1_ENG_0037_1.wav \
    --output_path outputs/test_high_pitch.wav \
    --method pca \
    --pc1 1.0

# PC1のみ調整（低い声）
python infer_with_editing.py \
    --model_path pre_trained/GenVC_small.pth \
    --src_wav samples/EF4_ENG_0112_1.wav \
    --ref_audio samples/EM1_ENG_0037_1.wav \
    --output_path outputs/test_low_pitch.wav \
    --method pca \
    --pc1 -1.0

# 複数成分を組み合わせ
python infer_with_editing.py \
    --model_path pre_trained/GenVC_small.pth \
    --src_wav samples/EF4_ENG_0112_1.wav \
    --ref_audio samples/EM1_ENG_0037_1.wav \
    --output_path outputs/test_combined.wav \
    --method pca \
    --pc1 0.5 --pc2 -0.3 --pc3 0.2
```

### dev-cleanの結果評価

**良好な結果の目安:**
- PC1寄与率: 12-18%
- PC1-5累積寄与率: 45-60%
- PC1-10累積寄与率: 65-80%
- 声質の変化が知覚的に明確

**結果が良好な場合:**
→ train-clean-100で本番実行（より精度の高い編集が可能）

**結果が微妙な場合:**
- 係数を大きくする（±3.0まで試す）
- `--standardize` フラグを外してみる
- または train-clean-100 に移行

### train-clean-100への移行

dev-cleanで動作確認できたら、より多くの話者で精度向上:

```bash
# ダウンロード（7.7GB）
cd data/LibriTTS
wget https://www.openslr.org/resources/60/train-clean-100.tar.gz
tar -xzf train-clean-100.tar.gz

# 抽出（2-4時間）
cd ../..
python extract_latents.py \
    --model_path pre_trained/GenVC_small.pth \
    --libritts_path data/LibriTTS \
    --split train-clean-100 \
    --output_dir latent_data_100 \
    --device cuda

# 分析
python analyze_latent_space.py \
    --data_dir latent_data_100 \
    --output_dir analysis_results_100 \
    --n_pca_components 50 \
    --standardize
```

## 使用方法（詳細）

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

### データ抽出に関する問題

#### Q: 話者数が少ない（<20人）

**原因:** `--min_utterances` が高すぎる、またはデータセットが小さい

**解決策:**
```bash
python extract_latents.py \
    --min_utterances 2  # 3から2に下げる
    # または
    --max_speakers 50  # より多くの話者を処理
```

#### Q: 処理が非常に遅い

**原因:** CPUを使っている、または1話者あたりの音声が多すぎる

**解決策:**
```bash
python extract_latents.py \
    --device cuda \  # GPUを使用
    --max_audios_per_speaker 3  # 5から3に減らす
```

#### Q: メモリ不足エラー

**原因:** バッチサイズが大きすぎる、またはGPUメモリ不足

**解決策:**
```bash
python extract_latents.py \
    --max_speakers 30  # 最大30人に制限してテスト
```

#### Q: エラー: "Failed to process [audio_path]"

**原因:** 音声ファイルが破損、または短すぎる

**解決策:**
- 警告として表示されるだけで、他の音声は処理されます
- 多数のエラーが出る場合は `--min_utterances` を調整

### 分析に関する問題

#### Q: 分析に時間がかかりすぎる

**解決策:** `--max_speakers` オプションで処理する話者数を制限してください。50-100人でも十分な結果が得られます。

#### Q: PCAの寄与率が低い（PC1-10で50%未満）

**原因:** データセットの話者数が少ない、または話者の多様性が不足

**解決策:**
1. より大きなデータセット（train-clean-100）を使用
2. `--standardize` フラグの有無を試す
3. dev-cleanの場合は正常（話者数が少ないため）

#### Q: 可視化グラフが表示されない

**原因:** matplotlib のバックエンド問題

**解決策:**
```bash
# グラフファイルは保存されているので、画像ビューアで開く
ls analysis_results/*.png
```

### 音声編集に関する問題

#### Q: 編集しても変化が少ない

**原因:** 係数が小さすぎる、または話者データが少ない

**解決策:**
1. 係数を大きくする（±3.0まで試す）
2. `--preserve_norm` フラグを外す
3. より大きなデータセット（train-clean-100）で再分析

```bash
python infer_with_editing.py \
    --pc1 3.0  # 係数を大きくする
    # --preserve_norm を削除
```

#### Q: 編集後の音声が不自然・歪む

**原因:** 係数が大きすぎる

**解決策:**
- ±3σの範囲内（通常±2.0程度）に抑える
- 複数の成分を同時に大きく動かしすぎない
- `--preserve_norm` を試す

#### Q: どの主成分が重要か分からない

**解決策:**
1. `explore_components.py` を使って、1つずつ成分を動かして聴く
2. 寄与率が高い（PC1, PC2, PC3など）ものから試す
3. メモを取りながら各成分の知覚的な効果を記録

```bash
# 各成分を単独でテスト
python infer_with_editing.py --pc1 2.0 --output_path test_pc1.wav
python infer_with_editing.py --pc2 2.0 --output_path test_pc2.wav
python infer_with_editing.py --pc3 2.0 --output_path test_pc3.wav
```

#### Q: ICAとPCA、どちらを使うべきか

**推奨:** まずPCAから試す

**理由:**
- PCAは寄与率が明確で解釈しやすい
- ICAは統計的に独立な成分を見つけるが、重要度の順序付けがない

**使い分け:**
- PCA: 全体的な変動を理解したい場合
- ICA: より独立した声質特徴を発見したい場合

### Gradioインターフェースに関する問題

#### Q: ブラウザで開けない

**解決策:**
```bash
# ポートを変更
python explore_components.py --port 8080

# または別のマシンからアクセス
python explore_components.py --share  # 公開URLを生成
```

#### Q: 音声生成が遅い

**原因:** CPUで実行している、またはモデルが大きい

**解決策:**
```bash
python explore_components.py --device cuda
# または GenVC_small.pth を使用
```

#### Q: アップロードした音声がエラーになる

**原因:** サンプリングレートが極端、またはステレオ音声

**解決策:**
- モノラル音声を使用
- サンプリングレート: 16kHz-48kHzを推奨
- 形式: WAVファイルを推奨

### その他の問題

#### Q: CUDA out of memory エラー

**解決策:**
```bash
# より小さいモデルを使用
--model_path pre_trained/GenVC_small.pth

# バッチサイズを減らす（extract_latents.pyの場合）
--max_audios_per_speaker 3
```

#### Q: 結果の再現性がない

**原因:** GenVCモデル自体がサンプリングベース

**解決策:**
```bash
# top_kを1に設定（greedy decoding）
python infer_with_editing.py --top_k 1
```

#### Q: 自分の声で試したら期待と違う結果になる

**原因:**
1. 自分の声が学習データの分布外
2. 参照音声が短すぎる
3. 声質編集の方向が期待と異なる

**解決策:**
1. より長い参照音声を使用（3-5秒以上）
2. 複数の自分の発話で試す
3. `explore_components.py` で試行錯誤して最適な係数を見つける

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
