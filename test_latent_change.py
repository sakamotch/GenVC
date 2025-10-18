"""
潜在ベクトルが本当に変化しているか確認するスクリプト
"""

import torch
from inference.model_init import model_init
from layers.voice_editor import VoiceLatentEditor
from utils import load_audio

# モデル初期化
model, config = model_init("pre_trained/GenVC_small.pth", "cuda")
editor = VoiceLatentEditor(method="pca", analysis_dir="analysis_results")

# 音声読み込み
ref_audio = load_audio("samples/EM1_ENG_0037_1.wav", config.audio.sample_rate)
ref_audio = ref_audio.to(model.device)

# 条件付き潜在表現を抽出
cond_latent = model.get_gpt_cond_latents(ref_audio, config.audio.sample_rate)

print("=" * 70)
print("Original cond_latent:")
print(f"  Shape: {cond_latent.shape}")
print(f"  Range: [{cond_latent.min():.3f}, {cond_latent.max():.3f}]")
print(f"  Mean: {cond_latent.mean():.3f}")
print(f"  Std: {cond_latent.std():.3f}")

# 各時刻の統計
print(f"\nPer-timestep statistics (32 timesteps):")
for t in range(min(5, cond_latent.shape[1])):  # 最初の5時刻
    vec = cond_latent[0, t, :]
    print(f"  t={t}: mean={vec.mean():.3f}, std={vec.std():.3f}, "
          f"range=[{vec.min():.3f}, {vec.max():.3f}]")

print("\n" + "=" * 70)

# 様々な係数で編集
test_coefficients = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]

for coef in test_coefficients:
    edited_latent = editor.edit(cond_latent, {"PC1": coef})

    diff = edited_latent - cond_latent
    diff_norm = torch.norm(diff).item()
    diff_mean = diff.mean().item()
    diff_max = diff.abs().max().item()

    print(f"\nPC1 = {coef}:")
    print(f"  Difference norm: {diff_norm:.6f}")
    print(f"  Difference mean: {diff_mean:.6f}")
    print(f"  Difference max (abs): {diff_max:.6f}")
    print(f"  Edited range: [{edited_latent.min():.3f}, {edited_latent.max():.3f}]")

    # 変化率
    relative_change = (diff_norm / torch.norm(cond_latent).item()) * 100
    print(f"  Relative change: {relative_change:.2f}%")

print("\n" + "=" * 70)
print("Analysis:")
print("- If 'Difference norm' is 0 for all coefficients → PCA not working")
print("- If 'Difference norm' increases with coefficient → PCA working")
print("- If 'Relative change' < 1% → change may be too small to affect generation")
