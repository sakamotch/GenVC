"""
声質編集付き推論スクリプト

PCA/ICAで発見した主成分を使って、自分の声を任意の方向に調整して変換します。

使用例:
    # 基本的な使用方法
    python infer_with_editing.py \\
        --model_path pre_trained/GenVC_small.pth \\
        --src_wav samples/source.wav \\
        --ref_audio samples/my_voice.wav \\
        --output_path output/edited.wav \\
        --pc1 0.5 --pc2 -0.3

    # プリセットを使用
    python infer_with_editing.py \\
        --model_path pre_trained/GenVC_small.pth \\
        --src_wav samples/source.wav \\
        --ref_audio samples/my_voice.wav \\
        --output_path output/edited.wav \\
        --preset deeper_voice

    # ICAを使用
    python infer_with_editing.py \\
        --model_path pre_trained/GenVC_small.pth \\
        --src_wav samples/source.wav \\
        --ref_audio samples/my_voice.wav \\
        --output_path output/edited.wav \\
        --method ica \\
        --ic1 1.0 --ic2 0.5
"""

import argparse
import torch
import torchaudio
from pathlib import Path
from inference.model_init import model_init
from inference.inference_utils import synthesize_utt, synthesize_utt_streaming
from layers.voice_editor import VoiceLatentEditor
from utils import load_audio


def parse_component_args(args, method="pca", max_components=20):
    """
    コマンドライン引数から主成分係数を抽出

    Args:
        args: argparse.Namespace
        method: "pca" or "ica"
        max_components: 最大成分数

    Returns:
        coefficients: dict - {"PC1": 0.5, ...} または {"IC1": 0.5, ...}
    """
    coefficients = {}
    prefix = "pc" if method == "pca" else "ic"

    for i in range(1, max_components + 1):
        arg_name = f"{prefix}{i}"
        if hasattr(args, arg_name):
            value = getattr(args, arg_name)
            if value != 0.0:  # 0以外のみ追加
                component_name = f"{'PC' if method == 'pca' else 'IC'}{i}"
                coefficients[component_name] = value

    return coefficients


def main():
    parser = argparse.ArgumentParser(
        description="Voice conversion with latent space editing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # 基本パラメータ
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to GenVC model checkpoint')
    parser.add_argument('--src_wav', type=str, required=True,
                        help='Source audio file path')
    parser.add_argument('--ref_audio', type=str, required=True,
                        help='Reference audio file path (your voice)')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output audio file path')

    # 編集パラメータ
    parser.add_argument('--method', type=str, default='pca', choices=['pca', 'ica'],
                        help='Analysis method to use')
    parser.add_argument('--analysis_dir', type=str, default='analysis_results',
                        help='Directory containing analysis results')
    parser.add_argument('--preset', type=str, default=None,
                        help='Load editing preset by name')

    # PCA係数 (PC1-PC20)
    for i in range(1, 21):
        parser.add_argument(f'--pc{i}', type=float, default=0.0,
                            help=f'PCA component {i} coefficient')

    # ICA係数 (IC1-IC20)
    for i in range(1, 21):
        parser.add_argument(f'--ic{i}', type=float, default=0.0,
                            help=f'ICA component {i} coefficient')

    # 推論オプション
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--top_k', type=int, default=15,
                        help='Top-k sampling parameter')
    parser.add_argument('--streaming', action='store_true',
                        help='Use streaming inference')
    parser.add_argument('--preserve_norm', action='store_true',
                        help='Preserve latent vector norm after editing')

    # デバッグ/詳細表示
    parser.add_argument('--show_info', action='store_true',
                        help='Show component information')
    parser.add_argument('--save_latent', type=str, default=None,
                        help='Save edited latent to file (for debugging)')

    args = parser.parse_args()

    # 出力ディレクトリ作成
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("GenVC with Voice Latent Editing")
    print("="*70)

    # モデル初期化
    print(f"\n[1/5] Loading model from {args.model_path}...")
    model, config = model_init(args.model_path, args.device)
    model.config.top_k = args.top_k

    # 声質編集モジュール初期化
    print(f"\n[2/5] Loading {args.method.upper()} editor from {args.analysis_dir}...")
    try:
        editor = VoiceLatentEditor(method=args.method, analysis_dir=args.analysis_dir)
    except Exception as e:
        print(f"Error loading editor: {e}")
        print("\nPlease run the following steps first:")
        print("  1. python extract_latents.py --model_path <model> --libritts_path <path>")
        print("  2. python analyze_latent_space.py --data_dir latent_data")
        return

    # 成分情報表示
    if args.show_info:
        editor.get_top_components_summary(n_top=10)

    # 編集係数取得
    print(f"\n[3/5] Preparing editing coefficients...")
    if args.preset:
        # プリセット読み込み
        print(f"Loading preset: {args.preset}")
        preset_data = VoiceLatentEditor.load_editing_preset(args.preset)
        coefficients = preset_data['coefficients']
        print(f"Description: {preset_data.get('description', 'N/A')}")
    else:
        # コマンドライン引数から取得
        coefficients = parse_component_args(args, method=args.method, max_components=20)

    if not coefficients:
        print("Warning: No editing coefficients specified. Using identity transformation.")
    else:
        print("Editing coefficients:")
        for comp, val in sorted(coefficients.items()):
            print(f"  {comp}: {val:+.3f}")

    # 音声読み込み
    print(f"\n[4/5] Loading audio files...")
    src_wav = load_audio(args.src_wav, model.content_sample_rate)
    ref_audio = load_audio(args.ref_audio, config.audio.sample_rate)

    if src_wav is None:
        print(f"Error: Failed to load source audio: {args.src_wav}")
        return
    if ref_audio is None:
        print(f"Error: Failed to load reference audio: {args.ref_audio}")
        return

    print(f"Source audio: {src_wav.shape[-1] / model.content_sample_rate:.2f}s")
    print(f"Reference audio: {ref_audio.shape[-1] / config.audio.sample_rate:.2f}s")

    # 条件付き潜在表現を抽出
    ref_audio_tensor = ref_audio.to(model.device)
    cond_latent = model.get_gpt_cond_latents(ref_audio_tensor, config.audio.sample_rate)
    print(f"Extracted cond_latent: {cond_latent.shape}")

    # 声質編集
    edited_latent = editor.edit(cond_latent, coefficients, preserve_norm=args.preserve_norm)
    edit_magnitude = torch.norm(edited_latent - cond_latent).item()
    print(f"Edit magnitude: {edit_magnitude:.4f}")

    # 編集後の潜在ベクトル保存（オプション）
    if args.save_latent:
        torch.save({
            'original': cond_latent.cpu(),
            'edited': edited_latent.cpu(),
            'coefficients': coefficients,
            'method': args.method
        }, args.save_latent)
        print(f"Saved latent vectors to {args.save_latent}")

    # 音声合成
    print(f"\n[5/5] Synthesizing audio...")
    if args.streaming:
        print("Using streaming mode...")
        synthesized = synthesize_utt_streaming(
            model,
            src_wav,
            cond_latent=edited_latent
        )
    else:
        synthesized = synthesize_utt(
            model,
            src_wav,
            cond_latent=edited_latent
        )

    # 保存
    torchaudio.save(
        args.output_path,
        synthesized.unsqueeze(0).detach().cpu(),
        config.audio.sample_rate
    )

    print(f"\n{'='*70}")
    print(f"Success! Output saved to: {args.output_path}")
    print(f"Duration: {synthesized.shape[-1] / config.audio.sample_rate:.2f}s")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
