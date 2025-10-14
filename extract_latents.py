"""
潜在ベクトル抽出スクリプト

LibriTTSデータセットから話者ごとのcond_latentを抽出し、
PCA/ICA分析用のデータセットを作成します。
"""

import torch
import argparse
from pathlib import Path
from inference.model_init import model_init
from utils import load_audio
import numpy as np
from tqdm import tqdm
import json

def extract_speaker_latents(model, audio_paths, sample_rate, max_audios=5):
    """
    1人の話者から複数の発話の潜在ベクトルを抽出して平均化

    Args:
        model: GenVCモデル
        audio_paths: 音声ファイルのパスリスト
        sample_rate: サンプリングレート
        max_audios: 使用する最大音声数（多すぎると時間がかかる）

    Returns:
        speaker_vector: (1024,) - 話者の代表ベクトル（時系列・発話で平均化）
        all_latents: (n_audios, 32, 1024) - 各発話の潜在ベクトル（オプション保存用）
    """
    latents = []

    # 最大max_audios個まで処理
    for audio_path in audio_paths[:max_audios]:
        try:
            audio = load_audio(str(audio_path), sample_rate)
            if audio is None:
                continue

            audio = audio.to(model.device)
            cond_latent = model.get_gpt_cond_latents(audio, sample_rate)  # (1, 32, 1024)
            latents.append(cond_latent.cpu())
        except Exception as e:
            print(f"Warning: Failed to process {audio_path}: {e}")
            continue

    if len(latents) == 0:
        return None, None

    # 全発話をスタック: (n_audios, 1, 32, 1024)
    all_latents = torch.stack(latents)

    # 発話と時系列で平均化: (1024,)
    speaker_vector = all_latents.mean(dim=(0, 1, 2))  # (1024,)

    return speaker_vector.numpy(), all_latents.squeeze(1).numpy()


def collect_libritts_speakers(libritts_path, split='train-clean-100', min_utterances=3):
    """
    LibriTTSから話者IDと音声ファイルのマッピングを作成

    Args:
        libritts_path: LibriTTSデータセットのルートパス
        split: 使用するサブセット
        min_utterances: 最低限必要な発話数

    Returns:
        speakers: {speaker_id: [audio_path1, audio_path2, ...]}
    """
    libritts_root = Path(libritts_path) / split

    if not libritts_root.exists():
        raise ValueError(f"LibriTTS path not found: {libritts_root}")

    speakers = {}

    # LibriTTSの構造: LibriTTS/train-clean-100/speaker_id/chapter_id/*.wav
    for speaker_dir in sorted(libritts_root.iterdir()):
        if not speaker_dir.is_dir():
            continue

        speaker_id = speaker_dir.name
        audio_files = []

        # 各チャプターから音声を収集
        for chapter_dir in speaker_dir.iterdir():
            if not chapter_dir.is_dir():
                continue

            wav_files = list(chapter_dir.glob('*.wav'))
            audio_files.extend(wav_files)

        # 最低限の発話数があるか確認
        if len(audio_files) >= min_utterances:
            speakers[speaker_id] = audio_files

    print(f"Found {len(speakers)} speakers with at least {min_utterances} utterances")
    return speakers


def main():
    parser = argparse.ArgumentParser(description="Extract latent vectors from LibriTTS")
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to GenVC model checkpoint')
    parser.add_argument('--libritts_path', type=str, required=True,
                        help='Path to LibriTTS dataset root')
    parser.add_argument('--split', type=str, default='train-clean-100',
                        choices=['train-clean-100', 'train-clean-360', 'train-other-500',
                                 'dev-clean', 'dev-other', 'test-clean', 'test-other'],
                        help='LibriTTS subset to use')
    parser.add_argument('--output_dir', type=str, default='latent_data',
                        help='Directory to save extracted latents')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--max_speakers', type=int, default=None,
                        help='Maximum number of speakers to process (for testing)')
    parser.add_argument('--max_audios_per_speaker', type=int, default=5,
                        help='Maximum audios per speaker to average')
    parser.add_argument('--min_utterances', type=int, default=3,
                        help='Minimum utterances per speaker')
    parser.add_argument('--save_all_latents', action='store_true',
                        help='Save all individual latents (requires more disk space)')

    args = parser.parse_args()

    # 出力ディレクトリ作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # モデル初期化
    print(f"Loading model from {args.model_path}...")
    model, config = model_init(args.model_path, args.device)
    sample_rate = config.audio.sample_rate

    # LibriTTSから話者データ収集
    print(f"Collecting speakers from {args.libritts_path}/{args.split}...")
    speakers = collect_libritts_speakers(
        args.libritts_path,
        args.split,
        args.min_utterances
    )

    # 最大話者数制限（テスト用）
    if args.max_speakers:
        speaker_ids = list(speakers.keys())[:args.max_speakers]
        speakers = {sid: speakers[sid] for sid in speaker_ids}
        print(f"Limited to {args.max_speakers} speakers for testing")

    # 潜在ベクトル抽出
    print(f"Extracting latent vectors from {len(speakers)} speakers...")
    speaker_vectors = {}
    all_latents_dict = {}
    speaker_metadata = {}

    for speaker_id in tqdm(speakers.keys(), desc="Processing speakers"):
        audio_paths = speakers[speaker_id]

        speaker_vec, all_latents = extract_speaker_latents(
            model,
            audio_paths,
            sample_rate,
            max_audios=args.max_audios_per_speaker
        )

        if speaker_vec is not None:
            speaker_vectors[speaker_id] = speaker_vec

            if args.save_all_latents:
                all_latents_dict[speaker_id] = all_latents

            speaker_metadata[speaker_id] = {
                'num_utterances': len(audio_paths),
                'used_utterances': min(len(audio_paths), args.max_audios_per_speaker),
                'vector_shape': speaker_vec.shape
            }

    print(f"\nSuccessfully extracted {len(speaker_vectors)} speaker vectors")

    # 保存
    # 1. 話者ベクトルの行列 (num_speakers, 1024)
    speaker_ids_list = list(speaker_vectors.keys())
    vectors_matrix = np.stack([speaker_vectors[sid] for sid in speaker_ids_list])

    print(f"Saving speaker vectors matrix: {vectors_matrix.shape}")
    np.save(output_dir / 'speaker_vectors.npy', vectors_matrix)

    # 2. 話者IDのマッピング
    with open(output_dir / 'speaker_ids.json', 'w') as f:
        json.dump(speaker_ids_list, f, indent=2)

    # 3. メタデータ
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump({
            'num_speakers': len(speaker_vectors),
            'vector_dim': 1024,
            'split': args.split,
            'model_path': args.model_path,
            'sample_rate': sample_rate,
            'speakers': speaker_metadata
        }, f, indent=2)

    # 4. 全潜在ベクトル（オプション）
    if args.save_all_latents:
        print("Saving all individual latents...")
        np.savez_compressed(
            output_dir / 'all_latents.npz',
            **all_latents_dict
        )

    print(f"\nData saved to {output_dir}/")
    print(f"  - speaker_vectors.npy: ({vectors_matrix.shape[0]}, {vectors_matrix.shape[1]})")
    print(f"  - speaker_ids.json: {len(speaker_ids_list)} speaker IDs")
    print(f"  - metadata.json: extraction metadata")
    if args.save_all_latents:
        print(f"  - all_latents.npz: individual latent vectors")

    print("\nNext step: Run analyze_latent_space.py to perform PCA/ICA analysis")


if __name__ == '__main__':
    main()
