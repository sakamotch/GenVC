"""
潜在空間分析スクリプト

extract_latents.pyで抽出した話者ベクトルに対して、
PCA/ICAを実行し、主要な声質変動軸を発見します。
"""

import numpy as np
import argparse
import json
from pathlib import Path
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')  # GUI不要のバックエンド
import matplotlib.pyplot as plt
import seaborn as sns


def load_speaker_data(data_dir):
    """抽出済みの話者データを読み込む"""
    data_dir = Path(data_dir)

    speaker_vectors = np.load(data_dir / 'speaker_vectors.npy')
    with open(data_dir / 'speaker_ids.json', 'r') as f:
        speaker_ids = json.load(f)
    with open(data_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    print(f"Loaded {len(speaker_ids)} speakers")
    print(f"Vector dimension: {speaker_vectors.shape[1]}")

    return speaker_vectors, speaker_ids, metadata


def perform_pca(speaker_vectors, n_components=50, standardize=True):
    """
    PCA分析を実行

    Args:
        speaker_vectors: (num_speakers, 1024)
        n_components: 抽出する主成分数
        standardize: 標準化するかどうか

    Returns:
        pca: 学習済みPCAオブジェクト
        transformed: PCA変換後のデータ (num_speakers, n_components)
        scaler: StandardScalerオブジェクト（standardize=Trueの場合）
    """
    scaler = None
    data = speaker_vectors

    if standardize:
        scaler = StandardScaler()
        data = scaler.fit_transform(speaker_vectors)
        print("Applied standardization (zero mean, unit variance)")

    print(f"\nPerforming PCA with {n_components} components...")
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(data)

    # 累積寄与率
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    print(f"Explained variance by top {n_components} components:")
    print(f"  PC1-5:  {cumulative_variance[4]:.4f}")
    print(f"  PC1-10: {cumulative_variance[9]:.4f}")
    print(f"  PC1-20: {cumulative_variance[19]:.4f}")
    print(f"  PC1-{n_components}: {cumulative_variance[-1]:.4f}")

    return pca, transformed, scaler


def perform_ica(speaker_vectors, n_components=20, standardize=True):
    """
    ICA分析を実行

    Args:
        speaker_vectors: (num_speakers, 1024)
        n_components: 抽出する独立成分数
        standardize: 標準化するかどうか

    Returns:
        ica: 学習済みICAオブジェクト
        transformed: ICA変換後のデータ (num_speakers, n_components)
        scaler: StandardScalerオブジェクト（standardize=Trueの場合）
    """
    scaler = None
    data = speaker_vectors

    if standardize:
        scaler = StandardScaler()
        data = scaler.fit_transform(speaker_vectors)
        print("Applied standardization (zero mean, unit variance)")

    print(f"\nPerforming ICA with {n_components} components...")
    ica = FastICA(n_components=n_components, random_state=42, max_iter=1000)
    transformed = ica.fit_transform(data)

    return ica, transformed, scaler


def plot_pca_variance(pca, output_path):
    """PCAの寄与率をプロット"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 個別寄与率
    ax1.bar(range(1, len(pca.explained_variance_ratio_) + 1),
            pca.explained_variance_ratio_,
            alpha=0.7)
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title('Individual Explained Variance')
    ax1.grid(alpha=0.3)

    # 累積寄与率
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    ax2.plot(range(1, len(cumsum) + 1), cumsum, marker='o', markersize=3)
    ax2.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='90%')
    ax2.axhline(y=0.95, color='orange', linestyle='--', alpha=0.5, label='95%')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.set_title('Cumulative Explained Variance')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved variance plot to {output_path}")


def plot_component_distribution(transformed, method_name, output_path, n_top=10):
    """主成分/独立成分の分布をプロット"""
    n_components = min(n_top, transformed.shape[1])

    fig, axes = plt.subplots(2, 5, figsize=(16, 6))
    axes = axes.flatten()

    for i in range(n_components):
        ax = axes[i]
        ax.hist(transformed[:, i], bins=30, alpha=0.7, edgecolor='black')
        ax.set_title(f'{method_name}{i+1}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.grid(alpha=0.3)

        # 統計情報を表示
        mean = transformed[:, i].mean()
        std = transformed[:, i].std()
        ax.axvline(mean, color='r', linestyle='--', alpha=0.5, linewidth=1)
        ax.text(0.02, 0.98, f'μ={mean:.2f}\nσ={std:.2f}',
                transform=ax.transAxes, va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved distribution plot to {output_path}")


def plot_scatter_matrix(transformed, method_name, output_path, n_top=5):
    """主成分間の散布図行列"""
    n_components = min(n_top, transformed.shape[1])
    data = transformed[:, :n_components]

    fig, axes = plt.subplots(n_components, n_components, figsize=(12, 12))

    for i in range(n_components):
        for j in range(n_components):
            ax = axes[i, j]
            if i == j:
                # 対角線上はヒストグラム
                ax.hist(data[:, i], bins=20, alpha=0.7, edgecolor='black')
                ax.set_ylabel('Count' if j == 0 else '')
            else:
                # 散布図
                ax.scatter(data[:, j], data[:, i], alpha=0.3, s=10)
                ax.set_ylabel(f'{method_name}{i+1}' if j == 0 else '')

            if i == n_components - 1:
                ax.set_xlabel(f'{method_name}{j+1}')
            else:
                ax.set_xticklabels([])

            if j != 0:
                ax.set_yticklabels([])

            ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved scatter matrix to {output_path}")


def save_analysis_results(pca, ica, pca_scaler, ica_scaler, output_dir):
    """分析結果を保存"""
    output_dir = Path(output_dir)

    # PCA
    if pca is not None:
        pca_dir = output_dir / 'pca'
        pca_dir.mkdir(exist_ok=True, parents=True)

        np.save(pca_dir / 'components.npy', pca.components_)  # (n_components, 1024)
        np.save(pca_dir / 'mean.npy', pca.mean_)  # (1024,)
        np.save(pca_dir / 'explained_variance.npy', pca.explained_variance_)
        np.save(pca_dir / 'explained_variance_ratio.npy', pca.explained_variance_ratio_)

        if pca_scaler is not None:
            np.save(pca_dir / 'scaler_mean.npy', pca_scaler.mean_)
            np.save(pca_dir / 'scaler_scale.npy', pca_scaler.scale_)

        # メタデータ
        with open(pca_dir / 'metadata.json', 'w') as f:
            json.dump({
                'n_components': pca.n_components_,
                'total_variance': float(pca.explained_variance_.sum()),
                'top5_variance_ratio': float(pca.explained_variance_ratio_[:5].sum()),
                'top10_variance_ratio': float(pca.explained_variance_ratio_[:10].sum()),
                'standardized': pca_scaler is not None
            }, f, indent=2)

        print(f"Saved PCA results to {pca_dir}/")

    # ICA
    if ica is not None:
        ica_dir = output_dir / 'ica'
        ica_dir.mkdir(exist_ok=True, parents=True)

        np.save(ica_dir / 'components.npy', ica.components_)  # (n_components, 1024)
        np.save(ica_dir / 'mixing.npy', ica.mixing_)  # (1024, n_components)
        np.save(ica_dir / 'mean.npy', ica.mean_)  # (1024,)

        if ica_scaler is not None:
            np.save(ica_dir / 'scaler_mean.npy', ica_scaler.mean_)
            np.save(ica_dir / 'scaler_scale.npy', ica_scaler.scale_)

        # メタデータ
        with open(ica_dir / 'metadata.json', 'w') as f:
            json.dump({
                'n_components': ica.n_components,
                'standardized': ica_scaler is not None,
                'max_iter': ica.max_iter,
                'converged': bool(ica.n_iter_ < ica.max_iter)
            }, f, indent=2)

        print(f"Saved ICA results to {ica_dir}/")


def print_component_statistics(pca, ica, pca_transformed, ica_transformed):
    """成分の統計情報を出力"""
    print("\n" + "="*60)
    print("COMPONENT STATISTICS")
    print("="*60)

    if pca is not None:
        print("\nPCA Components:")
        print("-" * 60)
        print(f"{'Component':<12} {'Variance':<12} {'Cumulative':<12} {'Std Dev'}")
        print("-" * 60)
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        for i in range(min(10, len(pca.explained_variance_ratio_))):
            std = pca_transformed[:, i].std()
            print(f"PC{i+1:<10} {pca.explained_variance_ratio_[i]:<12.4f} "
                  f"{cumsum[i]:<12.4f} {std:<.4f}")

    if ica is not None:
        print("\nICA Components:")
        print("-" * 60)
        print(f"{'Component':<12} {'Mean':<12} {'Std Dev':<12}")
        print("-" * 60)
        for i in range(min(10, ica_transformed.shape[1])):
            mean = ica_transformed[:, i].mean()
            std = ica_transformed[:, i].std()
            print(f"IC{i+1:<10} {mean:<12.4f} {std:<12.4f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze latent space with PCA/ICA")
    parser.add_argument('--data_dir', type=str, default='latent_data',
                        help='Directory containing extracted latents')
    parser.add_argument('--output_dir', type=str, default='analysis_results',
                        help='Directory to save analysis results')
    parser.add_argument('--n_pca_components', type=int, default=50,
                        help='Number of PCA components to extract')
    parser.add_argument('--n_ica_components', type=int, default=20,
                        help='Number of ICA components to extract')
    parser.add_argument('--standardize', action='store_true',
                        help='Standardize data before analysis')
    parser.add_argument('--skip_pca', action='store_true',
                        help='Skip PCA analysis')
    parser.add_argument('--skip_ica', action='store_true',
                        help='Skip ICA analysis')

    args = parser.parse_args()

    # 出力ディレクトリ作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # データ読み込み
    print("Loading speaker data...")
    speaker_vectors, speaker_ids, metadata = load_speaker_data(args.data_dir)

    # PCA分析
    pca, pca_transformed, pca_scaler = None, None, None
    if not args.skip_pca:
        pca, pca_transformed, pca_scaler = perform_pca(
            speaker_vectors,
            n_components=args.n_pca_components,
            standardize=args.standardize
        )

        # PCA可視化
        plot_pca_variance(pca, output_dir / 'pca_variance.png')
        plot_component_distribution(pca_transformed, 'PC', output_dir / 'pca_distributions.png')
        plot_scatter_matrix(pca_transformed, 'PC', output_dir / 'pca_scatter_matrix.png')

    # ICA分析
    ica, ica_transformed, ica_scaler = None, None, None
    if not args.skip_ica:
        ica, ica_transformed, ica_scaler = perform_ica(
            speaker_vectors,
            n_components=args.n_ica_components,
            standardize=args.standardize
        )

        # ICA可視化
        plot_component_distribution(ica_transformed, 'IC', output_dir / 'ica_distributions.png')
        plot_scatter_matrix(ica_transformed, 'IC', output_dir / 'ica_scatter_matrix.png')

    # 統計情報出力
    print_component_statistics(pca, ica, pca_transformed, ica_transformed)

    # 結果保存
    save_analysis_results(pca, ica, pca_scaler, ica_scaler, output_dir)

    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"Results saved to {output_dir}/")
    print("\nNext steps:")
    print("1. Review the plots to understand component distributions")
    print("2. Use infer_with_editing.py to test voice editing with these components")
    print("3. Use explore_components.py for interactive listening")


if __name__ == '__main__':
    main()
