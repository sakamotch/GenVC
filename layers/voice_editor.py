"""
声質編集モジュール

PCA/ICAで発見した主成分を使って、cond_latentを編集します。
"""

import torch
import numpy as np
from pathlib import Path
import json


class VoiceLatentEditor:
    """
    潜在空間での声質編集クラス

    使い方:
        editor = VoiceLatentEditor(method="pca", analysis_dir="analysis_results")
        edited_latent = editor.edit(cond_latent, {"PC1": 0.5, "PC2": -0.3})
    """

    def __init__(self, method="pca", analysis_dir="analysis_results"):
        """
        Args:
            method: "pca" or "ica"
            analysis_dir: analyze_latent_space.pyの出力ディレクトリ
        """
        self.method = method.lower()
        assert self.method in ["pca", "ica"], f"Unknown method: {method}"

        analysis_path = Path(analysis_dir) / self.method

        if not analysis_path.exists():
            raise ValueError(f"Analysis directory not found: {analysis_path}")

        # コンポーネント読み込み
        self.components = torch.from_numpy(
            np.load(analysis_path / 'components.npy')
        ).float()  # (n_components, 1024)

        self.mean = torch.from_numpy(
            np.load(analysis_path / 'mean.npy')
        ).float()  # (1024,)

        # 標準化パラメータ（オプション）
        scaler_mean_path = analysis_path / 'scaler_mean.npy'
        scaler_scale_path = analysis_path / 'scaler_scale.npy'

        if scaler_mean_path.exists() and scaler_scale_path.exists():
            self.scaler_mean = torch.from_numpy(np.load(scaler_mean_path)).float()
            self.scaler_scale = torch.from_numpy(np.load(scaler_scale_path)).float()
            self.standardized = True
        else:
            self.scaler_mean = None
            self.scaler_scale = None
            self.standardized = False

        # PCAの場合は分散も読み込む
        if self.method == "pca":
            self.explained_variance = torch.from_numpy(
                np.load(analysis_path / 'explained_variance.npy')
            ).float()
            self.explained_variance_ratio = torch.from_numpy(
                np.load(analysis_path / 'explained_variance_ratio.npy')
            ).float()
        else:
            self.explained_variance = None
            self.explained_variance_ratio = None

        # メタデータ
        with open(analysis_path / 'metadata.json', 'r') as f:
            self.metadata = json.load(f)

        self.n_components = self.components.shape[0]

        print(f"Loaded {self.method.upper()} editor:")
        print(f"  - Components: {self.n_components}")
        print(f"  - Dimension: {self.components.shape[1]}")
        print(f"  - Standardized: {self.standardized}")
        if self.method == "pca":
            print(f"  - Total variance explained: {self.metadata['top10_variance_ratio']:.4f} (top 10)")

    def edit(self, cond_latent, coefficients, preserve_norm=False, clip_to_range=False):
        """
        潜在ベクトルを編集

        Args:
            cond_latent: (1, 32, 1024) or (B, 32, 1024) - 元の条件付き潜在表現
            coefficients: 編集係数
                - dict形式: {"PC1": 0.5, "PC2": -0.3, ...} または {"IC1": 0.5, ...}
                - list形式: [0.5, -0.3, 0.0, ...] (長さはn_componentsまで)
                - float: 単一成分のみ編集 (PC1/IC1のみ)
            preserve_norm: Trueの場合、編集後のノルムを元のノルムに正規化
            clip_to_range: Trueの場合、編集後の値を妥当な範囲にクリップ

        Returns:
            edited_latent: (1, 32, 1024) or (B, 32, 1024) - 編集後の潜在表現
        """
        device = cond_latent.device
        batch_size = cond_latent.shape[0]

        # コンポーネントをデバイスに移動
        components = self.components.to(device)
        mean = self.mean.to(device) if self.mean is not None else None

        # 係数をテンソルに変換
        if isinstance(coefficients, dict):
            coef_list = []
            prefix = "PC" if self.method == "pca" else "IC"
            for i in range(self.n_components):
                key = f"{prefix}{i+1}"
                coef_list.append(coefficients.get(key, 0.0))
            coef_tensor = torch.tensor(coef_list, dtype=torch.float32, device=device)

        elif isinstance(coefficients, (list, tuple)):
            # 不足分は0で埋める
            coef_list = list(coefficients) + [0.0] * (self.n_components - len(coefficients))
            coef_tensor = torch.tensor(coef_list[:self.n_components], dtype=torch.float32, device=device)

        elif isinstance(coefficients, (int, float)):
            # 単一の値 = PC1/IC1のみ編集
            coef_tensor = torch.zeros(self.n_components, dtype=torch.float32, device=device)
            coef_tensor[0] = float(coefficients)

        else:
            raise ValueError(f"Invalid coefficients type: {type(coefficients)}")

        # 元のノルムを保存（オプション）
        if preserve_norm:
            original_norm = torch.norm(cond_latent, dim=-1, keepdim=True)  # (B, 32, 1)

        # 編集ベクトルを計算: edit_vector = Σ(coef_i × component_i)
        edit_vector = torch.matmul(coef_tensor, components)  # (1024,)

        # デバッグ情報
        print(f"[VoiceEditor] coef_tensor: {coef_tensor[:5].tolist()}")  # 最初の5要素
        print(f"[VoiceEditor] edit_vector range BEFORE scaling: [{edit_vector.min():.3f}, {edit_vector.max():.3f}]")
        print(f"[VoiceEditor] standardized: {self.standardized}, scaler exists: {self.scaler_scale is not None}")

        # 標準化されている場合、逆変換を適用
        if self.standardized and self.scaler_scale is not None:
            scaler_scale = self.scaler_scale.to(device)
            edit_vector = edit_vector * scaler_scale  # スケールを戻す
            print(f"[VoiceEditor] edit_vector range AFTER scaling: [{edit_vector.min():.3f}, {edit_vector.max():.3f}]")
            # 注意: 平均は編集ベクトル（差分）なので加算しない

        # cond_latentに編集を適用
        # cond_latent: (B, 32, 1024)
        # edit_vector: (1024,) → (1, 1, 1024) にreshapeしてブロードキャスト
        edited_latent = cond_latent + edit_vector.view(1, 1, -1)

        # クリッピング（オプション）
        # 編集なし（edit_vectorが0）の場合はクリッピングしない
        edit_norm = torch.norm(edit_vector).item()
        if clip_to_range and edit_norm > 1e-6:  # 編集がある場合のみクリップ
            # 元のデータの範囲を基準にクリップ
            # より緩い範囲（元の範囲の1.5倍）を使用
            original_min = cond_latent.min()
            original_max = cond_latent.max()
            margin = (original_max - original_min) * 0.25
            clip_min = original_min - margin
            clip_max = original_max + margin
            edited_latent = torch.clamp(edited_latent, clip_min, clip_max)
            print(f"[VoiceEditor] Clipped to range: [{clip_min:.3f}, {clip_max:.3f}]")

        # ノルム保存（オプション）
        if preserve_norm:
            edited_norm = torch.norm(edited_latent, dim=-1, keepdim=True)
            edited_latent = edited_latent * (original_norm / (edited_norm + 1e-8))

        return edited_latent

    def get_component_info(self, component_idx):
        """
        指定した成分の情報を取得

        Args:
            component_idx: 成分のインデックス (0-indexed)

        Returns:
            info: dict - 成分の情報
        """
        if component_idx >= self.n_components:
            raise ValueError(f"Component index {component_idx} out of range (max: {self.n_components-1})")

        info = {
            'index': component_idx,
            'name': f"{'PC' if self.method == 'pca' else 'IC'}{component_idx + 1}",
            'shape': tuple(self.components[component_idx].shape)
        }

        if self.method == "pca" and self.explained_variance is not None:
            info['explained_variance'] = float(self.explained_variance[component_idx])
            info['explained_variance_ratio'] = float(self.explained_variance_ratio[component_idx])
            info['std'] = float(torch.sqrt(self.explained_variance[component_idx]))

        return info

    def get_editing_range(self, component_idx, n_std=3.0):
        """
        推奨される編集範囲を取得

        Args:
            component_idx: 成分のインデックス
            n_std: 標準偏差の倍数（デフォルト3σ）

        Returns:
            (min_val, max_val): 推奨範囲
        """
        if self.method == "pca" and self.explained_variance is not None:
            std = float(torch.sqrt(self.explained_variance[component_idx]))
            return -n_std * std, n_std * std
        else:
            # ICAまたは分散情報がない場合は固定範囲
            return -n_std, n_std

    def get_top_components_summary(self, n_top=10):
        """
        上位N個の成分のサマリーを表示

        Args:
            n_top: 表示する成分数
        """
        n_top = min(n_top, self.n_components)

        print(f"\nTop {n_top} {self.method.upper()} Components:")
        print("-" * 70)

        if self.method == "pca":
            print(f"{'Component':<12} {'Variance':<12} {'Cumulative':<12} {'Range (±3σ)'}")
            print("-" * 70)
            cumsum = 0.0
            for i in range(n_top):
                var_ratio = float(self.explained_variance_ratio[i])
                cumsum += var_ratio
                min_val, max_val = self.get_editing_range(i)
                print(f"PC{i+1:<10} {var_ratio:<12.4f} {cumsum:<12.4f} "
                      f"[{min_val:>6.2f}, {max_val:>6.2f}]")
        else:
            print(f"{'Component':<12} {'Recommended Range'}")
            print("-" * 70)
            for i in range(n_top):
                min_val, max_val = self.get_editing_range(i)
                print(f"IC{i+1:<10} [{min_val:>6.2f}, {max_val:>6.2f}]")

    def save_editing_preset(self, preset_name, coefficients, description=""):
        """
        編集プリセットを保存

        Args:
            preset_name: プリセット名
            coefficients: 係数のdict
            description: 説明文
        """
        presets_dir = Path("editing_presets")
        presets_dir.mkdir(exist_ok=True)

        preset_data = {
            'name': preset_name,
            'method': self.method,
            'coefficients': coefficients,
            'description': description
        }

        preset_path = presets_dir / f"{preset_name}.json"
        with open(preset_path, 'w') as f:
            json.dump(preset_data, f, indent=2)

        print(f"Saved preset to {preset_path}")

    @staticmethod
    def load_editing_preset(preset_name):
        """
        編集プリセットを読み込み

        Args:
            preset_name: プリセット名

        Returns:
            preset_data: dict
        """
        preset_path = Path("editing_presets") / f"{preset_name}.json"

        if not preset_path.exists():
            raise ValueError(f"Preset not found: {preset_path}")

        with open(preset_path, 'r') as f:
            preset_data = json.load(f)

        return preset_data


if __name__ == '__main__':
    # テスト用コード
    print("Testing VoiceLatentEditor...")

    # ダミーデータでテスト
    dummy_latent = torch.randn(1, 32, 1024)

    try:
        editor = VoiceLatentEditor(method="pca", analysis_dir="analysis_results")
        editor.get_top_components_summary(n_top=5)

        # 編集テスト
        edited = editor.edit(dummy_latent, {"PC1": 1.0, "PC2": -0.5})
        print(f"\nOriginal shape: {dummy_latent.shape}")
        print(f"Edited shape: {edited.shape}")
        print(f"Difference norm: {torch.norm(edited - dummy_latent):.4f}")

    except Exception as e:
        print(f"Note: {e}")
        print("Run analyze_latent_space.py first to generate analysis results")
