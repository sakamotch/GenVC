"""
インタラクティブ声質編集ツール (Gradio)

Webインターフェースで主成分を調整しながらリアルタイムに声質編集を試せます。

使用方法:
    python explore_components.py \\
        --model_path pre_trained/GenVC_small.pth \\
        --method pca \\
        --n_components 10

    ブラウザで http://127.0.0.1:7860 を開く
"""

import argparse
import torch
import torchaudio
import gradio as gr
from pathlib import Path
import numpy as np
from inference.model_init import model_init
from inference.inference_utils import synthesize_utt
from layers.voice_editor import VoiceLatentEditor
from utils import load_audio


class VoiceEditingInterface:
    """Gradioインターフェースのラッパークラス"""

    def __init__(self, model, config, editor, device):
        self.model = model
        self.config = config
        self.editor = editor
        self.device = device
        self.cached_cond_latent = None
        self.cached_ref_path = None

    def process_audio(self, src_audio, ref_audio, *component_values, progress=gr.Progress()):
        """
        音声処理のメイン関数

        Args:
            src_audio: (sample_rate, audio_data) - Gradio audio format
            ref_audio: (sample_rate, audio_data) - Gradio audio format
            *component_values: 可変長の主成分係数
            progress: Gradioプログレスバー

        Returns:
            output_audio: 編集後の音声
            info_text: 編集情報のテキスト
        """
        try:
            if src_audio is None or ref_audio is None:
                return None, "Error: Please upload both source and reference audio"

            progress(0.1, desc="Loading audio...")

            # Gradio形式から音声データを抽出
            src_sr, src_data = src_audio
            ref_sr, ref_data = ref_audio

            # NumPy配列からTensorに変換
            if isinstance(src_data, np.ndarray):
                src_tensor = torch.from_numpy(src_data).float()
                if src_tensor.dim() == 1:
                    src_tensor = src_tensor.unsqueeze(0)
                elif src_tensor.dim() == 2 and src_tensor.shape[0] == 2:
                    # ステレオの場合はモノラルに変換
                    src_tensor = src_tensor.mean(dim=0, keepdim=True)
            else:
                src_tensor = src_data

            if isinstance(ref_data, np.ndarray):
                ref_tensor = torch.from_numpy(ref_data).float()
                if ref_tensor.dim() == 1:
                    ref_tensor = ref_tensor.unsqueeze(0)
                elif ref_tensor.dim() == 2 and ref_tensor.shape[0] == 2:
                    ref_tensor = ref_tensor.mean(dim=0, keepdim=True)
            else:
                ref_tensor = ref_data

            # リサンプリング
            if src_sr != self.model.content_sample_rate:
                src_tensor = torchaudio.functional.resample(
                    src_tensor, src_sr, self.model.content_sample_rate
                )

            if ref_sr != self.config.audio.sample_rate:
                ref_tensor = torchaudio.functional.resample(
                    ref_tensor, ref_sr, self.config.audio.sample_rate
                )

            progress(0.2, desc="Extracting conditioning latent...")

            # 条件付き潜在表現を抽出
            ref_tensor = ref_tensor.to(self.device)
            cond_latent = self.model.get_gpt_cond_latents(
                ref_tensor, self.config.audio.sample_rate
            )

            progress(0.3, desc="Applying voice editing...")

            # 係数を辞書に変換
            coefficients = {}
            prefix = "PC" if self.editor.method == "pca" else "IC"
            for i, value in enumerate(component_values):
                if value != 0.0:
                    coefficients[f"{prefix}{i+1}"] = float(value)

            # 声質編集
            edited_latent = self.editor.edit(cond_latent, coefficients)

            progress(0.5, desc="Synthesizing audio...")

            # 音声合成
            src_tensor = src_tensor.to(self.device)
            synthesized = synthesize_utt(
                self.model,
                src_tensor,
                cond_latent=edited_latent
            )

            progress(0.9, desc="Preparing output...")

            # 出力形式に変換
            output_audio = synthesized.detach().cpu().numpy()
            output_sr = self.config.audio.sample_rate

            # 情報テキスト作成
            info_lines = [
                "=== Voice Editing Info ===",
                f"Method: {self.editor.method.upper()}",
                f"Active components: {len(coefficients)}",
                "",
                "Coefficients:"
            ]
            if coefficients:
                for comp, val in sorted(coefficients.items()):
                    info_lines.append(f"  {comp}: {val:+.3f}")
            else:
                info_lines.append("  (No editing applied)")

            info_lines.extend([
                "",
                f"Source duration: {len(src_data) / src_sr:.2f}s",
                f"Output duration: {len(output_audio) / output_sr:.2f}s"
            ])

            info_text = "\n".join(info_lines)

            progress(1.0, desc="Done!")

            return (output_sr, output_audio), info_text

        except Exception as e:
            import traceback
            error_msg = f"Error: {str(e)}\n\n{traceback.format_exc()}"
            return None, error_msg

    def create_interface(self, n_components=10):
        """Gradioインターフェースを作成"""

        # スライダーの範囲を取得
        sliders = []
        for i in range(n_components):
            comp_info = self.editor.get_component_info(i)
            min_val, max_val = self.editor.get_editing_range(i, n_std=3.0)

            label = comp_info['name']
            if self.editor.method == "pca" and 'explained_variance_ratio' in comp_info:
                label += f" ({comp_info['explained_variance_ratio']*100:.1f}%)"

            slider = gr.Slider(
                minimum=min_val,
                maximum=max_val,
                value=0.0,
                step=0.05,
                label=label,
                interactive=True
            )
            sliders.append(slider)

        # インターフェース構築
        with gr.Blocks(title="Voice Latent Space Explorer") as interface:
            gr.Markdown(f"""
            # 🎙️ Voice Latent Space Explorer

            Explore and edit voice characteristics using {self.editor.method.upper()} components.

            ## How to use:
            1. Upload **source audio** (the content you want to convert)
            2. Upload **reference audio** (your voice that will be edited)
            3. Adjust the sliders to explore different voice characteristics
            4. Click **Generate** to synthesize
            """)

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Input Audio")
                    src_audio_input = gr.Audio(
                        label="Source Audio (content)",
                        type="numpy",
                        sources=["upload", "microphone"]
                    )
                    ref_audio_input = gr.Audio(
                        label="Reference Audio (your voice)",
                        type="numpy",
                        sources=["upload", "microphone"]
                    )

                with gr.Column():
                    gr.Markdown("### Output")
                    output_audio = gr.Audio(
                        label="Generated Audio",
                        type="numpy"
                    )
                    info_output = gr.Textbox(
                        label="Information",
                        lines=10,
                        max_lines=20
                    )

            gr.Markdown(f"### {self.editor.method.upper()} Component Controls")

            with gr.Accordion("Component Sliders", open=True):
                # 2列でスライダーを配置
                with gr.Row():
                    with gr.Column():
                        sliders_col1 = sliders[:n_components//2]
                        for slider in sliders_col1:
                            slider.render()

                    with gr.Column():
                        sliders_col2 = sliders[n_components//2:]
                        for slider in sliders_col2:
                            slider.render()

            # ボタン
            with gr.Row():
                generate_btn = gr.Button("🎵 Generate", variant="primary", size="lg")
                reset_btn = gr.Button("🔄 Reset Sliders", variant="secondary")

            # プリセット（将来の拡張用）
            with gr.Accordion("Presets (Coming Soon)", open=False):
                gr.Markdown("Preset functionality will be added in future updates.")

            # イベントハンドラ
            generate_btn.click(
                fn=self.process_audio,
                inputs=[src_audio_input, ref_audio_input] + sliders,
                outputs=[output_audio, info_output]
            )

            # リセットボタン
            def reset_sliders():
                return [0.0] * n_components

            reset_btn.click(
                fn=reset_sliders,
                outputs=sliders
            )

            # サンプル入力（オプション）
            gr.Markdown("""
            ---
            ### Tips:
            - **PC1** typically controls gender/pitch (in PCA mode)
            - Larger variance ratios indicate more important components
            - Try adjusting one component at a time to understand its effect
            - Use ±3σ range for natural-sounding edits
            """)

        return interface


def main():
    parser = argparse.ArgumentParser(description="Interactive voice editing interface")
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to GenVC model checkpoint')
    parser.add_argument('--method', type=str, default='pca', choices=['pca', 'ica'],
                        help='Analysis method to use')
    parser.add_argument('--analysis_dir', type=str, default='analysis_results',
                        help='Directory containing analysis results')
    parser.add_argument('--n_components', type=int, default=10,
                        help='Number of components to show (max 20)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--share', action='store_true',
                        help='Create public share link')
    parser.add_argument('--port', type=int, default=7860,
                        help='Port number for Gradio interface')

    args = parser.parse_args()

    print("="*70)
    print("Voice Latent Space Explorer")
    print("="*70)

    # モデル初期化
    print(f"\nLoading model from {args.model_path}...")
    model, config = model_init(args.model_path, args.device)

    # 編集モジュール初期化
    print(f"Loading {args.method.upper()} editor from {args.analysis_dir}...")
    try:
        editor = VoiceLatentEditor(method=args.method, analysis_dir=args.analysis_dir)
    except Exception as e:
        print(f"Error loading editor: {e}")
        print("\nPlease run the following steps first:")
        print("  1. python extract_latents.py --model_path <model> --libritts_path <path>")
        print("  2. python analyze_latent_space.py --data_dir latent_data")
        return

    # 成分情報表示
    editor.get_top_components_summary(n_top=min(args.n_components, 10))

    # インターフェース作成
    print(f"\nCreating Gradio interface with {args.n_components} components...")
    interface_wrapper = VoiceEditingInterface(model, config, editor, args.device)
    interface = interface_wrapper.create_interface(n_components=args.n_components)

    # 起動
    print(f"\nLaunching interface on port {args.port}...")
    print(f"Open your browser and go to: http://127.0.0.1:{args.port}")
    if args.share:
        print("Creating public share link...")

    interface.launch(
        server_port=args.port,
        share=args.share,
        server_name="0.0.0.0"
    )


if __name__ == '__main__':
    main()
