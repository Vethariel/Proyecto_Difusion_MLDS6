from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
import gradio as gr


# -----------------------------------------------------------------------------
# Constants and schedules
# -----------------------------------------------------------------------------
IMG_SHAPE: Tuple[int, int, int] = (16, 16, 3)
UPSCALED_SIZE = 256

# AD3 paths and linear schedule
AD3_PATH = Path("data/models/ddpm_conditional.keras")
T_AD3 = 300
BETA_START_AD3 = 1e-4
BETA_END_AD3 = 2e-2
betas_ad3 = np.linspace(BETA_START_AD3, BETA_END_AD3, T_AD3, dtype=np.float32)
alphas_ad3 = 1.0 - betas_ad3
alphas_cumprod_ad3 = np.cumprod(alphas_ad3, axis=0)
betas_ad3_tf = tf.constant(betas_ad3, dtype=tf.float32)
alphas_ad3_tf = tf.constant(alphas_ad3, dtype=tf.float32)
alphas_cumprod_ad3_tf = tf.constant(alphas_cumprod_ad3, dtype=tf.float32)

# AD6 paths and cosine schedule (loaded from npz)
AD6_PATH_CANDIDATES = [
    Path("data/models/ddpm_resunet_ad6_ema.keras"),
    Path("data/models/ddpm_resunetema.keras"),  # alternate naming
    Path("data/models/ddpm_resunet_ad6.keras"),  # non-EMA fallback
]
SCHEDULE_AD6_PATH = Path("data/models/schedule_ad6.npz")
if SCHEDULE_AD6_PATH.exists():
    sched = np.load(SCHEDULE_AD6_PATH)
    betas_ad6 = sched["betas"]
    alphas_ad6 = sched["alphas"]
    alphas_cumprod_ad6 = sched["alphas_cumprod"]
else:
    betas_ad6 = betas_ad3
    alphas_ad6 = alphas_ad3
    alphas_cumprod_ad6 = alphas_cumprod_ad3
T_AD6 = len(betas_ad6)
betas_ad6_tf = tf.constant(betas_ad6, dtype=tf.float32)
alphas_ad6_tf = tf.constant(alphas_ad6, dtype=tf.float32)
alphas_cumprod_ad6_tf = tf.constant(alphas_cumprod_ad6, dtype=tf.float32)

DEVICE = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"


# -----------------------------------------------------------------------------
# Custom layers / helpers
# -----------------------------------------------------------------------------
class TimestepEmbedding(layers.Layer):
    def __init__(self, dim: int, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def call(self, t: tf.Tensor) -> tf.Tensor:
        half = self.dim // 2
        freqs = tf.exp(tf.range(half, dtype=tf.float32) * -np.log(10000.0) / (half - 1))
        args = tf.cast(t, tf.float32)[:, None] * freqs[None, :]
        emb = tf.concat([tf.sin(args), tf.cos(args)], axis=-1)
        if self.dim % 2 == 1:
            emb = tf.pad(emb, [[0, 0], [0, 1]])
        return emb

    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim})
        return config


class SplitScaleShift(layers.Layer):
    """Deterministic split to avoid Lambda(tf.split) deserialization issues."""

    def call(self, x: tf.Tensor):
        scale, shift = tf.split(x, 2, axis=-1)
        return scale, shift

    def get_config(self):
        return super().get_config()


def infer_num_classes(model: tf.keras.Model) -> Optional[int]:
    for lyr in model.layers:
        if isinstance(lyr, layers.Embedding):
            return int(lyr.input_dim)
    return None


def infer_null_index(model: tf.keras.Model) -> int:
    for lyr in model.layers:
        if isinstance(lyr, layers.Embedding):
            return int(lyr.input_dim) - 1
    raise ValueError("Could not infer null token for AD6.")


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------
_model_ad3: Optional[tf.keras.Model] = None
_model_ad6: Optional[tf.keras.Model] = None
_num_classes: Optional[int] = None
_class_options: Optional[List[Tuple[str, int]]] = None


def load_ad3_model() -> tf.keras.Model:
    print("Loading AD3 model...")
    global _model_ad3, _num_classes
    if _model_ad3 is None:
        if not AD3_PATH.exists():
            raise FileNotFoundError(f"Trained AD3 model not found at {AD3_PATH}")
        try:
            _model_ad3 = tf.keras.models.load_model(
                AD3_PATH,
                custom_objects={"TimestepEmbedding": TimestepEmbedding, "SplitScaleShift": SplitScaleShift},
                safe_mode=False,
                compile=False,
            )
        except Exception as e:
            import traceback
            print("Failed to load AD3:", e)
            traceback.print_exc()
            raise
        _num_classes = _num_classes or infer_num_classes(_model_ad3)
    print("AD3 model loaded.")
    return _model_ad3


def load_ad6_model() -> tf.keras.Model:
    print("Loading AD6 model...")
    global _model_ad6, _num_classes
    if _model_ad6 is None:
        path = next((p for p in AD6_PATH_CANDIDATES if p.exists()), None)
        if path is None:
            raise FileNotFoundError(f"Trained AD6 model not found. Checked: {AD6_PATH_CANDIDATES}")
        print(f"AD6 path: {path.resolve()}")
        try:
            _model_ad6 = tf.keras.models.load_model(
                path,
                custom_objects={"TimestepEmbedding": TimestepEmbedding, "SplitScaleShift": SplitScaleShift},
                safe_mode=False,  # AD6 has Lambda layers
                compile=False,
            )
        except Exception as e:
            import traceback
            print("Failed to load AD6:", e)
            traceback.print_exc()
            raise
        _num_classes = _num_classes or infer_num_classes(_model_ad6)
    print("AD6 model loaded.")
    return _model_ad6


def get_class_options() -> List[Tuple[str, int]]:
    global _class_options, _num_classes
    if _class_options is not None:
        return _class_options

    options: Optional[List[Tuple[str, int]]] = None
    npz_path = Path("data/intermediate/pixel_art_data.npz")
    if npz_path.exists():
        try:
            data = np.load(npz_path, allow_pickle=True)
            labels = np.array(data["labels"]).astype(int)
            unique_labels = sorted(set(int(x) for x in np.unique(labels)))
            options = [(f"Class {lbl}", lbl) for lbl in unique_labels]
            _num_classes = len(unique_labels)
        except Exception:
            options = None

    if options is None:
        model = load_ad3_model()
        inferred = _num_classes or infer_num_classes(model) or 5
        options = [(f"Class {i}", i) for i in range(int(inferred))]

    _class_options = options
    return _class_options


# -----------------------------------------------------------------------------
# Sampling helpers
# -----------------------------------------------------------------------------
def tensor_to_pil(x: tf.Tensor, upscale_size: Optional[int] = UPSCALED_SIZE) -> Image.Image:
    x_np = tf.clip_by_value(x, 0.0, 1.0)[0].numpy()
    x_np = (x_np * 255.0).astype(np.uint8)
    img = Image.fromarray(x_np)
    if upscale_size:
        img = img.resize((upscale_size, upscale_size), resample=Image.NEAREST)
    return img


def build_t_schedule(steps: int, total_steps: int) -> Sequence[int]:
    steps = int(max(1, min(steps, total_steps)))
    if steps == total_steps:
        return list(range(total_steps))
    return list(np.linspace(0, total_steps - 1, num=steps, dtype=np.int32))


def sample_ad3(
    model: tf.keras.Model,
    class_id: int,
    steps: int,
    noise_level: float,
    seed: Optional[int],
    frame_interval: int = 10,
) -> Tuple[tf.Tensor, List[Image.Image]]:
    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)

    schedule = build_t_schedule(steps, T_AD3)
    num_samples = 1
    frames: List[Image.Image] = []

    with tf.device(DEVICE):
        x = tf.random.normal((num_samples,) + IMG_SHAPE, dtype=tf.float32) * float(noise_level)
        c = tf.fill((num_samples,), tf.cast(class_id, tf.int32))

        for idx, t_idx in enumerate(reversed(schedule)):
            t_val = tf.fill((num_samples,), tf.cast(t_idx, tf.int32))
            eps_theta = model([x, t_val, c], training=False)

            beta_t = betas_ad3_tf[t_idx]
            alpha_t = alphas_ad3_tf[t_idx]
            alpha_bar_t = alphas_cumprod_ad3_tf[t_idx]

            mean = (1.0 / tf.sqrt(alpha_t)) * (x - beta_t * eps_theta / tf.sqrt(1.0 - alpha_bar_t))

            if idx < len(schedule) - 1:
                noise = tf.random.normal(tf.shape(x), dtype=tf.float32) * float(noise_level)
                x = mean + tf.sqrt(beta_t) * noise
            else:
                x = mean

            if (idx % frame_interval == 0) or (idx == len(schedule) - 1):
                frames.append(tensor_to_pil(x))

        x = tf.clip_by_value(x, 0.0, 1.0)
    return x, frames


def sample_ad6(
    model: tf.keras.Model,
    class_id: int,
    steps: int,
    noise_level: float,
    seed: Optional[int],
    frame_interval: int = 10,
    guidance_scale: float = 1.5,
) -> Tuple[tf.Tensor, List[Image.Image]]:
    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)

    steps = int(max(1, min(steps, T_AD6)))
    schedule = list(reversed(range(steps)))
    num_samples = 1
    frames: List[Image.Image] = []
    null_idx = infer_null_index(model)

    with tf.device(DEVICE):
        x = tf.random.normal((num_samples,) + IMG_SHAPE, dtype=tf.float32) * float(noise_level)
        c = tf.fill((num_samples,), tf.cast(class_id, tf.int32))
        c_null = tf.fill((num_samples,), tf.cast(null_idx, tf.int32))

        for idx, t_idx in enumerate(schedule):
            t_val = tf.fill((num_samples,), tf.cast(t_idx, tf.int32))
            eps_cond = model([x, t_val, c], training=False)
            eps_uncond = model([x, t_val, c_null], training=False)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            beta_t = betas_ad6_tf[t_idx]
            alpha_t = alphas_ad6_tf[t_idx]
            alpha_bar_t = alphas_cumprod_ad6_tf[t_idx]

            mean = (1.0 / tf.sqrt(alpha_t)) * (x - beta_t * eps / tf.sqrt(1.0 - alpha_bar_t))
            if idx < len(schedule) - 1:
                noise = tf.random.normal(tf.shape(x), dtype=tf.float32) * float(noise_level)
                x = mean + tf.sqrt(beta_t) * noise
            else:
                x = mean

            if (idx % frame_interval == 0) or (idx == len(schedule) - 1):
                frames.append(tensor_to_pil(x))

        x = tf.clip_by_value(x, 0.0, 1.0)
    return x, frames


# -----------------------------------------------------------------------------
# Public generation entrypoint
# -----------------------------------------------------------------------------
def generate(
    class_label: int, steps: int, noise_level: float, seed: Optional[int], guidance_scale: float
) -> Tuple[Image.Image, List[Image.Image], Image.Image, List[Image.Image]]:
    ad3 = load_ad3_model()
    ad6 = load_ad6_model()
    seed_val = int(seed) if seed not in (None, "") else None

    _, frames_ad3 = sample_ad3(
        model=ad3,
        class_id=int(class_label),
        steps=int(steps),
        noise_level=float(noise_level),
        seed=seed_val,
        frame_interval=10,
    )
    _, frames_ad6 = sample_ad6(
        model=ad6,
        class_id=int(class_label),
        steps=int(steps),
        noise_level=float(noise_level),
        seed=seed_val,
        frame_interval=10,
        guidance_scale=float(guidance_scale),
    )

    final_ad3 = frames_ad3[-1] if frames_ad3 else None
    final_ad6 = frames_ad6[-1] if frames_ad6 else None
    return final_ad3, frames_ad3, final_ad6, frames_ad6


# -----------------------------------------------------------------------------
# Gradio interface
# -----------------------------------------------------------------------------
def build_interface() -> gr.Blocks:
    # Load AD3 early; defer AD6 until first use to avoid slow startup
    print("Loading AD3 (fast load)...")
    load_ad3_model()
    load_ad6_model()
    class_options = get_class_options()
    max_steps = max(T_AD3, T_AD6)

    with gr.Blocks(title="Conditional DDPM Comparison (AD3 vs AD6)") as demo:
        gr.Markdown("# Conditional DDPM Comparison\nAD3 (left) vs AD6 (right)")

        with gr.Row():
            class_dropdown = gr.Dropdown(
                label="Class label",
                choices=[label for label, _ in class_options],
                value=class_options[0][0] if class_options else None,
            )
            steps_slider = gr.Slider(
                minimum=1,
                maximum=max_steps,
                step=1,
                value=max_steps,
                label=f"Sampling steps (AD3≤{T_AD3}, AD6≤{T_AD6})",
            )
            noise_slider = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                step=0.05,
                value=1.0,
                label="Noise level (1.0 = training default)",
            )
            guidance_slider = gr.Slider(
                minimum=0.5,
                maximum=3.0,
                step=0.1,
                value=1.5,
                label="AD6 guidance scale",
            )
            seed_box = gr.Number(
                value=None,
                label="Seed (optional)",
                precision=0,
            )

        with gr.Row():
            output_image_ad3 = gr.Image(label="AD3 final", type="pil", height=320)
            output_image_ad6 = gr.Image(label="AD6 final", type="pil", height=320)
        with gr.Row():
            gallery_ad3 = gr.Gallery(label="AD3 denoising frames", preview=True, columns=5, height=320)
            gallery_ad6 = gr.Gallery(label="AD6 denoising frames", preview=True, columns=5, height=320)

        label_to_value = {lbl: val for lbl, val in class_options}

        def _ui_generate(label_name, steps, noise_level, guidance_scale, seed):
            class_id = label_to_value.get(label_name, 0)
            return generate(class_id, steps, noise_level, seed, guidance_scale)

        run_button = gr.Button("Generate")
        run_button.click(
            fn=_ui_generate,
            inputs=[class_dropdown, steps_slider, noise_slider, guidance_slider, seed_box],
            outputs=[output_image_ad3, gallery_ad3, output_image_ad6, gallery_ad6],
        )

    return demo


if __name__ == "__main__":
    print("Starting app.py with Gradio UI")
    demo_app = build_interface()
    print("Starting Gradio UI on http://127.0.0.1:7860 (or http://localhost:7860)")
    try:
        demo_app.launch(
            server_name="127.0.0.1",
            server_port=7860,
            inbrowser=False,
            prevent_thread_lock=False,
        )
    except Exception as e:
        import traceback
        print("Gradio failed to launch:", e)
        traceback.print_exc()
