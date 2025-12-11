from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
import gradio as gr


# -----------------------------------------------------------------------------
# Diffusion schedule and shared constants (mirrors scripts/training/AD3.ipynb)
# -----------------------------------------------------------------------------
IMG_SHAPE: Tuple[int, int, int] = (16, 16, 3)
T: int = 300
BETA_START: float = 1e-4
BETA_END: float = 2e-2
MODEL_PATH = Path("data/models/ddpm_conditional.keras")
UPSCALED_SIZE = 256  # Preserve pixel art by upscaling with nearest-neighbor

# Linear beta schedule exactly as in training
betas = np.linspace(BETA_START, BETA_END, T, dtype=np.float32)
alphas = 1.0 - betas
alphas_cumprod = np.cumprod(alphas, axis=0)

betas_tf = tf.constant(betas, dtype=tf.float32)
alphas_tf = tf.constant(alphas, dtype=tf.float32)
alphas_cumprod_tf = tf.constant(alphas_cumprod, dtype=tf.float32)


# -----------------------------------------------------------------------------
# Custom layers / model definition (from AD3)
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


def build_unet_conditional(num_classes: int, time_dim: int = 64, class_dim: int = 32, base_channels: int = 32) -> tf.keras.Model:
    x_in = layers.Input(shape=IMG_SHAPE, name="x")
    t_in = layers.Input(shape=(), dtype=tf.int32, name="t")
    c_in = layers.Input(shape=(), dtype=tf.int32, name="class_id")

    # Time embedding
    time_embedding_layer = TimestepEmbedding(time_dim)
    temb = time_embedding_layer(t_in)
    temb = layers.Dense(time_dim, activation="relu")(temb)
    temb = layers.Dense(time_dim, activation="relu")(temb)

    # Class embedding
    cemb = layers.Embedding(num_classes, class_dim)(c_in)
    cemb = layers.Dense(class_dim, activation="relu")(cemb)
    cemb = layers.Dense(time_dim, activation="relu")(cemb)

    # Fuse conditioning
    cond = layers.Concatenate()([temb, cemb])
    cond = layers.Dense(time_dim, activation="relu")(cond)

    def add_cond(x: tf.Tensor, channels: int) -> tf.Tensor:
        h = layers.Dense(channels)(cond)
        h = layers.ReLU()(h)
        h = layers.Reshape((1, 1, channels))(h)
        return x + h

    # Encoder
    h1 = layers.Conv2D(base_channels, 3, padding="same", activation="relu")(x_in)
    h1 = layers.Conv2D(base_channels, 3, padding="same", activation="relu")(h1)
    h1 = add_cond(h1, base_channels)
    d1 = layers.MaxPooling2D(2)(h1)

    h2 = layers.Conv2D(base_channels * 2, 3, padding="same", activation="relu")(d1)
    h2 = layers.Conv2D(base_channels * 2, 3, padding="same", activation="relu")(h2)
    h2 = add_cond(h2, base_channels * 2)
    d2 = layers.MaxPooling2D(2)(h2)

    # Bottleneck
    b = layers.Conv2D(base_channels * 4, 3, padding="same", activation="relu")(d2)
    b = layers.Conv2D(base_channels * 4, 3, padding="same", activation="relu")(b)
    b = add_cond(b, base_channels * 4)

    # Decoder
    u2 = layers.UpSampling2D()(b)
    u2 = layers.Concatenate()([u2, h2])
    u2 = layers.Conv2D(base_channels * 2, 3, padding="same", activation="relu")(u2)
    u2 = layers.Conv2D(base_channels * 2, 3, padding="same", activation="relu")(u2)

    u1 = layers.UpSampling2D()(u2)
    u1 = layers.Concatenate()([u1, h1])
    u1 = layers.Conv2D(base_channels, 3, padding="same", activation="relu")(u1)
    u1 = layers.Conv2D(base_channels, 3, padding="same", activation="relu")(u1)

    out = layers.Conv2D(3, 3, padding="same", activation=None)(u1)
    return models.Model([x_in, t_in, c_in], out)


# -----------------------------------------------------------------------------
# Model loading and helpers
# -----------------------------------------------------------------------------
_model: Optional[tf.keras.Model] = None
_num_classes: Optional[int] = None
_class_options: Optional[List[Tuple[str, int]]] = None


def infer_num_classes(model: tf.keras.Model) -> Optional[int]:
    for lyr in model.layers:
        if isinstance(lyr, layers.Embedding):
            return int(lyr.input_dim)
    return None


def load_ddpm_model() -> tf.keras.Model:
    global _model, _num_classes
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}")
        _model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={"TimestepEmbedding": TimestepEmbedding},
        )
        _num_classes = infer_num_classes(_model)
    return _model


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
        model = load_ddpm_model()
        inferred = _num_classes or infer_num_classes(model) or 5
        options = [(f"Class {i}", i) for i in range(int(inferred))]
    _class_options = options
    return _class_options


# -----------------------------------------------------------------------------
# Sampling utilities (matching AD3 sampling loop)
# -----------------------------------------------------------------------------
DEVICE = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"


def tensor_to_pil(x: tf.Tensor, upscale_size: Optional[int] = UPSCALED_SIZE) -> Image.Image:
    """Convert a single image tensor in [0,1] to a PIL image and upscale with nearest-neighbor."""
    x_np = tf.clip_by_value(x, 0.0, 1.0)[0].numpy()
    x_np = (x_np * 255.0).astype(np.uint8)
    img = Image.fromarray(x_np)
    if upscale_size:
        img = img.resize((upscale_size, upscale_size), resample=Image.NEAREST)
    return img


def build_t_schedule(steps: int) -> Sequence[int]:
    """Create a timestep schedule mapped onto the original 0..T-1 grid."""
    steps = int(max(1, min(steps, T)))
    if steps == T:
        return list(range(T))
    # Spread indices across the full diffusion horizon to keep ordering consistent.
    return list(np.linspace(0, T - 1, num=steps, dtype=np.int32))


def sample_ddpm(
    model: tf.keras.Model,
    class_id: int,
    steps: int = T,
    noise_level: float = 1.0,
    seed: Optional[int] = None,
    frame_interval: int = 10,
) -> Tuple[tf.Tensor, List[Image.Image]]:
    if seed is not None:
        tf.random.set_seed(int(seed))
        np.random.seed(int(seed))

    schedule = build_t_schedule(steps)
    num_samples = 1
    frame_interval = max(1, int(frame_interval))

    with tf.device(DEVICE):
        x = tf.random.normal((num_samples,) + IMG_SHAPE, dtype=tf.float32) * float(noise_level)
        c = tf.fill((num_samples,), tf.cast(class_id, tf.int32))
        frames: List[Image.Image] = []

        for idx, t_idx in enumerate(reversed(schedule)):
            t_val = tf.fill((num_samples,), tf.cast(t_idx, tf.int32))
            eps_theta = model([x, t_val, c], training=False)

            beta_t = betas_tf[t_idx]
            alpha_t = alphas_tf[t_idx]
            alpha_bar_t = alphas_cumprod_tf[t_idx]

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


# -----------------------------------------------------------------------------
# Public generation entrypoint
# -----------------------------------------------------------------------------
def generate(class_label: int, steps: int, noise_level: float, seed: Optional[int]) -> Tuple[Image.Image, List[Image.Image]]:
    model = load_ddpm_model()
    _, frames = sample_ddpm(
        model=model,
        class_id=int(class_label),
        steps=int(steps),
        noise_level=float(noise_level),
        seed=int(seed) if seed not in (None, "") else None,
        frame_interval=10,
    )
    final_image = frames[-1] if frames else None
    return final_image, frames


# -----------------------------------------------------------------------------
# Gradio interface
# -----------------------------------------------------------------------------
def build_interface() -> gr.Blocks:
    model = load_ddpm_model()  # Ensure loaded before UI starts to surface early errors.
    class_options = get_class_options()

    with gr.Blocks(title="Conditional DDPM (AD3) Viewer") as demo:
        gr.Markdown("# Conditional DDPM (AD3)\nVisualize denoising steps for the trained pixel-art diffusion model.")

        with gr.Row():
            class_dropdown = gr.Dropdown(
                label="Class label",
                choices=[label for label, _ in class_options],
                value=class_options[0][0] if class_options else None,
            )
            steps_slider = gr.Slider(
                minimum=1,
                maximum=T,
                step=1,
                value=T,
                label="Sampling steps",
            )
            noise_slider = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                step=0.05,
                value=1.0,
                label="Noise level (1.0 = training default)",
            )
            seed_box = gr.Number(
                value=None,
                label="Seed (optional)",
                precision=0,
            )

        output_image = gr.Image(label="Final image", type="pil", height=320)
        gallery = gr.Gallery(label="Denoising frames", preview=True, columns=5, height=320)

        label_to_value = {lbl: val for lbl, val in class_options}

        def _ui_generate(label_name, steps, noise_level, seed):
            class_id = label_to_value.get(label_name, 0)
            return generate(class_id, steps, noise_level, seed)

        run_button = gr.Button("Generate")
        run_button.click(
            fn=_ui_generate,
            inputs=[class_dropdown, steps_slider, noise_slider, seed_box],
            outputs=[output_image, gallery],
        )

    return demo


if __name__ == "__main__":
    demo_app = build_interface()
    demo_app.launch()
