"""Banco GPU de Wan: kernels crudos y etapas del pipeline real.

Con el worker apagado (VRAM vacía) separa dos preguntas distintas:

  * `kernels`: ¿la GPU es lenta de verdad? matmul, SDPA con las formas reales
    de Wan y un forward del transformer suelto, sin offload ni cuantización.
  * `stages`: ¿dónde se va el tiempo en el camino real? carga, text encoder
    4-bit, forward del transformer y decode del VAE con y sin tiling.

Uso:

    .venv/Scripts/python.exe bench_gpu.py kernels
    .venv/Scripts/python.exe bench_gpu.py stages
    .venv/Scripts/python.exe bench_gpu.py            # las dos cosas
"""

import sys
import time

import torch

REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
PROMPT = "dos gatos naranjas peleando en una azotea al atardecer"
HIDDEN = (1, 16, 3, 60, 104)  # 480p, 9 frames con el VAE de Wan
ENC = (1, 512, 4096)
SEQ, HEADS, DIM = 4680, 12, 128


def gb(n: float) -> float:
    return n / 1024**3


def vram(tag: str) -> None:
    free, total = torch.cuda.mem_get_info()
    print(
        f"[vram] {tag:24} alloc {gb(torch.cuda.memory_allocated()):5.2f} "
        f"reserv {gb(torch.cuda.memory_reserved()):5.2f} "
        f"libre {gb(free):5.2f}/{gb(total):5.2f}",
        flush=True,
    )


def ms(fn, iters: int = 3, warmup: int = 1) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000


def crono(tag: str, fn):
    """Una sola pasada de una etapa, en segundos."""
    torch.cuda.synchronize()
    t0 = time.time()
    out = fn()
    torch.cuda.synchronize()
    print(f"[etapa] {tag:34} {time.time() - t0:7.1f} s", flush=True)
    return out


def kernels() -> None:
    for dt in (torch.float16, torch.bfloat16):
        a = torch.randn(4096, 4096, device="cuda", dtype=dt)
        b = torch.randn(4096, 4096, device="cuda", dtype=dt)
        t = ms(lambda: a @ b)
        print(
            f"[matmul] 4096^3 {str(dt):16} {t:7.2f} ms = "
            f"{2 * 4096**3 / (t / 1000) / 1e12:6.1f} TFLOP/s",
            flush=True,
        )
        del a, b
        torch.cuda.empty_cache()

    from torch.nn.attention import SDPBackend, sdpa_kernel

    for dt in (torch.float16, torch.bfloat16):
        q = torch.randn(1, HEADS, SEQ, DIM, device="cuda", dtype=dt)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        for nombre, backend in (
            ("auto", None),
            ("efficient", SDPBackend.EFFICIENT_ATTENTION),
            ("math", SDPBackend.MATH),
        ):
            try:
                if backend is None:
                    t = ms(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v))
                else:
                    with sdpa_kernel(backend):
                        t = ms(
                            lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v)
                        )
                print(f"[sdpa] {str(dt):16} {nombre:10} {t:8.1f} ms", flush=True)
            except Exception as exc:  # noqa: BLE001 — el punto es saber cuál falla
                print(f"[sdpa] {str(dt):16} {nombre:10} NO DISPONIBLE: {exc}", flush=True)
        del q, k, v
        torch.cuda.empty_cache()
    vram("tras sdpa")

    from diffusers import WanTransformer3DModel

    for dt in (torch.float16, torch.bfloat16):
        tr = WanTransformer3DModel.from_pretrained(
            REPO, subfolder="transformer", torch_dtype=dt
        )
        tr.to("cuda").eval()
        vram(f"transformer {dt}")
        hidden = torch.randn(*HIDDEN, device="cuda", dtype=dt)
        enc = torch.randn(*ENC, device="cuda", dtype=dt)
        ts = torch.tensor([500.0], device="cuda")
        with torch.no_grad():
            t_full = ms(
                lambda: tr(
                    hidden_states=hidden, timestep=ts, encoder_hidden_states=enc,
                    return_dict=False,
                ),
                iters=2,
                warmup=1,
            )
        print(
            f"[forward] {dt} completo {t_full / 1000:8.1f} s | "
            f"{len(tr.blocks)} bloques = {t_full / len(tr.blocks):.0f} ms/bloque",
            flush=True,
        )
        del tr, hidden, enc
        torch.cuda.empty_cache()
        vram(f"tras liberar {dt}")


def stages() -> None:
    import engines.diffusion_factory as factory

    vram("inicio")
    t0 = time.time()
    pipe = factory._load_video_pipeline(REPO, True, "float16")
    print(f"[etapa] {'carga del pipeline':34} {time.time() - t0:7.1f} s", flush=True)
    vram("tras cargar")

    te = pipe.text_encoder
    with torch.no_grad():
        ids = pipe.tokenizer(
            [PROMPT], padding="max_length", max_length=226, truncation=True,
            return_tensors="pt",
        ).input_ids.to("cuda")
        for i in range(2):
            out = crono(f"text encoder forward #{i + 1}", lambda: te(ids))
        print(
            f"[info] salida del TE {tuple(out.last_hidden_state.shape)} "
            f"{out.last_hidden_state.dtype}",
            flush=True,
        )
        del out
        torch.cuda.empty_cache()
        vram("tras el TE")

        pipe.transformer.to("cuda")
        hidden = torch.randn(*HIDDEN, device="cuda", dtype=torch.float16)
        enc = torch.randn(*ENC, device="cuda", dtype=torch.float16)
        ts = torch.tensor([500.0], device="cuda")
        for i in range(2):
            crono(
                f"forward transformer #{i + 1}",
                lambda: pipe.transformer(
                    hidden_states=hidden, timestep=ts, encoder_hidden_states=enc,
                    return_dict=False,
                ),
            )
        vram("tras el transformer")
        del hidden, enc
        torch.cuda.empty_cache()

        vae = pipe.vae
        if hasattr(vae, "_hf_hook"):
            import accelerate

            accelerate.hooks.remove_hook_from_module(vae, recurse=True)
        vae.to("cuda")
        lat = torch.randn(*HIDDEN, device="cuda", dtype=vae.dtype)
        vae.enable_tiling()
        crono("vae decode (tiling)", lambda: vae.decode(lat, return_dict=False))
        vram("tras el VAE")
        vae.disable_tiling()
        crono("vae decode (sin tiling)", lambda: vae.decode(lat, return_dict=False))
        vram("final")


def main(argv: list[str]) -> int:
    print(
        f"torch {torch.__version__} cuda {torch.version.cuda} | "
        f"{torch.cuda.get_device_name(0)} cap {torch.cuda.get_device_capability(0)} | "
        f"tf32 matmul={torch.backends.cuda.matmul.allow_tf32}",
        flush=True,
    )
    pedidos = [a for a in argv if a in ("kernels", "stages")] or ["kernels", "stages"]
    vram("inicio")
    if "kernels" in pedidos:
        kernels()
    if "stages" in pedidos:
        stages()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
