"""Tests del registry de motores y del catálogo de modelos."""

import sys
import types

import numpy as np
import pytest
import torch

from engines import diffusion_factory, registry
from engines.base import MODEL_CATALOG, ModelSpec, resolve


class TestCatalog:

    def test_catalog_has_real_image_models(self):
        real = [
            s
            for s in MODEL_CATALOG.values()
            if s.engine == "diffusers" and s.mode == "image"
        ]
        assert len(real) >= 5
        assert all(s.repo for s in real)

    def test_catalog_has_real_video_models(self):
        video = [
            s for s in MODEL_CATALOG.values()
            if s.mode == "video" and s.engine == "diffusers"
        ]
        assert len(video) >= 3
        # Wan-1.3B queda en 6 tras cuantizar su text encoder a 4-bit (era 10);
        # los demás videos reales siguen pesando >= 12.
        assert all(s.vram_gb >= 6 for s in video)

    def test_every_spec_declares_pipeline_and_family(self):
        for key, spec in MODEL_CATALOG.items():
            assert spec.pipeline, f"{key} sin pipeline"
            assert spec.family, f"{key} sin familia"
            if spec.engine != "mock":
                assert spec.repo, f"{key} sin repo"

    def test_v1_bridge_families_declared(self):
        bridge = [s for s in MODEL_CATALOG.values() if s.engine == "v1_bridge"]
        assert len(bridge) >= 6  # ltx25, minimax, kandinsky, krea, longcat, z_image...

    def test_resolve_known_model(self):
        spec = resolve("SD-1.5", "image")
        assert spec.engine == "diffusers"
        assert "stable-diffusion" in spec.repo

    def test_resolve_unknown_falls_back_by_mode(self):
        spec = resolve("modelo-inexistente", "image")
        assert spec.engine == "diffusers"  # fallback a un motor real de imagen

        spec_v = resolve("modelo-inexistente", "video")
        assert spec_v.engine == "diffusers"  # el video ya tiene motores reales

    def test_resolve_empty_name(self):
        spec = resolve("", "video")
        assert spec.mode == "video"


class TestRegistry:

    async def test_mock_engine_produces_completed_event(self, tmp_path):
        import asyncio

        import inference_pb2

        req = inference_pb2.MediaGenerationRequest(
            job_id="reg1", mode="video", model_name="Wan2.1-T2V-14B",
            prompt="x", steps=3,
        )
        from engines.base import EngineContext

        ctx = EngineContext(out_dir=str(tmp_path), cancel=asyncio.Event())
        events = []
        async for evt in registry.run_generation(req, ctx):
            events.append(evt)

        statuses = [e[1] for e in events]
        assert statuses[-1] == "COMPLETED"
        assert events[-1][2] == "reg1.mp4"
        assert (tmp_path / "reg1.mp4").exists()

    async def test_force_mock_env_overrides_diffusers(self, tmp_path, monkeypatch):
        import asyncio

        import inference_pb2
        from engines.base import EngineContext

        monkeypatch.setattr(registry, "FORCE_MOCK", True)
        req = inference_pb2.MediaGenerationRequest(
            job_id="reg2", mode="image", model_name="SD-1.5",
            prompt="x", steps=2,
        )
        ctx = EngineContext(out_dir=str(tmp_path), cancel=asyncio.Event())
        events = []
        async for evt in registry.run_generation(req, ctx):
            events.append(evt)

        # Con mock forzado, el output es un PNG placeholder (no pasa por diffusers).
        assert events[-1][1] == "COMPLETED"
        assert events[-1][2] == "reg2.png"

    async def test_mock_image_writes_valid_png(self, tmp_path):
        import asyncio

        import inference_pb2
        from engines.base import EngineContext

        req = inference_pb2.MediaGenerationRequest(
            job_id="reg3", mode="image", model_name="desconocido",
            prompt="x", steps=2,
        )
        ctx = EngineContext(out_dir=str(tmp_path), cancel=asyncio.Event())
        async for _ in registry.run_generation(req, ctx):
            pass

        png = tmp_path / "reg3.png"
        assert png.exists()
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


class TestEngineDispatch:
    """El dispatch por `spec.engine` tiene que cubrir todo el catálogo."""

    def test_every_catalog_entry_has_a_handler(self):
        for key, spec in MODEL_CATALOG.items():
            handler = registry.engine_for(spec)
            assert callable(getattr(handler, "generate", None)), (
                f"{key} ({spec.engine}) sin generate"
            )

    @pytest.mark.parametrize(
        ("engine", "handler_name"),
        [
            ("diffusers", "diffusion_factory"),
            ("v1_bridge", "v1_bridge"),
            ("mock", "mock_engine"),
        ],
    )
    def test_handler_per_engine(self, engine, handler_name):
        spec = ModelSpec(key=engine, label=engine, engine=engine, repo=None)
        assert registry.engine_for(spec) is getattr(registry, handler_name)

    def test_unknown_engine_fails_loudly(self):
        with pytest.raises(ValueError, match="engine sin handler"):
            registry.engine_for(
                ModelSpec(key="x", label="x", engine="onnx", repo="r")
            )

    @pytest.mark.parametrize(
        "model_name",
        [key for key, spec in MODEL_CATALOG.items() if spec.engine == "mock"],
    )
    async def test_mock_catalog_model_dispatches_to_mock_engine(
        self, model_name, tmp_path, monkeypatch
    ):
        """Regresión: sin rama explícita, `engine: mock` caía en diffusion_factory
        y moría con KeyError: 'tts' sobre su tabla LOADERS."""

        import asyncio

        import inference_pb2
        from engines.base import EngineContext

        monkeypatch.setattr(registry, "FORCE_MOCK", False)
        spec = MODEL_CATALOG[model_name]
        req = inference_pb2.MediaGenerationRequest(
            job_id=f"mock_{spec.mode}",
            mode=spec.mode,
            model_name=model_name,
            prompt="hola",
            steps=2,
        )
        ctx = EngineContext(out_dir=str(tmp_path), cancel=asyncio.Event())

        events = []
        async for evt in registry.run_generation(req, ctx):
            events.append(evt)

        assert events[-1][1] == "COMPLETED", events[-1]
        assert (tmp_path / events[-1][2]).exists()


class TestDiffusionFactoryMode:
    """Un `mode` sin loader diffusers falla nombrando modelo y modo."""

    def test_mode_without_loader_fails_with_context(self):
        spec = ModelSpec(
            key="Fake-TTS", label="Fake", engine="diffusers", repo="r",
            mode="tts", pipeline="tts",
        )
        with pytest.raises(ValueError) as excinfo:
            diffusion_factory._get_pipeline(spec, use_gpu=False)

        # Falla antes de cargar nada y el mensaje identifica la spec culpable.
        assert "modo sin loader diffusers: 'tts' (modelo 'Fake-TTS')" in str(
            excinfo.value
        )


class TestVideoMemoryTweaks:
    """Las optimizaciones de VRAM no pueden romper la carga del pipeline.

    WanPipeline no expone `enable_vae_slicing`: el método vive en el VAE, y
    llamarlo en el pipeline hacía fallar la generación entera.
    """

    class _Vae:
        def __init__(self):
            self.slicing = self.tiling = False

        def enable_slicing(self):
            self.slicing = True

        def enable_tiling(self):
            self.tiling = True

    class _WanLike:
        """Pipeline que sólo tiene las mejoras en su VAE."""

        def __init__(self):
            self.vae = TestVideoMemoryTweaks._Vae()

    class _Legacy:
        """Pipeline con los métodos a nivel de pipeline (sin VAE expuesto)."""

        def __init__(self):
            self.sliced = self.tiled = False

        def enable_vae_slicing(self):
            self.sliced = True

        def enable_vae_tiling(self):
            self.tiled = True

    def test_uses_vae_methods_when_pipeline_has_none(self):
        pipe = self._WanLike()

        diffusion_factory._enable_vae_opt(
            pipe, "enable_slicing", "enable_vae_slicing"
        )
        diffusion_factory._enable_vae_opt(
            pipe, "enable_tiling", "enable_vae_tiling"
        )

        assert pipe.vae.slicing is True
        assert pipe.vae.tiling is True

    def test_falls_back_to_pipeline_methods(self):
        pipe = self._Legacy()

        diffusion_factory._enable_vae_opt(
            pipe, "enable_slicing", "enable_vae_slicing"
        )
        diffusion_factory._enable_vae_opt(
            pipe, "enable_tiling", "enable_vae_tiling"
        )

        assert pipe.sliced is True
        assert pipe.tiled is True

    def test_pipeline_without_vae_does_not_raise(self):
        diffusion_factory._enable_vae_opt(object(), "enable_tiling")


class TestGpuLoadPaths:
    """Cada ruta de carga (`image`/`video` x `use_gpu`) toma su decisión.

    Los loaders reales descargan decenas de GB, así que se reemplaza
    `diffusers` por clases falsas y se mira qué elige la factoría: offload de
    modelos, cuantización 4-bit del text encoder o slicing para CPU.
    """

    @pytest.fixture
    def entorno(self, monkeypatch):
        """Falsos de diffusers/accelerate y el pipeline que devolverá la carga."""
        estado = {"offload": []}

        class _Base:
            @classmethod
            def from_pretrained(cls, repo, **kwargs):
                estado["repo"] = repo
                estado["kwargs"] = kwargs
                return estado["pipe"]

        modulo = types.ModuleType("diffusers")
        for nombre in (
            "CogVideoXPipeline",
            "HunyuanVideoPipeline",
            "LTXPipeline",
            "WanPipeline",
            "FluxPipeline",
            "StableDiffusionPipeline",
            "StableDiffusionXLPipeline",
        ):
            setattr(modulo, nombre, type(nombre, (_Base,), {}))

        quant = types.ModuleType("diffusers.quantizers")

        class PipelineQuantizationConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                estado["quant"] = kwargs

        quant.PipelineQuantizationConfig = PipelineQuantizationConfig

        monkeypatch.setitem(sys.modules, "diffusers", modulo)
        monkeypatch.setitem(sys.modules, "diffusers.quantizers", quant)
        monkeypatch.setattr("model_ops.require_weights", lambda repo: None)
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
        monkeypatch.setattr(
            "accelerate.cpu_offload_with_hook",
            lambda module, device=None: estado["offload"].append(module),
        )
        return estado

    @staticmethod
    def _pipe_imagen(entorno):
        class _Pipe:
            def __init__(self):
                self.offloaded = self.sliced = False

            def enable_model_cpu_offload(self):
                self.offloaded = True

            def enable_attention_slicing(self):
                self.sliced = True

        entorno["pipe"] = _Pipe()
        return entorno["pipe"]

    @staticmethod
    def _pipe_video(entorno):
        class _Te:
            device = "cuda"

            def parameters(self):
                return iter(())

            def to(self, device):
                self.device = device

        class _Pipe:
            def __init__(self):
                # Módulos de verdad: el warmup y los hooks los mueven con `.to`.
                self.transformer = torch.nn.Module()
                self.vae = torch.nn.Module()
                self.text_encoder = _Te()
                self.offloaded = self.sliced = False
                self.encode_prompt = lambda *a, **k: "embeds"

            def enable_model_cpu_offload(self):
                self.offloaded = True

            def enable_attention_slicing(self):
                self.sliced = True

        entorno["pipe"] = _Pipe()
        return entorno["pipe"]

    def test_imagen_gpu_usa_offload_de_modelos(self, entorno):
        pipe = self._pipe_imagen(entorno)

        diffusion_factory._load_image_pipeline(
            "stabilityai/sdxl-turbo", True, "float16"
        )

        assert pipe.offloaded is True
        assert "quantization_config" not in entorno["kwargs"]

    def test_imagen_cpu_usa_attention_slicing(self, entorno):
        pipe = self._pipe_imagen(entorno)

        diffusion_factory._load_image_pipeline(
            "stabilityai/sdxl-turbo", False, "float16"
        )

        assert pipe.sliced is True
        assert pipe.offloaded is False

    def test_video_gpu_sin_cuantizar_usa_offload_de_modelos(self, entorno):
        # LTX no necesita 4-bit: entra con el offload genérico de diffusers.
        pipe = self._pipe_video(entorno)

        diffusion_factory._load_video_pipeline(
            "Lightricks/LTX-Video", True, "float16"
        )

        assert pipe.offloaded is True
        assert entorno["offload"] == []

    def test_video_cpu_activa_slicing_y_tiling_del_vae(self, entorno):
        pipe = self._pipe_video(entorno)
        pipe.vae = TestVideoMemoryTweaks._Vae()

        diffusion_factory._load_video_pipeline(
            "Lightricks/LTX-Video", False, "float16"
        )

        assert pipe.sliced is True
        assert pipe.vae.slicing is True
        assert pipe.vae.tiling is True

    def test_wan_gpu_sin_bitsandbytes_falla_antes_de_cargar(
        self, entorno, monkeypatch
    ):
        monkeypatch.setitem(sys.modules, "bitsandbytes", None)
        self._pipe_video(entorno)

        with pytest.raises(RuntimeError, match="requiere 'bitsandbytes'"):
            diffusion_factory._load_video_pipeline(
                "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", True, "float16"
            )

        # Falla antes de descargar/cargar nada.
        assert "kwargs" not in entorno

    def test_wan_gpu_cuantiza_el_text_encoder_y_offloadea_por_etapas(
        self, entorno, monkeypatch
    ):
        monkeypatch.setitem(
            sys.modules, "bitsandbytes", types.ModuleType("bitsandbytes")
        )
        pipe = self._pipe_video(entorno)
        original = pipe.encode_prompt

        diffusion_factory._load_video_pipeline(
            "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", True, "float16"
        )

        assert entorno["quant"]["quant_backend"] == "bitsandbytes_4bit"
        assert entorno["quant"]["components_to_quantize"] == ["text_encoder"]
        # Transformer y VAE con hooks propios, no el offload genérico.
        assert entorno["offload"] == [pipe.transformer, pipe.vae]
        assert pipe.offloaded is False
        # El encode queda envuelto: sube el TE, codifica y lo devuelve a CPU.
        assert pipe.encode_prompt is not original
        # Y tras cargar el TE queda en RAM, no ocupando 5,5 GB ociosos.
        assert pipe.text_encoder.device == "cpu"

    def test_offload_sequentially_castea_los_fp32_del_text_encoder(self):
        """bnb deja las FFN `wo` en fp32 y `te.to(dtype)` está bloqueado."""

        class _Te:
            def __init__(self):
                self.plano = torch.nn.Parameter(
                    torch.zeros(2, 2, dtype=torch.float32)
                )
                self.cuantizado = torch.nn.Parameter(
                    torch.zeros(2, 2, dtype=torch.uint8), requires_grad=False
                )

            def parameters(self):
                return iter((self.plano, self.cuantizado))

            def to(self, device):
                self.device = device

        class _Pipe:
            encode_prompt = staticmethod(lambda *a, **k: "embeds")

            def __init__(self):
                # Módulos reales: acá se ejercita el hook de accelerate de verdad.
                self.transformer = torch.nn.Module()
                self.vae = torch.nn.Module()
                self.text_encoder = _Te()

        pipe = _Pipe()
        diffusion_factory._offload_sequentially(pipe)

        assert pipe.text_encoder.plano.dtype is torch.float16
        assert pipe.text_encoder.cuantizado.dtype is torch.uint8

    def test_offload_sequentially_sin_text_encoder_no_rompe(self):
        """Algunas familias de video no tienen text encoder (p. ej. I2V puro)."""
        pipe = types.SimpleNamespace(
            transformer=torch.nn.Module(), vae=torch.nn.Module()
        )

        diffusion_factory._offload_sequentially(pipe)

        assert hasattr(pipe.transformer, "_hf_hook")

    def test_wrap_encode_prompt_no_se_envuelve_dos_veces(self):
        """Envolver dos veces subiría y bajaría el TE dos veces por prompt."""

        class _Te:
            def to(self, device):
                self.device = device

        class _Pipe:
            encode_prompt = staticmethod(lambda *a, **k: "embeds")

        pipe, te = _Pipe(), _Te()
        diffusion_factory._wrap_encode_prompt(pipe, te, torch)
        envuelto = pipe.encode_prompt

        diffusion_factory._wrap_encode_prompt(pipe, te, torch)

        assert pipe.encode_prompt is envuelto
        assert pipe.encode_prompt("un prompt") == "embeds"

    def test_encode_prompt_devuelve_el_te_a_cpu_aunque_falle(self):
        """Si el encode explota, el TE no puede quedar ocupando 5,5 GB."""

        class _Te:
            def __init__(self):
                self.device = "cpu"

            def to(self, device):
                self.device = device

        class _Pipe:
            @staticmethod
            def encode_prompt(*args, **kwargs):
                raise RuntimeError("sin memoria")

        pipe, te = _Pipe(), _Te()
        diffusion_factory._wrap_encode_prompt(pipe, te, torch)

        with pytest.raises(RuntimeError):
            pipe.encode_prompt("un prompt")

        assert te.device == "cpu"


class TestReleaseVram:
    """Al terminar un job el transformer y el VAE vuelven a RAM."""

    def test_usa_offload_y_si_no_hay_usa_init_hook(self):
        llamadas = []

        class _HookOffload:
            def offload(self):
                llamadas.append("offload")

            def init_hook(self, module):  # no debe usarse si hay `offload`
                llamadas.append("init_hook")

        class _HookInit:
            def init_hook(self, module):
                llamadas.append("init_hook-cpuoffload")

        con_offload = types.SimpleNamespace(_hf_hook=_HookOffload())
        con_init = types.SimpleNamespace(_hf_hook=_HookInit())
        sin_hook = types.SimpleNamespace()
        pipe = types.SimpleNamespace(
            components={"a": con_offload, "b": con_init, "c": sin_hook}
        )

        diffusion_factory.release_vram(pipe)

        # Un componente sin hook (p. ej. el TE 4-bit) no se toca acá.
        assert llamadas == ["offload", "init_hook-cpuoffload"]


class TestWarmupTextEncoder:
    """El primer forward del TE (40 s de kernels de bnb) se adelanta al cargar."""

    class _Te:
        def __init__(self):
            self.llamadas = 0

        def __call__(self, ids):
            self.llamadas += 1
            return None

    class _Tokenizer:
        model_max_length = 512

        def __init__(self):
            self.largo = None

        def __call__(self, textos, **kwargs):
            self.largo = kwargs["max_length"]
            return types.SimpleNamespace(
                input_ids=torch.zeros(1, 4, dtype=torch.long)
            )

    def test_adelanta_el_primer_forward(self, monkeypatch):
        # Sin GPU el `.to("cuda")` del warmup fallaría: acá sólo importa que el
        # forward se ejecute una vez y con el largo acotado.
        monkeypatch.setattr(torch.Tensor, "to", lambda self, *a, **k: self)
        pipe = types.SimpleNamespace(tokenizer=self._Tokenizer())
        te = self._Te()

        diffusion_factory._warmup_text_encoder(pipe, te, torch)

        assert te.llamadas == 1
        # 64 tokens alcanzan para calentar: no se rellena a los 512 de UMT5.
        assert pipe.tokenizer.largo == 64

    def test_sin_tokenizer_no_hace_nada(self):
        te = self._Te()

        diffusion_factory._warmup_text_encoder(object(), te, torch)

        assert te.llamadas == 0

    def test_si_el_forward_falla_no_rompe_la_carga(self, monkeypatch):
        monkeypatch.setattr(torch.Tensor, "to", lambda self, *a, **k: self)

        class _Roto:
            def __call__(self, ids):
                raise RuntimeError("sin kernels")

        # No debe propagar: el calentamiento es un adelanto, no un requisito.
        diffusion_factory._warmup_text_encoder(
            types.SimpleNamespace(tokenizer=self._Tokenizer()), _Roto(), torch
        )


class TestSeedVideo:
    """Dos jobs de video con la misma seed tienen que dar el mismo clip."""

    class _Pipe:
        def __init__(self):
            self.kwargs = None

        def __call__(self, **kwargs):
            self.kwargs = kwargs
            frame = np.zeros((8, 8, 3), dtype=np.uint8)
            return types.SimpleNamespace(frames=[np.stack([frame] * 8)])

    @staticmethod
    def _correr(pipe, tmp_path, seed: int):
        req = types.SimpleNamespace(
            job_id="seed", prompt="un gato", num_frames=8, fps=8, seed=seed,
            negative_prompt="",
        )
        spec = types.SimpleNamespace(guidance_scale=0.0)
        diffusion_factory._run_video(
            pipe, req, spec, str(tmp_path), lambda *a, **k: None,
            cancel=types.SimpleNamespace(is_set=lambda: False), steps=2,
        )
        return pipe.kwargs

    def test_pasa_el_generator_con_la_seed(self, tmp_path):
        kwargs = self._correr(self._Pipe(), tmp_path, 1234)

        generador = kwargs["generator"]
        assert isinstance(generador, torch.Generator)
        # El estado del generator lo determina la seed: comparación exacta.
        esperado = torch.Generator(device=generador.device)
        esperado.manual_seed(1234)
        assert torch.equal(generador.get_state(), esperado.get_state())

    def test_sin_seed_no_pasa_generator(self, tmp_path):
        kwargs = self._correr(self._Pipe(), tmp_path, 0)

        assert "generator" not in kwargs


class TestVideoDims:
    """El video respeta el ratio pedido.

    Regresión: `_run_video` no pasaba width/height, así que un pedido 9:16
    salía con el default del pipeline (832x480, horizontal) aunque el back
    mandara 480x832.
    """

    @pytest.mark.parametrize(
        ("ratio_w", "ratio_h", "esperado"),
        [
            (480, 832, (480, 832)),   # 9:16
            (832, 480, (832, 480)),   # 16:9
            (512, 512, (480, 480)),   # 1:1
            (640, 480, (640, 480)),   # 4:3
            (480, 640, (480, 640)),   # 3:4
        ],
    )
    def test_lado_corto_en_el_max_side_y_multiplos_de_32(
        self, ratio_w, ratio_h, esperado
    ):
        req = types.SimpleNamespace(width=ratio_w, height=ratio_h)
        spec = types.SimpleNamespace(max_side=480)

        dims = diffusion_factory._video_dims(req, spec)

        assert dims == esperado
        assert all(lado % 32 == 0 for lado in dims)

    def test_sin_ratio_usa_el_cuadrado_del_max_side(self):
        req = types.SimpleNamespace(width=0, height=0)
        spec = types.SimpleNamespace(max_side=704)

        assert diffusion_factory._video_dims(req, spec) == (704, 704)

    def test_ltx_9_16_queda_en_multiplos_de_32(self):
        # 704 es el max_side de LTX: el largo se redondea al múltiplo de 32.
        req = types.SimpleNamespace(width=480, height=832)
        spec = types.SimpleNamespace(max_side=704)

        ancho, alto = diffusion_factory._video_dims(req, spec)

        assert (ancho, alto) == (704, 1216)
        assert ancho % 32 == 0 and alto % 32 == 0

    def test_run_video_le_pasa_las_dims_al_pipeline(self, tmp_path):
        class _Pipe:
            def __init__(self):
                self.kwargs = None

            def __call__(self, **kwargs):
                self.kwargs = kwargs
                frame = np.zeros((8, 8, 3), dtype=np.uint8)
                return types.SimpleNamespace(frames=[np.stack([frame] * 4)])

        pipe = _Pipe()
        req = types.SimpleNamespace(
            job_id="ratio", prompt="un gato astronauta", num_frames=4, fps=8,
            seed=0, negative_prompt="", width=480, height=832,
        )
        spec = types.SimpleNamespace(guidance_scale=0.0, max_side=480)

        diffusion_factory._run_video(
            pipe, req, spec, str(tmp_path), lambda *a, **k: None,
            cancel=types.SimpleNamespace(is_set=lambda: False), steps=2,
        )

        assert (pipe.kwargs["width"], pipe.kwargs["height"]) == (480, 832)


class TestMockArtifacts:
    """El artefacto mock tiene que corresponder al modo pedido."""

    @pytest.mark.parametrize(
        ("mode", "ext", "magic"),
        [
            ("image", ".png", b"\x89PNG"),
            ("video", ".mp4", b"BM_MOCK_VIDEO"),
            ("tts", ".wav", b"RIFF"),
            ("audio", ".wav", b"RIFF"),
        ],
    )
    async def test_artifact_matches_mode(
        self, mode, ext, magic, tmp_path, monkeypatch
    ):
        import asyncio

        import inference_pb2
        from engines.base import EngineContext

        monkeypatch.setattr(registry, "FORCE_MOCK", True)
        req = inference_pb2.MediaGenerationRequest(
            job_id=f"m_{mode}", mode=mode, model_name="cualquiera",
            prompt="hola", steps=1,
        )
        ctx = EngineContext(out_dir=str(tmp_path), cancel=asyncio.Event())
        events = []
        async for evt in registry.run_generation(req, ctx):
            events.append(evt)

        assert events[-1][1] == "COMPLETED", events[-1]
        assert events[-1][2] == f"m_{mode}{ext}"
        data = (tmp_path / events[-1][2]).read_bytes()
        assert data.startswith(magic)
        if ext == ".wav":
            # WAV real: cabecera RIFF/WAVE + bloque fmt PCM.
            assert data[8:12] == b"WAVE"
            assert data[12:16] == b"fmt "
