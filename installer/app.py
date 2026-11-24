"""GUI del instalador del Synapta Worker (una ventana, tres pantallas).

Pensada para clientes sin conocimiento técnico — todo click o copiar-pegar:

  Pantalla 1 «Instalar»  → botón grande + barra de progreso + log en vivo.
  Pantalla 2 «Emparejar» → pega el código de 6 dígitos que muestra la app
                           Synapta (campo que acepta pegar con Ctrl+V) y
                           clic en Conectar. La URL del back viene precargada.
  Pantalla 3 «Listo»     → estado (GPU conectada / worker corriendo),
                           checkbox «Iniciar con Windows», botón Finalizar.

La lógica vive en installer.core (la misma que usa el CLI y los tests);
acá solo se encolan logs y se llaman esas funciones desde threads para no
bloquear el evento de Tk. Sin dependencias fuera de la stdlib de Python:
PyInstaller la embebe todas.
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from typing import Optional

try:
    from installer import core
except ImportError:  # PyInstaller empaqueta app.py como script raíz
    import core  # type: ignore[no-redef]

try:
    from bridge import BRIDGE_VERSION as WORKER_VERSION
except ImportError:  # ejecutado fuera del paquete (tests sueltos)
    try:
        from ..bridge import BRIDGE_VERSION as WORKER_VERSION  # type: ignore[no-redef]
    except ImportError:
        WORKER_VERSION = "dev"

APP_TITLE = "Synapta Worker"
APP_VERSION = WORKER_VERSION
FONT_FAMILY = "Segoe UI"

# Colores del panel (dark-friendly, coherente con el front Synapta).
BG = "#111827"
CARD = "#1f2937"
FG = "#f3f4f6"
MUTED = "#9ca3af"
ACCENT = "#d7ff3e"   # verde lima Synapta
ACCENT_FG = "#111827"


class QueueLogHandler(logging.Handler):
    """Handler que volca los logs de core a la queue de la GUI."""

    def __init__(self, sink: "queue.Queue[tuple[str, str]]") -> None:
        super().__init__()
        self.sink = sink
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.sink.put(("log", self.format(record)))
        except Exception:  # noqa: BLE001 — nunca romper la GUI por un log
            pass


class InstallerApp(tk.Tk):
    """Ventana única con tres frames conmutados según el paso."""

    def __init__(self) -> None:
        super().__init__()
        self.title(f"{APP_TITLE} v{APP_VERSION}")
        self.geometry("720x560")
        self.minsize(640, 480)
        self.configure(bg=BG)

        self.msgs: "queue.Queue[tuple[str, str]]" = queue.Queue()
        self.busy = False
        self.workdir = core.default_dir()

        # Log en vivo: lo que imprime core (descargas, pip, doctor) llega
        # al cuadro de texto de la pantalla de instalación.
        self._log_handler = QueueLogHandler(self.msgs)
        installer_logger = logging.getLogger("synapta.installer")
        installer_logger.setLevel(logging.INFO)
        installer_logger.addHandler(self._log_handler)

        self._build_styles()
        self._build_header()
        self._build_screens()
        self._poll_queue()

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        # Arranca mostrando install o status según lo que ya exista.
        self.after(150, self._bootstrap)

    # ── construcción ─────────────────────────────────────────────
    def _build_styles(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Bg.TFrame", background=BG)
        style.configure("Card.TFrame", background=CARD)
        style.configure("Bg.TLabel", background=BG, foreground=FG,
                        font=(FONT_FAMILY, 11))
        style.configure("Card.TLabel", background=CARD, foreground=FG,
                        font=(FONT_FAMILY, 11))
        style.configure("Muted.TLabel", background=CARD, foreground=MUTED,
                        font=(FONT_FAMILY, 10))
        style.configure("H1.TLabel", background=BG, foreground=FG,
                        font=(FONT_FAMILY, 16, "bold"))
        style.configure("Accent.TButton", font=(FONT_FAMILY, 12, "bold"))
        style.configure("TProgressbar", troughcolor=CARD,
                        background=ACCENT)

    def _build_header(self) -> None:
        header = ttk.Frame(self, style="Bg.TFrame")
        header.pack(fill="x", padx=24, pady=(18, 6))
        ttk.Label(header, text=f"Synapta Worker v{APP_VERSION}",
                  style="H1.TLabel").pack(side="left")
        ttk.Label(header, text="Instalador",
                  style="Bg.TLabel").pack(side="left", padx=(10, 0))

    def _build_screens(self) -> None:
        self.screens: dict[str, ttk.Frame] = {}
        body = ttk.Frame(self, style="Bg.TFrame")
        body.pack(fill="both", expand=True, padx=24, pady=8)
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=1)

        self.screens["install"] = self._screen_install(body)
        self.screens["pair"] = self._screen_pair(body)
        self.screens["done"] = self._screen_done(body)
        for frame in self.screens.values():
            frame.grid(row=0, column=0, sticky="nsew")
        self.show("install")

    def _card(self, parent: ttk.Frame) -> ttk.Frame:
        card = ttk.Frame(parent, style="Card.TFrame", padding=20)
        card.pack(fill="both", expand=True)
        return card

    def _screen_install(self, parent: ttk.Frame) -> ttk.Frame:
        frame = ttk.Frame(parent, style="Bg.TFrame")
        card = self._card(frame)
        ttk.Label(card, text="Paso 1 — Instalar",
                  style="Card.TLabel",
                  font=(FONT_FAMILY, 13, "bold")).pack(anchor="w")
        ttk.Label(
            card, style="Muted.TLabel", wraplength=620,
            text=("Instala el worker en esta máquina (código + Python + "
                  "motores). Tarda unos minutos la primera vez y solo "
                  "depende de tu conexión."),
        ).pack(anchor="w", pady=(4, 12))

        self.progress = ttk.Progressbar(card, mode="determinate", maximum=100)
        self.progress.pack(fill="x", pady=(0, 8))
        self.install_log = tk.Text(
            card, height=14, wrap="word", state="disabled",
            bg="#0b1120", fg="#d1d5db", insertbackground=FG,
            relief="flat", font=("Consolas", 9), padx=8, pady=8,
        )
        self.install_log.pack(fill="both", expand=True, pady=(0, 12))

        self.install_btn = ttk.Button(
            card, text="Instalar Synapta Worker", style="Accent.TButton",
            command=self.on_install)
        self.install_btn.pack(fill="x", ipady=6)
        return frame

    def _screen_pair(self, parent: ttk.Frame) -> ttk.Frame:
        frame = ttk.Frame(parent, style="Bg.TFrame")
        card = self._card(frame)
        ttk.Label(card, text="Paso 2 — Conectar tu GPU",
                  style="Card.TLabel",
                  font=(FONT_FAMILY, 13, "bold")).pack(anchor="w")
        ttk.Label(
            card, style="Muted.TLabel", wraplength=620,
            text=("En la app Synapta: Tu GPU → Conectar GPU. Copiá el "
                  "código de 6 dígitos y pegalo acá (Ctrl+V)."),
        ).pack(anchor="w", pady=(4, 16))

        self.paired_banner = ttk.Label(card, text="", style="Muted.TLabel",
                                       wraplength=620)
        self.paired_banner.pack(anchor="w", pady=(0, 8))

        ttk.Label(card, text="URL de tu Synapta",
                  style="Muted.TLabel").pack(anchor="w")
        self.back_var = tk.StringVar(value=core.DEFAULT_BACK)
        back_entry = ttk.Entry(card, textvariable=self.back_var,
                               font=(FONT_FAMILY, 11))
        back_entry.pack(fill="x", pady=(2, 12))

        ttk.Label(card, text="Código de 6 dígitos",
                  style="Muted.TLabel").pack(anchor="w")
        self.code_var = tk.StringVar()
        self.code_entry = ttk.Entry(
            card, textvariable=self.code_var, justify="center",
            font=(FONT_FAMILY, 22, "bold"))
        self.code_entry.pack(fill="x", pady=(2, 6), ipady=6)
        # Solo dígitos, máximo 6: pegar el código completo siempre funciona.
        vcmd = (self.register(self._digit_code), "%P")
        self.code_entry.configure(validate="key", validatecommand=vcmd)

        self.pair_msg = ttk.Label(card, text="", style="Muted.TLabel",
                                  wraplength=620)
        self.pair_msg.pack(anchor="w", pady=(0, 12))

        self.pair_btn = ttk.Button(
            card, text="Conectar GPU", style="Accent.TButton",
            command=self.on_pair)
        self.pair_btn.pack(fill="x", ipady=6)
        self.diag_btn = ttk.Button(
            card, text="Diagnosticar conexión", command=self.on_diagnose)
        self.diag_btn.pack(fill="x", ipady=2, pady=(4, 0))
        return frame

    def _screen_done(self, parent: ttk.Frame) -> ttk.Frame:
        frame = ttk.Frame(parent, style="Bg.TFrame")
        card = self._card(frame)
        ttk.Label(card, text="Paso 3 — ¡Listo!",
                  style="Card.TLabel",
                  font=(FONT_FAMILY, 13, "bold")).pack(anchor="w")

        self.done_status = tk.StringVar(value="Verificando estado…")
        ttk.Label(card, textvariable=self.done_status, style="Card.TLabel",
                  wraplength=620).pack(anchor="w", pady=(4, 12))

        # Estado real: un checkbox "marcado" que no corresponde a un atajo
        # existente le mentiría al usuario.
        self.autostart_var = tk.BooleanVar(value=core.autostart_enabled())
        ttk.Checkbutton(
            card, text="Iniciar Synapta Worker cada vez que encienda "
                      "esta computadora",
            variable=self.autostart_var, command=self.on_autostart,
        ).pack(anchor="w", pady=(0, 14))

        ttk.Label(card, text="Últimos eventos (sin terminal):",
                  style="Muted.TLabel").pack(anchor="w", pady=(6, 2))
        self.done_log = tk.Text(
            card, height=8, wrap="word", state="disabled",
            bg="#0b1120", fg="#d1d5db", insertbackground=FG,
            relief="flat", font=("Consolas", 9), padx=8, pady=8,
        )
        self.done_log.pack(fill="both", expand=True, pady=(0, 8))
        logbtns = ttk.Frame(card, style="Card.TFrame")
        logbtns.pack(fill="x", pady=(0, 10))
        ttk.Button(logbtns, text="Actualizar eventos",
                   command=self.refresh_logs).pack(side="left", ipady=2)
        ttk.Button(logbtns, text="Abrir carpeta de logs",
                   command=self.on_open_logs).pack(side="left", padx=(8, 0), ipady=2)
        ttk.Button(logbtns, text="Copiar diagnóstico",
                   command=self.on_copy_diag).pack(side="left", padx=(8, 0), ipady=2)

        # Administración sin consola: modelos (misma API del front) y update.
        wbtns = ttk.Frame(card, style="Card.TFrame")
        wbtns.pack(fill="x", pady=(0, 8))
        ttk.Button(wbtns, text="Modelos…",
                   command=self.on_models).pack(side="left", ipady=2)
        self.update_btn = ttk.Button(wbtns, text="Buscar actualización",
                                     command=self.on_check_update)
        self.update_btn.pack(side="left", padx=(8, 0), ipady=2)
        self.models_summary = tk.StringVar(value="")
        ttk.Label(card, textvariable=self.models_summary,
                  style="Muted.TLabel", wraplength=620).pack(anchor="w",
                                                             pady=(0, 8))

        btns = ttk.Frame(card, style="Card.TFrame")
        btns.pack(fill="x")
        self.start_btn = ttk.Button(btns, text="Detener worker",
                                    command=self.on_toggle_worker)
        self.start_btn.pack(side="left", ipady=4, padx=(0, 8))
        ttk.Button(btns, text="Volver a conectar otra GPU",
                   command=lambda: self.show("pair")).pack(side="left",
                                                           ipady=4)
        ttk.Button(btns, text="Finalizar", style="Accent.TButton",
                   command=self.destroy).pack(side="right", ipady=4)
        return frame

    # ── navegación ───────────────────────────────────────────────
    def show(self, name: str) -> None:
        self.screens[name].tkraise()
        if name == "done":
            self.refresh_done()
        elif name == "pair":
            self.refresh_pair_banner()

    @staticmethod
    def _digit_code(value: str) -> bool:
        # "" habilita borrar todo (si no, Tk rechaza el Ctrl+A+Supr).
        return value == "" or (value.isdigit() and len(value) <= 6)

    def _bootstrap(self) -> None:
        """Si ya está instalado, saltea directo a emparejar/estado."""
        st = core.worker_status(self.workdir)
        if st["installed"]:
            if st["paired"]:
                self.show("done")
            else:
                self.show("pair")
        self._log_line(f"Instalación: {self.workdir}")

    # ── acciones ─────────────────────────────────────────────────
    def _set_busy(self, busy: bool, label: Optional[str] = None) -> None:
        self.busy = busy
        for btn in (self.install_btn, self.pair_btn, self.diag_btn):
            btn.state(["disabled"] if busy else ["!disabled"])
        if label and not busy:
            self.install_btn.configure(text=label)

    def _run_bg(self, fn, on_ok=None) -> None:
        """Ejecuta fn en un thread; on_ok (callable) corre en el hilo de Tk."""

        def worker() -> None:
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 — error user-facing
                self.msgs.put(("err", str(exc)))
                return
            self.msgs.put(("ok", on_ok))

        threading.Thread(target=worker, daemon=True).start()

    def on_install(self) -> None:
        if self.busy:
            return
        self._set_busy(True)
        self.install_btn.configure(text="Instalando… no cierres esta ventana")
        self.progress.configure(mode="indeterminate")
        self.progress.start(12)
        self._log_line("— Instalación iniciada —")

        def job() -> None:
            core.install(self.workdir)

        def ok() -> None:
            self.progress.stop()
            self.progress.configure(mode="determinate", value=100)
            self._set_busy(False)
            self.install_btn.configure(text="Instalar")
            self.show("pair")

        self._run_bg(job, ok)

    def refresh_pair_banner(self) -> None:
        """Avisa a qué back ya está atada la máquina (códigos no son portables)."""
        try:
            st = core.worker_status(self.workdir)
        except Exception:  # noqa: BLE001 — banner informativo, nunca bloquea
            return
        if st.get("paired") and st.get("back_url"):
            self.paired_banner.configure(
                text=f"Esta máquina ya está conectada a {st['back_url']}. "
                     "Si la querés pasar a otra URL, pegá un código NUEVO de esa app.")
        else:
            self.paired_banner.configure(text="")

    def on_pair(self) -> None:
        if self.busy:
            return
        code = self.code_var.get().strip()
        back = core.normalize_back_url(self.back_var.get())
        self.back_var.set(back)
        fixed = core.suggest_back_fix(back)
        if fixed:
            back = fixed
            self.back_var.set(back)
            self.pair_msg.configure(
                text=f"Corregí la URL a {back} (era un typo común). Generá un código nuevo y reintentá.")
            return
        try:
            hint = core.back_mismatch_hint(back)
        except Exception:  # noqa: BLE001 — hint opcional
            hint = ""
        if hint:
            self.paired_banner.configure(text="⚠ " + hint)
        if len(code) != 6 or not code.isdigit():
            self.pair_msg.configure(
                text="Pegá el código de 6 dígitos que muestra la app.")
            self.code_entry.focus_set()
            return
        self._set_busy(True)
        self.pair_msg.configure(text="Conectando…")

        def ok(warn: str = "") -> None:
            self._set_busy(False)
            self.pair_msg.configure(text="")
            self.code_var.set("")
            self.show("done")
            if warn:
                messagebox.showwarning(APP_TITLE, warn)

        def worker() -> None:
            try:
                core.pair(self.workdir, back=back, code=code)
            except RuntimeError as exc:
                self.msgs.put(("pair_err", str(exc)))
                return
            # Conectar ya implica arrancar: el usuario no ve consolas.
            # (También en el thread — start() espera hasta 20 s el puerto
            # y no hay que congelar la ventana.)
            warn = ""
            try:
                core.start(self.workdir)
            except RuntimeError as exc:
                warn = str(exc)
            self.msgs.put(("ok", lambda: ok(warn)))

        threading.Thread(target=worker, daemon=True).start()

    def on_diagnose(self) -> None:
        """Pasos URL → DNS → TCP → HTTP contra el back, en texto claro."""
        if self.busy:
            return
        back = self.back_var.get().strip()
        self._set_busy(True)
        self.pair_msg.configure(text="Diagnosticando…")

        def worker() -> None:
            steps = core.diagnose_back(back)
            self.msgs.put(("pair_diag", steps))

        threading.Thread(target=worker, daemon=True).start()

    def on_autostart(self) -> None:
        enable = bool(self.autostart_var.get())
        try:
            core.set_autostart(self.workdir, enable=enable)
        except RuntimeError as exc:
            messagebox.showwarning(APP_TITLE, str(exc))

    def on_toggle_worker(self) -> None:
        if self.busy:
            return
        running = core.port_open()
        if running:
            core.stop(self.workdir)
            self.refresh_done()
            return

        def job() -> None:
            core.start(self.workdir)

        def ok() -> None:
            self.refresh_done()

        self._run_bg(job, ok)

    def refresh_logs(self) -> None:
        """Volca la actividad reciente en la pantalla Listo (sin PowerShell).

        Vista filtrada sin ruido de sondeo; el diagnóstico crudo completo
        sigue disponible con «Copiar diagnóstico» (worker_log_tail)."""
        try:
            text = core.worker_log_activity(self.workdir)
        except Exception as exc:  # noqa: BLE001 — mostrar, no romper
            text = f"no pude leer los logs: {exc}"
        self.done_log.configure(state="normal")
        self.done_log.delete("1.0", "end")
        self.done_log.insert("end", text)
        self.done_log.see("end")
        self.done_log.configure(state="disabled")

    def on_open_logs(self) -> None:
        import os as _os
        import subprocess as _sp
        folder = str(core.log_dir(self.workdir))
        try:
            if _os.name == "nt":
                _os.startfile(folder)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                _sp.Popen(["open", folder])
            else:
                _sp.Popen(["xdg-open", folder])
        except Exception as exc:  # noqa: BLE001 — user-facing
            messagebox.showwarning(APP_TITLE, f"No pude abrir {folder}: {exc}")

    def on_copy_diag(self) -> None:
        try:
            text = core.worker_log_tail(self.workdir)
        except Exception as exc:  # noqa: BLE001 — user-facing
            messagebox.showwarning(APP_TITLE, f"No pude leer los logs: {exc}")
            return
        self.clipboard_clear()
        self.clipboard_append(text)
        messagebox.showinfo(APP_TITLE, "Diagnóstico copiado: pegalo en el chat de soporte.")

    # ── modelos y actualización ──────────────────────────────────
    def on_models(self) -> None:
        """Ventana con el catálogo del worker y sus pesos."""
        ModelsWindow(self)

    def on_check_update(self) -> None:
        """Compara la versión instalada con la publicada y ofrece actualizar."""
        if self.busy:
            return
        self.update_btn.state(["disabled"])
        self.update_btn.configure(text="Buscando…")

        def job() -> None:
            self.msgs.put(("update_info", core.check_update(self.workdir)))

        threading.Thread(target=job, daemon=True).start()

    def _on_update_info(self, info: dict) -> None:
        self.update_btn.state(["!disabled"])
        self.update_btn.configure(text="Buscar actualización")
        error = info.get("error") or ""
        if error:
            messagebox.showwarning(APP_TITLE, error)
            return
        current = info.get("current") or "desconocida"
        if not info.get("available"):
            messagebox.showinfo(
                APP_TITLE,
                f"Ya tenés la última versión instalada (v{current}).")
            return
        if not messagebox.askyesno(
                APP_TITLE,
                f"Hay una versión nueva: v{info.get('latest')}\n"
                f"Instalada: v{current}\n\n"
                "¿Actualizar ahora? El worker se reinicia solo y vuelve a "
                "quedar en línea en menos de un minuto."):
            return
        self.update_btn.state(["disabled"])
        self.update_btn.configure(text="Actualizando…")
        self.models_summary.set(
            "Actualizando el worker… se reinicia solo, no cierres la ventana.")

        def job() -> None:
            core.update(self.workdir)

        def ok() -> None:
            self.update_btn.configure(text="Buscar actualización")
            self.update_btn.state(["!disabled"])
            installed = core.local_version(self.workdir)
            self.refresh_done()
            messagebox.showinfo(
                APP_TITLE,
                f"Worker actualizado a v{installed or '?'} y en línea de nuevo.")

        self._run_bg(job, ok)

    def _load_models_summary(self) -> None:
        """Resumen de modelos para la pantalla Listo (en un thread, no bloquea)."""
        try:
            models = core.list_models()
        except Exception:  # noqa: BLE001 — worker caído: se informa, no rompe
            self.msgs.put(("models_summary",
                           "Modelos: el worker no está respondiendo."))
            return
        ready = sum(1 for m in models if m.get("available"))
        dropping = sum(1 for m in models if m.get("downloading"))
        on_disk = sum(int(m.get("size_bytes") or 0) for m in models)
        text = (f"Modelos: {ready} listos de {len(models)} · "
                f"{_human_bytes(on_disk)} en disco")
        if dropping:
            text += f" · {dropping} descargando"
        self.msgs.put(("models_summary", text))

    # ── estado final ─────────────────────────────────────────────
    def refresh_done(self) -> None:
        st = core.worker_status(self.workdir)
        parts = [f"v{APP_VERSION}"]
        parts.append("Instalado ✓" if st["installed"] else "No instalado ✗")
        parts.append("GPU conectada ✓" if st["paired"]
                     else "Sin conectar ✗")
        parts.append("Worker corriendo ✓" if st["running"]
                     else "Worker detenido")
        if st["back_url"]:
            parts.append(f"({st['back_url']})")
        self.done_status.set(" · ".join(parts))
        self.start_btn.configure(
            text="Detener worker" if st["running"] else "Iniciar worker")
        self.autostart_var.set(core.autostart_enabled())
        self.refresh_logs()
        if st["running"]:
            threading.Thread(target=self._load_models_summary,
                             daemon=True).start()
        else:
            self.models_summary.set("Modelos: el worker está detenido.")

    # ── cola de logs/errores ─────────────────────────────────────
    def _log_line(self, text: str) -> None:
        self.install_log.configure(state="normal")
        self.install_log.insert("end", text + "\n")
        self.install_log.see("end")
        self.install_log.configure(state="disabled")

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self.msgs.get_nowait()
                if kind == "log":
                    self._log_line(payload)
                elif kind == "ok":
                    if callable(payload):
                        payload()
                elif kind == "err":
                    self.progress.stop()
                    self._set_busy(False)
                    self.update_btn.configure(text="Buscar actualización")
                    self.update_btn.state(["!disabled"])
                    self._log_line(f"✘ {payload}")
                    messagebox.showerror(APP_TITLE, payload)
                elif kind == "pair_err":
                    self._set_busy(False)
                    self.pair_msg.configure(text=payload)
                elif kind == "pair_diag":
                    self._set_busy(False)
                    lines = [
                        f"{'✓' if s['ok'] else '✘'} {s['step']}: {s['detail']}"
                        for s in payload
                    ]
                    self.pair_msg.configure(text="\n".join(lines))
                elif kind == "update_info":
                    self._on_update_info(payload)
                elif kind == "models_summary":
                    self.models_summary.set(payload)
        except queue.Empty:
            pass
        self.after(120, self._poll_queue)

    def _on_close(self) -> None:
        if self.busy:
            if not messagebox.askokcancel(
                    APP_TITLE,
                    "La instalación sigue en curso. ¿Cerrar de todos modos?"):
                return
        self.destroy()

    def destroy(self) -> None:
        # Sin esto, un log posterior volvería a una queue huérfana.
        handler = getattr(self, "_log_handler", None)
        if handler is not None:
            logging.getLogger("synapta.installer").removeHandler(handler)
            self._log_handler = None
        super().destroy()


def _human_bytes(size: float) -> str:
    """Tamaño legible (GB/MB) sin dependencias."""
    value = float(size or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit in ("B", "KB") else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


class ModelsWindow(tk.Toplevel):
    """Catálogo del worker con gestión de pesos, sin salir del instalador.

    Mismas operaciones que el panel del front (listar / descargar / cancelar /
    borrar) pero contra el artifact server local del worker, que es el que
    tiene los pesos de esta máquina.
    """

    COLUMNS = (
        ("model", "Modelo", 250),
        ("mode", "Tipo", 70),
        ("vram", "VRAM", 70),
        ("state", "Estado", 170),
        ("size", "En disco", 100),
    )

    def __init__(self, app: InstallerApp) -> None:
        super().__init__(app)
        self.app = app
        self.title(f"{APP_TITLE} — Modelos")
        self.geometry("820x480")
        self.minsize(680, 400)
        self.configure(bg=BG)
        self.models: list[dict] = []
        self.busy = False
        self.closed = False
        self._build()
        self.refresh()
        self.protocol("WM_DELETE_WINDOW", self.close)

    # ── construcción ─────────────────────────────────────────────
    def _build(self) -> None:
        wrap = ttk.Frame(self, style="Bg.TFrame", padding=14)
        wrap.pack(fill="both", expand=True)

        self.summary = tk.StringVar(value="Cargando modelos…")
        ttk.Label(wrap, textvariable=self.summary, style="Bg.TLabel",
                  wraplength=760).pack(anchor="w", pady=(0, 8))

        columns = [c[0] for c in self.COLUMNS]
        self.tree = ttk.Treeview(wrap, columns=columns, show="headings",
                                 height=14, selectmode="browse")
        for key, title, width in self.COLUMNS:
            self.tree.heading(key, text=title)
            self.tree.column(key, width=width, anchor="w")
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", lambda _e: self._sync_buttons())
        self.tree.bind("<Double-1>", lambda _e: self.on_download())

        buttons = ttk.Frame(wrap, style="Bg.TFrame")
        buttons.pack(fill="x", pady=(10, 0))
        self.download_btn = ttk.Button(buttons, text="Descargar pesos",
                                       command=self.on_download)
        self.download_btn.pack(side="left", ipady=2)
        self.cancel_btn = ttk.Button(buttons, text="Cancelar descarga",
                                     command=self.on_cancel)
        self.cancel_btn.pack(side="left", padx=(8, 0), ipady=2)
        self.delete_btn = ttk.Button(buttons, text="Borrar pesos",
                                     command=self.on_delete)
        self.delete_btn.pack(side="left", padx=(8, 0), ipady=2)
        ttk.Button(buttons, text="Actualizar lista",
                   command=self.refresh).pack(side="left", padx=(8, 0), ipady=2)
        ttk.Button(buttons, text="Cerrar",
                   command=self.close).pack(side="right", ipady=2)
        self._sync_buttons()

    # ── datos ────────────────────────────────────────────────────
    def refresh(self) -> None:
        if self.closed:
            return
        try:
            models = core.list_models()
        except Exception:  # noqa: BLE001 — worker caído: se avisa y se reintenta
            self.summary.set(
                "El worker no responde en el puerto local. Iniciá el worker y "
                "volvé a probar (reintento automático en 3 s).")
            self._schedule(3000)
            return
        self.models = models
        selected = self.tree.selection()
        for item in self.tree.get_children():
            self.tree.delete(item)
        for model in models:
            key = str(model.get("key") or "")
            state, _dl, _cancel, _delete = self._state(model)
            self.tree.insert("", "end", iid=key, values=(
                model.get("label") or key,
                model.get("mode") or "",
                f"{model.get('vram_gb')} GB" if model.get("vram_gb") else "—",
                state,
                _human_bytes(model.get("size_bytes") or 0),
            ))
        for iid in selected:
            if self.tree.exists(iid):
                self.tree.selection_set(iid)
        ready = sum(1 for m in models if m.get("available"))
        on_disk = sum(int(m.get("size_bytes") or 0) for m in models)
        self.summary.set(
            f"{ready} listos de {len(models)} · {_human_bytes(on_disk)} en disco. "
            "Elegí un modelo de la lista para descargar o borrar sus pesos.")
        self._sync_buttons()
        self._schedule(2500)

    def _schedule(self, ms: int) -> None:
        if self.closed:
            return
        try:
            self.after(ms, self.refresh)
        except tk.TclError:
            pass

    @staticmethod
    def _state(model: dict) -> tuple:
        """(texto de estado, puede_descargar, puede_cancelar, puede_borrar)."""
        if model.get("downloading"):
            return (f"Descargando {float(model.get('progress') or 0):.0f}%",
                    False, True, False)
        if model.get("available"):
            return "Listo", False, False, bool(model.get("size_bytes"))
        error = str(model.get("error") or "")
        if error:
            return f"Error: {error[:40]}", True, False, bool(model.get("size_bytes"))
        if model.get("size_bytes"):
            return "Pesos incompletos", True, False, True
        return "No descargado", True, False, False

    def _selected(self) -> Optional[dict]:
        selected = self.tree.selection()
        if not selected:
            return None
        key = selected[0]
        for model in self.models:
            if str(model.get("key")) == key:
                return model
        return None

    def _sync_buttons(self) -> None:
        model = self._selected()
        _state, can_download, can_cancel, can_delete = (
            self._state(model) if model else ("", False, False, False))
        idle = not self.busy and model is not None
        for btn, allowed in ((self.download_btn, can_download),
                             (self.cancel_btn, can_cancel),
                             (self.delete_btn, can_delete)):
            btn.state(["!disabled"] if (allowed and idle) else ["disabled"])

    # ── acciones ─────────────────────────────────────────────────
    def on_download(self) -> None:
        model = self._selected()
        if not model:
            return
        key = str(model.get("key"))
        self._run(lambda: core.download_model(key),
                  f"Descargando {model.get('label') or key}…")

    def on_cancel(self) -> None:
        model = self._selected()
        if not model:
            return
        key = str(model.get("key"))
        self._run(lambda: core.cancel_model_download(key),
                  f"Cancelando la descarga de {model.get('label') or key}…")

    def on_delete(self) -> None:
        model = self._selected()
        if not model:
            return
        key = str(model.get("key"))
        name = model.get("label") or key
        size = _human_bytes(model.get("size_bytes") or 0)
        if not messagebox.askyesno(
                APP_TITLE,
                f"¿Borrar los pesos de {name} ({size})?\n"
                "Se pueden volver a descargar, pero hay que bajarlos de nuevo."):
            return
        self._run(lambda: core.delete_model(key), f"Borrando {name}…")

    def _run(self, action, message: str) -> None:
        if self.busy:
            return
        self.busy = True
        self.summary.set(message)
        self._sync_buttons()

        def job() -> None:
            try:
                action()
            except Exception as exc:  # noqa: BLE001 — user-facing
                self._post(self._failed, str(exc))
                return
            self._post(self._finished)

        threading.Thread(target=job, daemon=True).start()

    def _post(self, fn, *args) -> None:
        if self.closed:
            return
        try:
            self.after(0, lambda: fn(*args))
        except tk.TclError:
            pass

    def _failed(self, error: str) -> None:
        self.busy = False
        self.summary.set(f"No se pudo: {error}")
        self._sync_buttons()

    def _finished(self) -> None:
        self.busy = False
        self.refresh()

    def close(self) -> None:
        self.closed = True
        self.destroy()


def main() -> int:
    # El .exe se empaqueta SIN consola: sys.stderr puede ser None y
    # basicConfig(stream=None) revienta. La GUI ve los logs igual, por
    # QueueLogHandler.
    if sys.stderr is not None:
        logging.basicConfig(level=logging.INFO, format="%(message)s",
                            stream=sys.stderr, force=True)
    else:
        logging.getLogger("synapta.installer").addHandler(
            logging.NullHandler())
    app = InstallerApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
