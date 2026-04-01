from __future__ import annotations

import argparse
import queue
import signal
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

RIOPLATENSE_PROMPT = (
    "Transcripción en español rioplatense. "
    "Usar voseo y vocabulario frecuente de Argentina y Uruguay cuando corresponda: "
    "vos, che, dale, laburo, boludo. "
    "Mantener puntuación clara y natural."
)


@dataclass
class AudioConfig:
    sample_rate: int = 16_000
    channels: int = 1
    dtype: str = "float32"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Subtitulado en vivo en español (optimizado para rioplatense)."
    )
    parser.add_argument("--model", default="small", help="Modelo Whisper (tiny/base/small/medium/large-v3)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--compute-type", default="int8", help="Compute type de faster-whisper")
    parser.add_argument("--language", default="es", help="Idioma forzado (default: es)")
    parser.add_argument("--input-device", type=int, default=None, help="Índice de dispositivo de micrófono")
    parser.add_argument("--list-devices", action="store_true", help="Listar dispositivos de audio y salir")
    parser.add_argument("--chunk-seconds", type=float, default=2.0, help="Ventana de audio por transcripción")
    parser.add_argument("--overlap-seconds", type=float, default=0.5, help="Solapamiento entre ventanas")
    parser.add_argument(
        "--energy-threshold",
        type=float,
        default=0.004,
        help="Umbral RMS mínimo para transcribir (reduce silencio)",
    )
    return parser.parse_args()


def list_devices() -> None:
    print(sd.query_devices())


def main() -> None:
    args = parse_args()

    if args.list_devices:
        list_devices()
        return

    audio_cfg = AudioConfig()
    audio_queue: queue.Queue[np.ndarray] = queue.Queue()
    stop_event = threading.Event()

    print("Cargando modelo Whisper...", file=sys.stderr)
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    print("Modelo listo. Hablá para ver subtítulos (Ctrl+C para salir).", file=sys.stderr)

    def _audio_callback(indata: np.ndarray, frames: int, _time, status) -> None:
        if status:
            print(f"[audio status] {status}", file=sys.stderr)
        if frames <= 0:
            return
        audio_queue.put(indata.copy().reshape(-1))

    def _run_transcriber() -> None:
        chunk_samples = int(args.chunk_seconds * audio_cfg.sample_rate)
        overlap_samples = int(args.overlap_seconds * audio_cfg.sample_rate)
        step_samples = max(1, chunk_samples - overlap_samples)

        rolling = np.zeros((0,), dtype=np.float32)
        last_printed = ""

        while not stop_event.is_set():
            try:
                block = audio_queue.get(timeout=0.2)
                rolling = np.concatenate([rolling, block])
            except queue.Empty:
                continue

            while rolling.size >= chunk_samples:
                window = rolling[:chunk_samples]
                rolling = rolling[step_samples:]

                rms = float(np.sqrt(np.mean(window**2)))
                if rms < args.energy_threshold:
                    continue

                segments, _info = model.transcribe(
                    window,
                    language=args.language,
                    task="transcribe",
                    vad_filter=True,
                    initial_prompt=RIOPLATENSE_PROMPT,
                    condition_on_previous_text=True,
                    temperature=0.0,
                    beam_size=1,
                )
                text = " ".join(seg.text.strip() for seg in segments).strip()
                if not text or text == last_printed:
                    continue

                timestamp = time.strftime("%H:%M:%S")
                print(f"[{timestamp}] {text}")
                last_printed = text

    worker = threading.Thread(target=_run_transcriber, daemon=True)
    worker.start()

    def _signal_handler(_sig, _frame) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, _signal_handler)

    with sd.InputStream(
        samplerate=audio_cfg.sample_rate,
        channels=audio_cfg.channels,
        dtype=audio_cfg.dtype,
        callback=_audio_callback,
        device=args.input_device,
        blocksize=0,
    ):
        while not stop_event.is_set():
            time.sleep(0.1)

    worker.join(timeout=1.0)
    print("Subtitulo finalizado.", file=sys.stderr)


if __name__ == "__main__":
    main()
