# Subtitulo (Español rioplatense en tiempo real)

Subtitulo es una app de consola que escucha tu micrófono y muestra subtítulos en **español en tiempo real**, optimizada para registro **rioplatense** (vos, che, etc.) usando un prompt lingüístico para mejorar la transcripción.

## Qué hace

- Captura audio del micrófono en vivo.
- Transcribe continuamente con Whisper (vía `faster-whisper`).
- Muestra subtítulos parciales en terminal sin bloquear.
- Aplica un `initial_prompt` orientado a español rioplatense para mejorar resultados.

## Requisitos

- Python 3.10+
- Micrófono funcional
- FFmpeg instalado (requerido por `faster-whisper`/`ctranslate2` en muchos entornos)

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso rápido

```bash
python -m subtitulo \
  --model small \
  --device cpu \
  --compute-type int8 \
  --language es
```

Opciones útiles:

- `--model`: `tiny`, `base`, `small`, `medium`, `large-v3` (más grande = más precisión, más latencia).
- `--device`: `cpu` o `cuda`.
- `--chunk-seconds`: segundos por ventana de transcripción (default 2.0).
- `--overlap-seconds`: solapamiento entre ventanas para no “cortar” palabras (default 0.5).
- `--energy-threshold`: filtro simple para descartar silencio.

## Ajuste para rioplatense

La app fuerza:

- `language="es"`
- `task="transcribe"`
- `initial_prompt` con formas comunes de Río de la Plata (ej. *vos*, *che*, *laburo*, *boludo*, *dale*).

Esto no "convierte" tu habla: mejora cómo el modelo **elige grafías y vocabulario** cuando hay ambigüedad.

## Solución de problemas

- Si no detecta audio, probá `--list-devices` para ver índices de entrada.
- Si va lento en CPU, bajá modelo (`base` o `small`) y/o subí `--chunk-seconds`.
- Si tenés GPU NVIDIA, usá `--device cuda`.

## Aviso

No almacena audio en disco por defecto; procesa en memoria para subtitulado en vivo.
