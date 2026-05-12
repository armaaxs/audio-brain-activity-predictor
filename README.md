#  Brain Response Prediction Pipeline using TRIBE v2 Audio →
```markdown
A production-style demo pipeline that takes an **audio file** (music, speech, soundscapes) and predicts an **fMRI-like brain response over time** using Meta’s **TRIBE v2**. It outputs (1) a simple response-strength timeline and (2) cortical surface visualizations, plus **deck-ready artifacts** (plots + timelapse GIF) for investor demos.

## What It Does

**Input**
- Audio file: `.wav` recommended (music, ambience, speech, etc.)

**Output**
- Predicted cortical activity over time (vertexwise, fsaverage5 mesh)
- “Response strength” time-series (single curve per clip)
- QA plots: waveform, spectrogram, loudness (RMS)
- Summary table + overlay comparisons across clips
- Exported images + optional GIF for presentations

## How To Run (Google Colab)

### 1) Install + Load TRIBE v2
```bash
!uv pip install "tribev2[plotting] @ git+https://github.com/facebookresearch/tribev2.git"
```

```python
from tribev2.demo_utils import TribeModel
from tribev2.plotting import PlotBrain
from pathlib import Path

CACHE_FOLDER = Path("./cache")
CACHE_FOLDER.mkdir(exist_ok=True)

model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE_FOLDER)
plotter = PlotBrain(mesh="fsaverage5")
```

### 2) Provide Audio
Option A: upload from your computer
```python
from google.colab import files
uploaded = files.upload()

audio_path = CACHE_FOLDER / next(iter(uploaded.keys()))
with open(audio_path, "wb") as f:
    f.write(next(iter(uploaded.values())))

print("Using:", audio_path)
```

Option B: use a local path already in the notebook VM
```python
from pathlib import Path
audio_path = Path("./cache/your_audio.wav")
```

### 3) Run the Demo Dashboard
Run the **Cell 6** and **Cell 7** blocks from this project:
- Cell 6 generates analytics plots + runs predictions for:
  - `clip_user` (your audio)
  - `clip_silence` (generated silence, same duration)
  - `clip_baseline` (synthetic noise baseline, offline-safe)
- Cell 7 generates the cortical surface panels and (optionally) a timelapse GIF.

## Artifacts (Saved Outputs)

All exported assets go to:
- `./cache/figures/`

Typical files:
- `clip_user_01_waveform.png`
- `clip_user_02_spectrogram.png`
- `clip_user_03_rms.png`
- `clip_user_04_reaction.png`
- `clip_user_05_metrics.png`
- `compare_06_reaction_overlay.png`
- `compare_07_summary_table.png`
- `clip_user_brain_01_timesteps.png`
- `clip_user_brain_03_timesteps.gif`

In Colab, open the left sidebar **Files** → `cache/figures/`.

Download everything as a zip:
```bash
!zip -r figures.zip cache/figures
```
```python
from google.colab import files
files.download("figures.zip")
```

## Interpreting the Results (Important)

- The model predicts **fMRI-like cortical response**, not emotions, not “preference”, and not actual measured brain data.
- The “Response strength” curve is a **derived metric** (RMS of baseline-centered activity). It is best used for:
  - comparing **relative changes over time**
  - comparing **clips side-by-side** (user vs baseline vs silence)
- Silence is not always “zero” in model output; treat it as a **baseline** and interpret user clips relative to it.

## No-Transcription / Audio-Only Mode

This demo is configured to prefer **audio-only event construction** to avoid speech-to-text features. If the audio-only helper import fails in your environment, it falls back to `model.get_events_dataframe(audio_path=...)` and filters to `type == "Audio"` rows.

## Performance + Stability Notes

- Long clips can spike RAM. The dashboard caps processing to a short window (e.g. 20s) to keep runs stable and demo-friendly.
- Brain-mesh GIF rendering is expensive. If you hit OOM:
  - export only `*_brain_01_timesteps.png` (single render), or
  - reduce frames (e.g. 6–10 timesteps).

## Common Issues

### `librosa/numba` crashes with NumPy
TRIBE v2 often requires newer NumPy versions that conflict with `numba` (used by librosa). This repo avoids that by using `soundfile + scipy` for audio analytics.

### Network/DNS download errors
If Colab can’t download sample files, use:
- upload from device, or
- synthetic baseline generation (noise) included in Cell 6.

## Disclaimer
This is a demo/visualization pipeline for predicted responses. Use it for exploration, comparisons, and storytelling—not as medical or diagnostic output.

## Credits
- Meta AI Research: TRIBE v2
- Hugging Face model: `facebook/tribev2`
```

