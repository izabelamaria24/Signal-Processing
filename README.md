## Signal Processing Labs

This repository contains 5 laboratory exercises covering basic signal processing concepts: signal generation, sampling and noise, Fourier analysis, DFT/FFT, and simple time-series exploration. All labs save figures in both PNG and PDF formats under `charts/LabX`.

### Quick start

1. Install Python 3.10+.
2. Create and activate a virtual environment (recommended).
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run a single lab or all labs:

```bash
# Run Lab 1
python run.py --lab 1

# Run Lab 2 (audio playback is disabled by default)
python run.py --lab 2

# Run all labs
python run.py --lab all
```

Notes:

- Figures are saved to `charts/LabX` as both `.png` and `.pdf`.
- Lab 2 contains optional audio playback. To enable it, set the environment variable `SKIP_AUDIO=0` before running. Playback is skipped by default so the project runs on machines without audio devices.

### Step-by-step setup after cloning

Follow these steps exactly once after cloning the repository.

1. Ensure you have Python 3.10+:

```bash
python --version
```

2. Create a virtual environment and activate it.

- Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

- macOS/Linux (bash/zsh):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run labs:

```bash
# Run all labs (plots will display and files will be saved)
python run.py --lab all

# Or run a specific lab
python run.py --lab 3
```

Optional:

- Audio playback in Lab 2 is OFF by default. To enable it:
  - Windows PowerShell: `setx SKIP_AUDIO 0` (then restart your shell)
  - macOS/Linux: `export SKIP_AUDIO=0`

Where to find outputs:

- All charts are saved under `charts/Lab1`…`charts/Lab5` as `.png` and `.pdf`. The console also prints the saved paths.

### Labs overview

- Lab 1:
  - Cosine signals with different frequencies and phases (continuous vs sampled).
  - Basic 1D signals (sine, sawtooth, square) and 2D images (random and gradient).
- Lab 2:
  - Sine vs cosine phase relation, phase sweeps, SNR impact.
  - Audio generation/playback and saving to WAV (optional playback).
  - Special sampling frequencies, decimation effects, simple function approximations.
- Lab 3:
  - Discrete Fourier matrix, unitarity check.
  - Phasor/wrapping visualization in the complex plane.
  - Manual DFT magnitude of a multi-tone signal.
- Lab 4:
  - Timing comparison: DFT matrix (O(N^2)) vs FFT (O(N log N)).
  - Aliasing and correct sampling visualizations.
  - Spectrograms for provided vowel recordings.
- Lab 5:
  - Traffic time series: sampling, duration, Nyquist.
  - FFT with and without DC component.
  - Dominant periodicities (daily, weekly).
  - One-month slice visualization and simple low-pass filtering.

### Project structure

- `signal_processing_core.py`: Shared utilities for all labs (signal generation, plotting, Fourier helpers, and figure saving).
- `LabX/`: Individual lab packages with `run()` entry points.
- `run.py`: Top-level runner with a simple CLI (`--lab 1..5 | all`).
- `charts/`: All saved figures, organized by lab.

### Reproducibility and notes

- We keep code simple and self-contained; where exercises ask for written answers, we provide concise explanations as comments next to the relevant code.
- Some labs rely on provided assets (e.g., `Lab4/*.wav`, `Lab5/Train.csv`). These are included in the repository so any user can run the project.
