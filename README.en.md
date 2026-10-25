# LumenFigure

[中文](README.md)

LumenFigure turns the motion in a video into a **grayscale depth image** that a later AI video model can follow. The result drops the source color and texture. Background patterns, clothing colors, and fabric prints — details a generator tends to copy along with the motion — are removed. What remains is silhouette and distance: closer to the camera is brighter. An arm reaching toward the lens, for example, brightens from the shoulder out to the hand.

This project reproduces the video person-motion depth extraction in [LibTV (里布 TV)](https://www.liblib.tv/) and [OiiOii](https://www.oiioii.ai/). The screenshots below put the three results next to each other. LumenFigure’s gray levels sit closer to OiiOii’s depth extraction. LibTV’s depth-motion capture fills the person with a flat color, so the inner contours of the head, arms, and torso disappear. A flat silhouette carries less body shape when a generator needs those inner contours.

```text
input video → optional RVM matte → Video Depth Anything streaming depth → near-bright / far-dark → silent H.264
```

The repository includes Robust Video Matting inference code, so the project as a whole is released under **GPL-3.0**. Third-party sources, licenses, and weight URLs are listed in [THIRD_PARTY.md](THIRD_PARTY.md).

## Contents

- [Comparison](#comparison)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Options](#options)
- [Troubleshooting](#troubleshooting)
- [Measured speed](#measured-speed)
- [License](#license)

## Comparison

All three images are the same selfie shot. The first two are product UI screenshots. The third is a crop of a frame exported by this project.

**OiiOii depth extraction.** The source frame is on the left. The grayscale depth on the right keeps shading inside the person, and the background stays as a darker depth layer.

![OiiOii depth extraction: source frame on the left, grayscale depth with inner contours on the right](examples/oiioii.png)

**LibTV depth-motion capture.** The person becomes a solid white silhouette. The arm, neck, and torso are no longer separated inside the body.

![LibTV depth-motion capture: the person is a flat silhouette](examples/LibTV.png)

**This project.** The crop below comes from this command. The arm toward the camera is the brightest region, and the neck and torso keep their relative-depth shading, closer to OiiOii’s grayscale depth.

```powershell
python -m lumenfigure -i examples/clip2.mp4 -o examples/clip2_output.mp4 --no-matte --depth-size 392 --max-side 1080 --shadow 0.01
```

![Crop of a LumenFigure frame: the near arm is brighter and the torso keeps inner contours](examples/ss3.png)

`--no-matte` shades the whole frame by depth. Background distance remains, without the original color or texture. The default path mattes the person with Robust Video Matting and fills the background with black. Clothing color and print are left out either way; clothes survive only as shape and shading. Both modes discard the source pixels. They differ in whether the background is full-frame depth or cut away.

### Motion-replication example

GitHub plays these three files when you open them. `clip1_result.mp4` is an AI motion replica made from the depth video. Its resolution and frame rate come from the generator, so they differ from the first two files.

| Step | File | Frame | Frame rate | Duration |
| --- | --- | --- | --- | --- |
| Source | [examples/clip1.mp4](examples/clip1.mp4) | 1518×622 | 18 fps | 7.2 s, 129 frames |
| Depth export | [examples/clip1_output.mp4](examples/clip1_output.mp4) | 1080×442 | 18 fps | 7.2 s, 129 frames |
| AI motion replica | [examples/clip1_result.mp4](examples/clip1_result.mp4) | 864×496 | 30 fps | 6.5 s, 195 frames |

The long side of `clip1_output.mp4` is 1080 because `--max-side 1080` scaled the source before inference.

## Requirements

The setup below is not tied to a particular CPU or GPU. Any machine that can install a matching PyTorch build can run it.

| Item | Requirement |
| --- | --- |
| OS | Windows, Linux, or macOS |
| Python | 3.10 or newer. This repository is developed and tested on Python 3.13 |
| GPU | An NVIDIA GPU with a working driver uses CUDA and is much faster. Without one, the CPU path still produces a result, slowly |
| Disk | A CUDA PyTorch wheel is about 2–3 GB. The two weight files are about 15 MB and 111 MB |
| Network | PyPI, the PyTorch wheel index, GitHub Releases, and Hugging Face need to be reachable |

macOS has no CUDA. On a Mac the program uses the CPU.

Export calls `ffmpeg` on `PATH`, and that build must include `libx264`. The file is an H.264 `yuv420p` MP4 at CRF 18, without an audio track, at the input frame rate.

## Setup

Run every command from the repository root: the directory that contains `lumenfigure/`, `requirements.txt`, and `models/`. Start the program with `python -m lumenfigure`. There is no `pip install -e .` step.

### 1. Get the code

```bash
git clone git@github.com:sksk7108/LumenFigure.git
cd LumenFigure
```

### 2. Check Python

```bash
python --version
```

On Windows, if `python` opens the Microsoft Store or the shell cannot find it, use the launcher installed from [python.org](https://www.python.org/downloads/):

```powershell
py -3 --version
```

During installation, enable **Add python.exe to PATH**. The version must be 3.10 or newer. The rest of this document writes `python`. If your machine only has `py -3` or `python3`, substitute that command.

On Linux, if the matching packages are missing:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip
```

Debian/Ubuntu needs `python3-venv` to create a virtual environment. Use `python3` in place of `python` after that.

### 3. Create and activate a virtual environment

The virtual environment keeps PyTorch and the rest of the dependencies away from the system Python.

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell refuses to run the activation script, allow local scripts for the current user and activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

**Windows Command Prompt (cmd)**

```bat
python -m venv .venv
.\.venv\Scripts\activate.bat
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

A successful activation prefixes the prompt with `(.venv)`. `python` and `pip` should then point at this environment:

```bash
python -c "import sys; print(sys.executable)"
python -m pip install -U pip
```

The printed path should be inside this repository’s `.venv`. Leave the environment with `deactivate`.

### 4. Install PyTorch

`requirements.txt` does not install PyTorch. Install `torch` and `torchvision` first, from the **same index**, in versions that match your Python and CUDA. Do not mix a CPU build of one with a CUDA build of the other.

See which GPU the driver reports, and the newest CUDA version that driver supports:

```bash
nvidia-smi
```

`CUDA Version` at the top right is the driver’s upper bound, not the build already installed in Python. Open the [official PyTorch install page](https://pytorch.org/get-started/locally/) and use the command it currently shows. The three commands below are common choices; prefer the page when its versions move on.

CUDA 12.6 wheels:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

CUDA 12.8 wheels:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

CPU only, including macOS:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

A `cu126` wheel is compiled for CUDA 12.6. The driver must support at least that CUDA version. It does not require a specific graphics card. If the driver is older and `torch.cuda.is_available()` stays `False` after installation, install a wheel at or below the driver’s CUDA limit, or update the NVIDIA driver.

Confirm that this Python can import PyTorch and whether CUDA is available:

```bash
python -c "import torch; print(torch.__version__); print('cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

On an NVIDIA GPU this should print `cuda True` and the card name. If it prints `cuda False`, the program can still run with `--device cpu`.

### 5. Install the remaining dependencies

```bash
pip install -r requirements.txt
```

This installs OpenCV, NumPy, tqdm, einops, and easydict.

### 6. Install FFmpeg

Export shells out to `ffmpeg` on `PATH` and encodes with `libx264`. A minimal build without x264 fails when writing the video.

**Windows**

Install a full build with winget:

```powershell
winget install --id Gyan.FFmpeg -e
```

**Open a new terminal** so PATH updates, then activate `.venv` again.

You can also download the release essentials build from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/), unzip it, and add its `bin` directory to the system PATH.

**Debian / Ubuntu**

```bash
sudo apt update
sudo apt install -y ffmpeg
```

**Fedora**

```bash
sudo dnf install -y ffmpeg
```

**macOS**

```bash
brew install ffmpeg
```

Check that the command runs and that `libx264` is one of the encoders:

```bash
ffmpeg -version
```

```bash
ffmpeg -hide_banner -encoders | grep libx264
```

On Windows PowerShell, replace `grep` with `findstr`:

```powershell
ffmpeg -hide_banner -encoders | findstr libx264
```

### 7. Download the weights

Put the checkpoints in `models/`. Weight files in that directory are gitignored, so a fresh clone does not contain them.

| File | Size | Role |
| --- | --- | --- |
| `models/rvm_mobilenetv3.pth` | 15 MB | Robust Video Matting, MobileNetV3. Used by default. Optional when you pass `--no-matte` |
| `models/video_depth_anything_vits.pth` | 111 MB | Video Depth Anything Small, relative depth, Apache-2.0 |

On **Windows PowerShell**, call `curl.exe`. PowerShell aliases `curl` to `Invoke-WebRequest`, which does not take the same flags.

```powershell
New-Item -ItemType Directory -Force -Path models | Out-Null
curl.exe -L --fail -o models\rvm_mobilenetv3.pth https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth
curl.exe -L --fail -o models\video_depth_anything_vits.pth https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth
```

**Linux / macOS**

```bash
mkdir -p models
curl -L --fail -o models/rvm_mobilenetv3.pth https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth
curl -L --fail -o models/video_depth_anything_vits.pth https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth
```

Check the file size when the download finishes. The depth checkpoint is about 111 MB. A file of only a few KB is usually an error page; delete it and download again, or save the URL above from a browser under the same file name.

Use this Small `vits` checkpoint. The Base and Large Video Depth Anything weights are CC-BY-NC-4.0, they are not the default model for this repository, and they are not a drop-in replacement. Details are in [THIRD_PARTY.md](THIRD_PARTY.md).

### 8. Check that the program starts

From the repository root, with the virtual environment active:

```bash
python -m lumenfigure -h
```

A help listing means the code path and Python dependencies import. A real conversion also needs the weights and `ffmpeg`.

## Usage

Input and output are file paths. Missing output directories are created. Write your own results under `output/`. That directory is gitignored, so they stay out of the sample files.

Full-frame depth, with the background kept as grayscale distance:

```bash
python -m lumenfigure -i input.mp4 -o output/depth.mp4 --no-matte --depth-size 392 --max-side 1080 --shadow 0.01
```

The default mattes the person, makes the background black, and still shades the body by depth:

```bash
python -m lumenfigure -i input.mp4 -o output/person.mp4
```

A log line saying `xFormers not available` means the depth model did not load the optional xFormers kernels. Inference continues. You do not need to install xFormers.

Run it from this repository’s root. From the parent directory Python reports `No module named lumenfigure`. Activate `.venv` again in every new terminal.

If CUDA runs out of memory, the current depth pass is discarded and retried: `--depth-size` drops to 392, then to 280. The fallback runs only on an out-of-memory error. It does not change the size on its own to go faster.

## Options

```text
python -m lumenfigure -i INPUT -o OUTPUT [options]
```

### Models and device

| Option | Default | Meaning |
| --- | --- | --- |
| `--rvm-weights` | `models/rvm_mobilenetv3.pth` | Matting checkpoint |
| `--depth-weights` | `models/video_depth_anything_vits.pth` | Depth checkpoint, Small / vits |
| `--device` | `auto` | `auto`, `cuda`, or `cpu`. `auto` uses the GPU when CUDA is available |
| `--fp16` / `--no-fp16` | on | FP16 on CUDA. CPU inference stays FP32 |

### Speed and resolution

| Option | Default | Meaning |
| --- | --- | --- |
| `--depth-size` | `518` | Network input side for depth. Smaller is faster and coarser. 518, 392, and 280 are the usual values, all multiples of 14 |
| `--max-side` | `1920` | Scale the long side to this length before inference. `0` keeps the original resolution |
| `--depth-stride` | `1` | Compatibility flag. Streaming depth has to run every frame, so the value is ignored |

`--max-side` sets the exported resolution. `--depth-size` sets how fine the network’s view is. Raising `--max-side` above the source long side does not upscale the frame.

### Matting and tone

| Option | Default | Meaning |
| --- | --- | --- |
| `--matte` / `--no-matte` | matte | Matting fills the background with black. `--no-matte` shades the whole frame by depth |
| `--highlight` / `--max-luma` | `1.0` | Brightness of the nearest pixels, 0–1 |
| `--shadow` / `--min-luma` | `0.12` | Floor for the farthest person pixels, 0–1, and lower than the highlight |
| `--contrast` | `1.0` | Near/far separation. Above 1 stretches it, below 1 flattens it |
| `--gamma` | `0.85` | Tone curve. Below 1 brightens near surfaces, above 1 holds the highlights down |
| `--temporal-smooth` / `--no-temporal-smooth` | on | Smooth depth and brightness statistics over time to reduce flicker |
| `--blur-depth` | off | Small blur on the depth map for a flatter poster look |
| `--blur-kernel` | `5` | Blur kernel. An even value is rounded up to the next odd number |

Tone adjustments: if people farther back are too dark, raise `--shadow` to `0.25`. If a hand is not bright enough, keep `--highlight 1.0` and lower `--gamma` to `0.7`. If near and far collapse together, raise `--contrast` to `1.4`. These options do not change inference speed.

People are not split into instances. Everyone shares one alpha, and shading comes from per-pixel relative depth. Compositing does not read the source pixels. Soft matte edges stay, so the outline is antialiased. Video Depth Anything keeps a streaming temporal cache, which is steadier than a single-image depth model on each frame. The default temporal smooth pass reduces flicker further.

## Troubleshooting

**`No module named lumenfigure`**

The shell is not in the repository root, or `.venv` is inactive and another Python is running. `cd` into this repository and activate the environment.

**`ffmpeg was not found on PATH`**

Open a new terminal and run `ffmpeg -version`. On Windows, a terminal that was already open does not pick up a PATH change.

**`RVM weights not found` or `Depth weights not found`**

The checkpoint is missing from `models/`, or the file name does not match. `--no-matte` can run without the RVM file. The depth file is always required.

**CUDA out of memory**

The program lowers `depth-size` to 392 and then to 280. If that still fails, use a smaller `--max-side`, or add `--no-matte` to skip the matting network.

**A downloaded `.pth` is only a few KB**

Hugging Face or GitHub returned an error page. `curl --fail` avoids saving that page as a checkpoint. Delete a bad file and download it again.

**The depth result flickers**

Leave `--temporal-smooth` at its default. The streaming model has to see every frame. `--depth-stride` does not skip frames.

## Measured speed

These numbers were measured on the author’s machine. They show how `--depth-size` and `--max-side` change throughput on one clip. Another GPU, driver, or resolution will differ.

| Item | Hardware and software |
| --- | --- |
| CPU | AMD Ryzen 5 5600 |
| GPU | NVIDIA GeForce GTX 1660 Ti, 6 GB |
| OS | Windows |
| Python | 3.13 |
| PyTorch | 2.14.0+cu126, torchvision 0.29.0+cu126 |
| Precision | CUDA FP16, xFormers not installed |

The clip is [examples/clip2.mp4](examples/clip2.mp4): 1280×720, 24 fps, 121 frames. The speed is the progress-bar average and includes reading, inference, and encoding. Repeated runs are listed together. Tone options such as `gamma` and `shadow` did not change the speed.

**Without matting (`--no-matte`)**

| `--depth-size` | `--max-side` | Speed |
| --- | --- | --- |
| 280 | 720 |  6.70 frame/s |
| 280 | 1080 | 6.58 frame/s |
| 392 | 1080 | 3.21 frame/s |
| 518 | 1080 | 1.55 frame/s |

**With matting (the default)**

| `--depth-size` | `--max-side` | Speed |
| --- | --- | --- |
| 280 | 1080 | 4.88 / 4.97 frame/s |

On this 720p clip, `depth-size` 518 measured 1.55 frame/s at both a 1080 and a 1920 long side. The depth network’s input size dominated the time. All of the sizes above finished in 6 GB. A tighter GPU hits the automatic fallback described earlier. The comparison crop used no matting, `depth-size` 392, and `max-side` 1080, which is the 3.21 frame/s row.

## License

[GPL-3.0](LICENSE). The reason, and the upstream project links, are in [THIRD_PARTY.md](THIRD_PARTY.md).
