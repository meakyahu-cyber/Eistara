# Eistara

Eistara 是一个偏 Windows 本地运行的视频翻译与中文配音工作流。它通过
WebUI 和 CLI 串起视频获取、ASR 转录、LLM 翻译、Demucs 人声分离、
IndexTTS 配音、字幕生成、音频混合和最终视频渲染。

英文说明：[README.md](README.md)

发包版安装说明：[README_RELEASE.md](README_RELEASE.md)

## 运行环境

正常本地配音建议按 Windows + NVIDIA GPU 环境准备。CPU 模式只适合有限调试，
不是推荐的发包运行方式。

运行 `setup_env.py` 前，先安装这些宿主机依赖。普通用户保持默认安装路径即可：

- Python 3.10.x
- NVIDIA Driver
- [CUDA Toolkit 12.8](https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_571.96_windows.exe)
- [CUDNN 9.11.0](https://developer.download.nvidia.com/compute/cudnn/9.11.0/local_installers/cudnn_9.11.0_windows.exe)
- FFmpeg
- IndexTTS 服务需要单独启动，默认 API 地址：
  `http://127.0.0.1:8010/tts`

推荐用 Chocolatey 安装 FFmpeg：

```powershell
choco install ffmpeg
```

安装完成后，关闭旧 PowerShell，重新打开一个 PowerShell，运行：

```powershell
python --version
ffmpeg -version
ffprobe -version
nvidia-smi
nvcc --version
where cudnn64_9.dll
```

这些命令能显示版本号或 CUDNN 路径，就可以继续运行 `python setup_env.py`。

只有检查失败时，才需要处理 `Path`：

- 如果 `nvcc --version` 失败，先重新打开 PowerShell，或者重装 CUDA Toolkit
  12.8。仍然失败时，再把
  `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin` 加入系统
  `Path`。
- 如果 `where cudnn64_9.dll` 失败，把 CUDNN 的 `bin` 目录加入系统 `Path`。
  默认安装时通常是 `C:\Program Files\NVIDIA\CUDNN\v9.11\bin\12.9`。
- 如果手动解压安装 FFmpeg 后 `ffmpeg -version` 失败，把 FFmpeg 的 `bin`
  目录加入系统 `Path`。

通常不需要手动修改 `CUDA_PATH`；CUDA Toolkit 安装器会自动写好。

Eistara 安装脚本会创建 `.venv`、安装 Python 依赖、准备非 TTS 模型缓存，
并从 `config.example.yaml` 创建 `config.local.yaml`。它不负责安装 FFmpeg、
NVIDIA Driver、CUDA Toolkit、CUDNN、IndexTTS，也不负责下载或管理 TTS 模型。

## 安装

```powershell
git clone https://github.com/meakyahu-cyber/Eistara.git
cd Eistara
python setup_env.py
```

常用安装选项：

```powershell
python setup_env.py --skip-models
python setup_env.py --torch cpu
python setup_env.py --torch cu128
python setup_env.py --with-zh-asr
python setup_env.py --no-china-mirror
```

默认情况下，pip 使用清华 PyPI 镜像，HuggingFace 下载使用 `hf-mirror.com`，
PyTorch wheel 仍使用官方 PyTorch wheel 源。

## 启动

```powershell
start_eistara.bat
```

WebUI 默认地址：

```text
http://localhost:10127
```

在 WebUI 中填写 LLM `base_url`、`key` 和 `model`。这些设置会写入本地
`config.local.yaml`，该文件不会提交到 Git。

## 默认运行策略

`config.example.yaml` 是复制到 `config.local.yaml` 的配置模板。

- 翻译批次默认每批 20 条。
- ASR 默认使用 `Systran/faster-whisper-large-v3`。
- 可选中文 ASR 缓存：`Huan69/Belle-whisper-large-v3-zh-punct-fasterwhisper`。
- 人声分离使用 Demucs `htdemucs`。
- 配音时间线默认使用 `source_window`。
- source-window retime 默认启用借窗和局部音频加速。
- IndexTTS 自适应 source-window duration-control retry 默认开启，只对仍然
  需要兜底的片段生效。
- 二档 retime 默认关闭。
- 背景音默认使用自适应宽频 ducking。

## 任务目录

进行中的任务放在 `jobs/<job_id>`。

完成后的任务归档到 `history/<视频标题>`。归档根目录放用户真正需要的交付物，
例如源视频、配音视频和各式字幕；完整可恢复的工作树放在
`history/<视频标题>/work`。

较大的跨阶段交接数据会放在 `output/internal`，`state.json` 只保留计数和
JSON 路径。例如后续阶段可以从 `output/internal/tts_segments.json` 恢复
TTS 输入，即使持久化的 job state 里没有内联大数组。

## CLI 检查

日常使用以 WebUI 为主；CLI 主要用于健康检查和调试：

```powershell
python -m apps.cli.main --jobs-dir .\jobs health
python -m apps.cli.main --jobs-dir .\jobs status
python -m apps.cli.main --jobs-dir .\jobs events
python -m apps.cli.main --jobs-dir .\jobs stages
python -m apps.cli.main --config .\config.local.yaml --jobs-dir .\jobs run-once --preset production
```

## 代码结构

- `apps.webui`：Streamlit 运行界面。
- `apps.cli`：健康检查、调度器、交付物和调试 CLI。
- `eistara.runtime`：生产流水线装配和运行时健康检查。
- `eistara.config`：默认配置、配置加载和类型化设置构建。
- `eistara.core.jobs`：任务状态、JSON 任务存储和归档。
- `eistara.core.scheduler`：阶段调度、锁、心跳和恢复。
- `eistara.core.source`：本地文件和 URL 源获取。
- `eistara.core.asr`：ASR 请求/结果模型和转录阶段。
- `eistara.core.translation`：分批、提示词、校验和发布。
- `eistara.core.tts`：TTS 请求模型、缓存、重试服务和文本清洗。
- `eistara.core.timeline`：source-window 时间线准备。
- `eistara.core.dubbing`：音频放置、retime、混音和渲染流程。
- `eistara.core.delivery`：面向用户的视频和字幕交付物。
- `eistara.adapters`：ASR、TTS、LLM、媒体和源获取适配器边界。
- `eistara.core.diagnostics`：可选本地诊断 hook，默认关闭，不参与核心流程行为。
