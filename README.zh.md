# Eistara

Eistara 是一个视频翻译与中文配音工作流，包含视频下载、转录、翻译、原声分离、TTS 调用、混音、字幕和最终视频输出。Eistara 本体只维护工作流和 Python 运行环境，不接管宿主机的 FFmpeg、CUDA、CUDNN，也不内置 TTS 服务。

## 安装前准备

在 Windows 上使用本地 GPU 加速前，建议先完成宿主机依赖安装：

1. 安装 Python 3.10；如果系统没有 Python 3.10，`setup_env.py` 会尝试通过 uv 自动拉取 Python 3.10 创建 `.venv`。
2. 安装 FFmpeg，并确保 `ffmpeg` 和 `ffprobe` 可在命令行直接运行。
3. 安装 NVIDIA Driver。
4. 如本机 GPU 生态需要，安装 [CUDA Toolkit 12.6](https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda_12.6.0_560.76_windows.exe)。
5. 如本机 GPU 生态需要，安装 [CUDNN 9.3.0](https://developer.download.nvidia.com/compute/cudnn/9.3.0/local_installers/cudnn_9.3.0_windows.exe)。
6. 单独部署并启动 IndexTTS 服务。

FFmpeg 可以通过包管理器安装：

```powershell
choco install ffmpeg
```

如果不用 Chocolatey，也可以自行下载安装 FFmpeg，并把 `bin` 目录加入系统 `PATH`。

## 安装 Eistara

克隆仓库：

```powershell
git clone https://github.com/meakyahu-cyber/Eistara.git
cd Eistara
```

创建虚拟环境并安装依赖：

```powershell
python setup_env.py
```

如果只想先验证依赖安装，不下载模型：

```powershell
python setup_env.py --skip-models
```

如果明确使用 CUDA 12.8 PyTorch wheel：

```powershell
python setup_env.py --torch cu128
```

## 模型缓存

默认模型：

- ASR：`Systran/faster-whisper-large-v3`
- 中文 ASR 可选：`Huan69/Belle-whisper-large-v3-zh-punct-fasterwhisper`
- 人声分离默认：Demucs `htdemucs`

默认使用 `hf-mirror.com` 和清华 PyPI 源以改善国内下载体验。TTS 模型不由 Eistara 下载或管理。

## 启动

```powershell
start_eistara.bat
```

WebUI 默认地址：

```text
http://localhost:10127
```

在 WebUI 中填写 LLM `base_url`、`key` 和 `model`。这些设置会写入本地 `config.local.yaml`，该文件不会提交到 Git。

## 边界说明

Eistara 安装器负责：

- 创建 `.venv`
- 安装 Python 包
- 准备非 TTS 模型缓存
- 创建本地配置模板
- 检测宿主机 FFmpeg / NVIDIA Driver

Eistara 安装器不负责：

- 安装 FFmpeg
- 安装 CUDA Toolkit
- 安装 CUDNN
- 安装或启动 IndexTTS
- 下载 TTS 模型
