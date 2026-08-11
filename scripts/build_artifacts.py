from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
NOTEBOOKS = ROOT / "notebooks"
PROJECT_ZIP = ARTIFACTS / "IndexTTS_DubFlow_Source_v0.2.0.zip"
NOTEBOOK = ARTIFACTS / "IndexTTS_DubFlow_Colab_R7_Compact.ipynb"
VISIBLE_LOG_NOTEBOOK = ARTIFACTS / "IndexTTS_DubFlow_Colab_R7.ipynb"
REPO_NOTEBOOK = NOTEBOOKS / NOTEBOOK.name
REPO_VISIBLE_LOG_NOTEBOOK = NOTEBOOKS / VISIBLE_LOG_NOTEBOOK.name
CHECKSUMS = ARTIFACTS / "SHA256SUMS.txt"

EXCLUDED_DIRS = {
    ".git", ".pytest_cache", ".ruff_cache", ".mypy_cache", "__pycache__",
    ".venv", ".runtime", "artifacts", "notebooks", "build", "dist",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def configure_utf8_stdio() -> None:
    """Keep redirected Windows output from falling back to a legacy code page."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass


def source_files() -> list[Path]:
    result: list[Path] = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if any(
            part in EXCLUDED_DIRS or part.endswith(".egg-info")
            for part in relative.parts
        ):
            continue
        if path.is_file() and path.suffix not in EXCLUDED_SUFFIXES:
            result.append(path)
    return sorted(result)


def _write_archive_file(archive: zipfile.ZipFile, relative: Path, payload: bytes) -> None:
    info = zipfile.ZipInfo(str(Path("IndexTTS-DubFlow") / relative).replace(os.sep, "/"))
    info.date_time = (2026, 8, 11, 0, 0, 0)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = (0o755 if relative.name.endswith(".sh") else 0o644) << 16
    archive.writestr(info, payload)


def make_zip_bytes(extra_files: dict[Path, bytes] | None = None) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in source_files():
            _write_archive_file(archive, path.relative_to(ROOT), path.read_bytes())
        for relative, payload in sorted((extra_files or {}).items(), key=lambda item: str(item[0])):
            _write_archive_file(archive, relative, payload)
    return output.getvalue()


def validate_zip_bytes(bundle: bytes) -> None:
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        broken = archive.testzip()
        if broken:
            raise ValueError(f"源码 ZIP 校验失败：{broken}")
        names = archive.namelist()
    if not names or any("/.runtime/" in name or "/.venv/" in name for name in names):
        raise ValueError("源码 ZIP 为空或包含本地运行环境")


def validate_notebook_bytes(payload: bytes, name: str) -> None:
    notebook = json.loads(payload.decode("utf-8"))
    if notebook.get("nbformat") != 4:
        raise ValueError(f"{name} 不是 Notebook v4")
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        compile(source, f"{name}:cell-{index}", "exec")


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def make_notebook(bundle: bytes, *, notebook_name: str, visible_log_edition: bool) -> dict:
    encoded = base64.b64encode(bundle).decode("ascii")
    edition = "（全程可见日志版）" if visible_log_edition else ""
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": notebook_name, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "cells": [
            markdown_cell(
                f"# IndexTTS-DubFlow{edition}\n\n"
                "基于 IndexTTS 2.5 的原声视频翻译与智能配音工作流。\n\n"
                "这个 Notebook 自带完整项目源码，可在 Colab 中启动 Gradio Web 面板。支持上传、路径、Google Drive、"
                "对白/背景分离、歌曲人声保护、多 ASR、热词库、字幕导入、人工译文锁定、智能断句、"
                "HY-MT2-7B 上下文翻译、原音色/情感配音、自然语速优先时间规划、时间轴导入导出和视频输出。\n\n"
                "**Web 构建：2026.08.11-r7-ccd8105**。如果打开面板后没有看到 Qwen3/FunASR/FireRedASR2、"
                "热词库和 HY-MT2，请确认运行的是本 Notebook，而不是旧标签页。\n\n"
                "> 安装与下载每 10 秒至少输出一次状态；Web 启动每 5 秒输出一次状态。任何失败都会显示日志尾部。\n\n"
                "> 推荐 L4/A100/A10。T4 不支持 BF16，程序会自动切到 FP32，速度更慢且可能显存不足。\n\n"
                "> 已跟进 IndexTTS 2.5 正式合并至官方 `main` 的提交 "
                "`ccd81054de9859faeb19b773fff0e2e1ae9e959e`。安装器会验证模型目录中的 2.5 专属结构，"
                "并修复官方 Hub `config.yaml` 当前错误的 2.0 标记与 `/cubefs` 内部路径；"
                "真实的 2.0 配置仍会被拒绝。"
            ),
            markdown_cell("## 1. 运行选项"),
            code_cell(
                "import os, secrets\n"
                "MOUNT_DRIVE = True              # 是否挂载 Google Drive\n"
                "MODEL_SOURCE = 'modelscope'     # 'modelscope' 或 'huggingface'\n"
                "INSTALL_DEMUCS = True           # 推荐：分离原对白与背景\n"
                "INSTALL_SINGING_DETECTOR = True # 保留歌曲人声，只从背景移除对白\n"
                "INSTALL_DIARIZATION = False     # 多人物自动分离；需要 HF token，安装更慢\n"
                "PERSIST_MODEL_CACHE = False     # True 会把模型缓存在云盘（约需 10GB+，以后重开免下载）\n"
                "HYMT_DEFAULT_MODEL = 'tencent/Hy-MT2-7B' # 7B 质量更高；1.8B 约省 12GB 磁盘\n"
                "HYMT_CACHE_ON_DRIVE = False     # True 仅把 HY-MT2 权重放到 Drive，保留本地 Qwen ASR\n"
                "CLEAN_INSTALLER_CACHE = True    # 安装成功后回收 uv/pip 下载缓存，不影响已安装环境\n"
                "DISABLE_HF_XET = True           # 避免大模型下载产生额外 Xet 分块/重建峰值\n"
                "CREATE_PUBLIC_LINK = False      # 默认使用更快、更稳的 Colab 私有代理；True 再建 gradio.live 链接\n"
                "LOG_HEARTBEAT_SECONDS = 10      # 安装静默时每多少秒输出一次心跳（5～60）\n"
                "WEB_START_TIMEOUT = 180         # 本地 Web 服务启动超时（秒）\n"
                "GRADIO_AUTH = globals().get('GRADIO_AUTH') or ('dubber:' + secrets.token_urlsafe(12))\n"
                "if PERSIST_MODEL_CACHE and not MOUNT_DRIVE:\n"
                "    raise ValueError('PERSIST_MODEL_CACHE=True 时必须同时开启 MOUNT_DRIVE')\n"
                "if HYMT_CACHE_ON_DRIVE and not MOUNT_DRIVE:\n"
                "    raise ValueError('HYMT_CACHE_ON_DRIVE=True 时必须同时开启 MOUNT_DRIVE')\n"
                "MODEL_DIR = (\n"
                "    '/content/drive/MyDrive/IndexTTS25_Cache/checkpoints'\n"
                "    if PERSIST_MODEL_CACHE else '/content/index-tts/checkpoints'\n"
                ")\n"
                "RUNTIME_CACHE_DIR = (\n"
                "    '/content/drive/MyDrive/IndexTTS25_Cache/runtime'\n"
                "    if PERSIST_MODEL_CACHE else '/content/IndexTTS25_Runtime_Cache'\n"
                ")\n"
                "ASR_RUNTIME_DIR = (\n"
                "    '/content/drive/MyDrive/IndexTTS25_Cache/asr_runtimes'\n"
                "    if PERSIST_MODEL_CACHE else '/content/IndexTTS25_ASR_Runtimes'\n"
                ")\n"
                "TRANSLATION_RUNTIME_DIR = (\n"
                "    '/content/drive/MyDrive/IndexTTS25_Cache/translation_runtimes'\n"
                "    if PERSIST_MODEL_CACHE else '/content/IndexTTS25_Translation_Runtimes'\n"
                ")\n"
                "HYMT_HF_HOME = (\n"
                "    '/content/drive/MyDrive/IndexTTS25_Cache/hymt_huggingface'\n"
                "    if HYMT_CACHE_ON_DRIVE else os.path.join(RUNTIME_CACHE_DIR, 'huggingface')\n"
                ")\n"
                "print('Web 面板登录凭据：', GRADIO_AUTH)\n"
                "print('模型目录：', MODEL_DIR)\n"
                "print('可选 ASR 缓存：', ASR_RUNTIME_DIR)\n"
                "print('HY-MT2 运行时缓存：', TRANSLATION_RUNTIME_DIR)\n"
                "print('HY-MT2 权重缓存：', HYMT_HF_HOME)\n"
                "print('HY-MT2 默认模型：', HYMT_DEFAULT_MODEL)\n"
                "print('安装日志心跳：每', LOG_HEARTBEAT_SECONDS, '秒')\n"
                "print('公网链接：', '启用' if CREATE_PUBLIC_LINK else '关闭（使用 Colab 私有链接）')\n"
            ),
            markdown_cell("## 2. 解包内嵌项目源码"),
            code_cell(
                "import base64, io, shutil, zipfile\n"
                "from pathlib import Path\n\n"
                f"BUNDLE_B64 = '''{encoded}'''\n"
                "workspace = Path('/content')\n"
                "project = workspace / 'IndexTTS-DubFlow'\n"
                "if project.is_dir():\n"
                "    shutil.rmtree(project)\n"
                "    print('已移除旧项目源码，防止旧 Web/UI 缓存混入。')\n"
                "with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE_B64))) as archive:\n"
                "    archive.extractall(workspace)\n"
                "print('项目已解包：', project)\n"
            ),
            markdown_cell("## 3. 挂载 Google Drive（可选）"),
            code_cell(
                "if MOUNT_DRIVE:\n"
                "    from google.colab import drive\n"
                "    drive.mount('/content/drive')\n"
                "else:\n"
                "    print('跳过 Google Drive')\n"
            ),
            markdown_cell("## 4. 检查 GPU"),
            code_cell(
                "import subprocess\n"
                "subprocess.run(['nvidia-smi'], check=False)\n"
            ),
            markdown_cell(
                "## 5. 安装环境并下载模型\n\n"
                "首次运行会创建独立 Python 3.11/uv 环境并下载数 GB 权重。每个阶段都会显示耗时，"
                "静默命令每 10 秒也会显示一次心跳和目录大小；全部输出同时保存到 "
                "`/content/indextts25_install.log`，中途失败可直接重跑续传。"
            ),
            code_cell(
                "import os, queue, subprocess, threading, time\n"
                "from pathlib import Path\n"
                "env = os.environ.copy()\n"
                "env['PYTHONUNBUFFERED'] = '1'\n"
                "env['PYTHONIOENCODING'] = 'utf-8'\n"
                "env['INDEXTTS_DIR'] = '/content/index-tts'\n"
                "env['INDEXTTS_MODEL_DIR'] = MODEL_DIR\n"
                "env['MODEL_SOURCE'] = MODEL_SOURCE\n"
                "env['INSTALL_DEMUCS'] = '1' if INSTALL_DEMUCS else '0'\n"
                "env['INSTALL_SINGING_DETECTOR'] = '1' if INSTALL_SINGING_DETECTOR else '0'\n"
                "env['INSTALL_DIARIZATION'] = '1' if INSTALL_DIARIZATION else '0'\n"
                "env['CLEAN_INSTALLER_CACHE'] = '1' if CLEAN_INSTALLER_CACHE else '0'\n"
                "env['LOG_HEARTBEAT_SECONDS'] = str(LOG_HEARTBEAT_SECONDS)\n"
                "env['HF_HOME'] = os.path.join(RUNTIME_CACHE_DIR, 'huggingface')\n"
                "env['HF_HUB_DISABLE_XET'] = '1' if DISABLE_HF_XET else '0'\n"
                "env['TORCH_HOME'] = os.path.join(RUNTIME_CACHE_DIR, 'torch')\n"
                "env['UV_CACHE_DIR'] = '/content/IndexTTS25_Installer_Cache/uv'\n"
                "env['PIP_CACHE_DIR'] = '/content/IndexTTS25_Installer_Cache/pip'\n"
                "install_log = Path('/content/indextts25_install.log')\n"
                "started = time.monotonic()\n"
                "print('=' * 72, flush=True)\n"
                "print('开始安装：下面会持续输出阶段、耗时、目录大小和下载进度。', flush=True)\n"
                "print('完整日志：', install_log, flush=True)\n"
                "print('=' * 72, flush=True)\n"
                "process = subprocess.Popen(\n"
                "    ['bash', '/content/IndexTTS-DubFlow/scripts/colab_setup.sh'],\n"
                "    env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,\n"
                "    text=True, encoding='utf-8', errors='replace', bufsize=1,\n"
                ")\n"
                "lines = queue.Queue()\n"
                "def _read_install_output():\n"
                "    assert process.stdout is not None\n"
                "    for line in iter(process.stdout.readline, ''):\n"
                "        lines.put(line)\n"
                "    process.stdout.close()\n"
                "reader = threading.Thread(target=_read_install_output, daemon=True)\n"
                "reader.start()\n"
                "last_output = time.monotonic()\n"
                "with install_log.open('w', encoding='utf-8', buffering=1) as log_handle:\n"
                "    try:\n"
                "        while process.poll() is None or reader.is_alive() or not lines.empty():\n"
                "            try:\n"
                "                line = lines.get(timeout=1)\n"
                "            except queue.Empty:\n"
                "                line = ''\n"
                "            if line:\n"
                "                print(line, end='', flush=True)\n"
                "                log_handle.write(line)\n"
                "                last_output = time.monotonic()\n"
                "            elif process.poll() is None and time.monotonic() - last_output >= LOG_HEARTBEAT_SECONDS:\n"
                "                elapsed = (time.monotonic() - started) / 60\n"
                "                heartbeat = f'[Notebook] ⏳ 安装进程仍在运行，总耗时 {elapsed:.1f} 分钟\\n'\n"
                "                print(heartbeat, end='', flush=True)\n"
                "                log_handle.write(heartbeat)\n"
                "                last_output = time.monotonic()\n"
                "    except KeyboardInterrupt:\n"
                "        process.terminate()\n"
                "        print('已停止本次安装；重新运行此单元会继续使用已下载内容。', flush=True)\n"
                "        raise\n"
                "return_code = process.wait()\n"
                "if return_code != 0:\n"
                "    tail = install_log.read_text(encoding='utf-8', errors='replace')[-12000:]\n"
                "    raise RuntimeError(f'安装失败（退出码 {return_code}）。最后日志：\\n{tail}')\n"
                "print(f'安装单元完成，总耗时 {(time.monotonic() - started) / 60:.1f} 分钟。', flush=True)\n"
            ),
            markdown_cell(
                "## 5.1 磁盘诊断与安全清理（下载失败后运行）\n\n"
                "只回收可重建的 Xet、uv、pip 临时缓存；不会删除 IndexTTS、Qwen ASR、ForcedAligner、"
                "HY-MT2 已完成的 Hub 分片、视频或输出。出现 `No space left on device` 时运行一次，再重启 Web 重试。"
            ),
            code_cell(
                "import os, shutil, subprocess\n"
                "from pathlib import Path\n\n"
                "AUTO_CLEAN_TEMP_CACHE = True\n"
                "GIB = 1024 ** 3\n"
                "def _allocated(path):\n"
                "    if not path.exists():\n"
                "        return 0\n"
                "    total = 0\n"
                "    seen = set()\n"
                "    for item in path.rglob('*'):\n"
                "        try:\n"
                "            stat = item.stat()\n"
                "        except OSError:\n"
                "            continue\n"
                "        if item.is_file() and (stat.st_dev, stat.st_ino) not in seen:\n"
                "            seen.add((stat.st_dev, stat.st_ino))\n"
                "            total += stat.st_blocks * 512 if stat.st_blocks else stat.st_size\n"
                "    return total\n\n"
                "targets = [\n"
                "    Path(HYMT_HF_HOME) / 'xet',\n"
                "    Path(RUNTIME_CACHE_DIR) / 'huggingface' / 'xet',\n"
                "    Path('/root/.cache/huggingface/xet'),\n"
                "    Path('/root/.cache/uv'),\n"
                "    Path('/root/.cache/pip'),\n"
                "    Path('/content/IndexTTS25_Installer_Cache'),\n"
                "]\n"
                "unique_targets = list(dict.fromkeys(path.resolve() for path in targets))\n"
                "reclaimed = 0\n"
                "for target in unique_targets:\n"
                "    size = _allocated(target)\n"
                "    if size:\n"
                "        print(f'可回收：{target} = {size/GIB:.2f} GiB')\n"
                "        if AUTO_CLEAN_TEMP_CACHE:\n"
                "            shutil.rmtree(target)\n"
                "            reclaimed += size\n"
                "os.environ['HF_HUB_DISABLE_XET'] = '1' if DISABLE_HF_XET else '0'\n"
                "print(f'✅ 已回收 {reclaimed/GIB:.2f} GiB 临时缓存。模型和已完成分片均保留。')\n"
                "subprocess.run(['df', '-h', '/content'], check=False)\n"
                "for path in [\n"
                "    '/content/index-tts', ASR_RUNTIME_DIR, TRANSLATION_RUNTIME_DIR,\n"
                "    str(Path(RUNTIME_CACHE_DIR) / 'huggingface'), HYMT_HF_HOME,\n"
                "]:\n"
                "    if Path(path).exists():\n"
                "        subprocess.run(['du', '-sh', path], check=False)\n"
            ),
            markdown_cell(
                "## 6. 启动 Web 面板\n\n"
                "该单元会把 Web 服务放到后台，探测成功后立即显示 **Colab 私有链接**，不再无限等待"
                " Gradio 公网隧道。单元运行结束不代表面板关闭；可用下一单元查看日志或停止服务。"
            ),
            code_cell(
                "import html, json, os, re, socket, subprocess, time, urllib.error, urllib.request\n"
                "from pathlib import Path\n"
                "from IPython.display import HTML, display\n\n"
                "def _free_port(start=7860, stop=7870):\n"
                "    for candidate in range(start, stop + 1):\n"
                "        with socket.socket() as sock:\n"
                "            try:\n"
                "                sock.bind(('127.0.0.1', candidate))\n"
                "            except OSError:\n"
                "                continue\n"
                "            return candidate\n"
                "    raise RuntimeError(f'{start}-{stop} 端口都被占用，请先运行下面的状态/停止单元。')\n\n"
                "def _http_ready(port):\n"
                "    try:\n"
                "        with urllib.request.urlopen(f'http://127.0.0.1:{port}/', timeout=2) as response:\n"
                "            return 200 <= response.status < 500\n"
                "    except urllib.error.HTTPError as exc:\n"
                "        return exc.code in (401, 403)\n"
                "    except Exception:\n"
                "        return False\n\n"
                "web_port = _free_port()\n"
                "web_log = Path('/content/indextts25_web.log')\n"
                "startup_info = Path('/content/indextts25_web_status.json')\n"
                "web_log.write_text('', encoding='utf-8')\n"
                "startup_info.unlink(missing_ok=True)\n"
                "env = os.environ.copy()\n"
                "env['PYTHONUNBUFFERED'] = '1'\n"
                "env['PYTHONIOENCODING'] = 'utf-8'\n"
                "env['GRADIO_ANALYTICS_ENABLED'] = 'False'\n"
                "env['INDEXTTS_DIR'] = '/content/index-tts'\n"
                "env['INDEXTTS_MODEL_DIR'] = MODEL_DIR\n"
                "env['INDEXTTS_CONFIG'] = os.path.join(MODEL_DIR, 'config.yaml')\n"
                "env['MODEL_SOURCE'] = MODEL_SOURCE\n"
                "env['ASR_RUNTIME_ROOT'] = ASR_RUNTIME_DIR\n"
                "env['TRANSLATION_RUNTIME_ROOT'] = TRANSLATION_RUNTIME_DIR\n"
                "env['HF_HOME'] = os.path.join(RUNTIME_CACHE_DIR, 'huggingface')\n"
                "env['HYMT_HF_HOME'] = HYMT_HF_HOME\n"
                "env['HYMT_DEFAULT_MODEL'] = HYMT_DEFAULT_MODEL\n"
                "env['HF_HUB_DISABLE_XET'] = '1' if DISABLE_HF_XET else '0'\n"
                "env['PIP_NO_CACHE_DIR'] = '1'\n"
                "env['TORCH_HOME'] = os.path.join(RUNTIME_CACHE_DIR, 'torch')\n"
                "env['DUBBER_OUTPUT_ROOT'] = (\n"
                "    '/content/drive/MyDrive/IndexTTS_Dubber_Outputs'\n"
                "    if MOUNT_DRIVE else '/content/IndexTTS_Dubber_Outputs'\n"
                ")\n"
                "env['SEPARATE_BACKGROUND_DEFAULT'] = '1' if INSTALL_DEMUCS else '0'\n"
                "env['PROTECT_SINGING_DEFAULT'] = '1' if INSTALL_SINGING_DETECTOR else '0'\n"
                "if GRADIO_AUTH:\n"
                "    env['GRADIO_AUTH'] = GRADIO_AUTH\n"
                "command = [\n"
                "    '/content/index-tts/.venv/bin/python', '-u',\n"
                "    '/content/IndexTTS-DubFlow/app.py',\n"
                "    '--server-name', '0.0.0.0', '--server-port', str(web_port),\n"
                "    '--startup-info', str(startup_info),\n"
                "]\n"
                "if CREATE_PUBLIC_LINK:\n"
                "    command.append('--share')\n"
                "_DUBBER_LOG_HANDLE = web_log.open('w', encoding='utf-8', buffering=1)\n"
                "_DUBBER_WEB_PROCESS = subprocess.Popen(\n"
                "    command, env=env, stdout=_DUBBER_LOG_HANDLE, stderr=subprocess.STDOUT,\n"
                "    start_new_session=True,\n"
                ")\n"
                "_DUBBER_WEB_PORT = web_port\n"
                "Path('/content/indextts25_web.pid').write_text(str(_DUBBER_WEB_PROCESS.pid), encoding='ascii')\n"
                "print(f'Web 子进程 PID={_DUBBER_WEB_PROCESS.pid}，端口={web_port}', flush=True)\n"
                "deadline = time.monotonic() + WEB_START_TIMEOUT\n"
                "next_heartbeat = 0.0\n"
                "last_log_position = 0\n"
                "while time.monotonic() < deadline:\n"
                "    if web_log.exists():\n"
                "        with web_log.open('r', encoding='utf-8', errors='replace') as handle:\n"
                "            handle.seek(last_log_position)\n"
                "            fresh = handle.read()\n"
                "            last_log_position = handle.tell()\n"
                "        if fresh:\n"
                "            print(fresh, end='', flush=True)\n"
                "    return_code = _DUBBER_WEB_PROCESS.poll()\n"
                "    if return_code is not None:\n"
                "        tail = web_log.read_text(encoding='utf-8', errors='replace')[-8000:]\n"
                "        raise RuntimeError(f'Web 进程提前退出（{return_code}）。最后日志：\\n{tail}')\n"
                "    if _http_ready(web_port):\n"
                "        break\n"
                "    now = time.monotonic()\n"
                "    if now >= next_heartbeat:\n"
                "        stage = '等待启动信息'\n"
                "        if startup_info.exists():\n"
                "            try:\n"
                "                stage = json.loads(startup_info.read_text(encoding='utf-8')).get('message', stage)\n"
                "            except Exception:\n"
                "                pass\n"
                "        elapsed = WEB_START_TIMEOUT - max(0, int(deadline - now))\n"
                "        print(f'⏳ Web 仍在启动：{stage}（已等待 {elapsed} 秒）', flush=True)\n"
                "        next_heartbeat = now + 5\n"
                "    time.sleep(0.5)\n"
                "else:\n"
                "    tail = web_log.read_text(encoding='utf-8', errors='replace')[-8000:]\n"
                "    _DUBBER_WEB_PROCESS.terminate()\n"
                "    raise TimeoutError(f'{WEB_START_TIMEOUT} 秒内 Web 端口未就绪。最后日志：\\n{tail}')\n"
                "print('✅ 本地 Web 服务已通过 HTTP 探活；此时尚未加载 IndexTTS 权重。', flush=True)\n"
                "try:\n"
                "    from google.colab.output import eval_js\n"
                "    proxy_url = eval_js(f'google.colab.kernel.proxyPort({web_port})')\n"
                "    display(HTML(\n"
                "        '<p><a style=\"font-size:20px;font-weight:700\" target=\"_blank\" href=\"'\n"
                "        + html.escape(proxy_url, quote=True) + '\">打开 IndexTTS-DubFlow 面板</a></p>'\n"
                "    ))\n"
                "    print('Colab 私有链接：', proxy_url, flush=True)\n"
                "except Exception as exc:\n"
                "    proxy_url = ''\n"
                "    print('无法生成 Colab 私有代理链接：', exc, flush=True)\n"
                "print('登录凭据：', GRADIO_AUTH, flush=True)\n"
                "if CREATE_PUBLIC_LINK:\n"
                "    print('本地面板已经可用，继续等待可选的 gradio.live 公网链接（最多 90 秒）…', flush=True)\n"
                "    public_deadline = time.monotonic() + 90\n"
                "    public_url = ''\n"
                "    while time.monotonic() < public_deadline and _DUBBER_WEB_PROCESS.poll() is None:\n"
                "        content = web_log.read_text(encoding='utf-8', errors='replace')\n"
                "        match = re.search(r'https://[a-zA-Z0-9-]+\\.gradio\\.live', content)\n"
                "        if match:\n"
                "            public_url = match.group(0)\n"
                "            break\n"
                "        time.sleep(1)\n"
                "    if public_url:\n"
                "        display(HTML('<p><a target=\"_blank\" href=\"' + html.escape(public_url, quote=True) + '\">打开公网 Gradio 链接</a></p>'))\n"
                "        print('公网链接：', public_url, flush=True)\n"
                "    else:\n"
                "        print('⚠️ 公网隧道未在 90 秒内返回；上面的 Colab 私有链接仍可正常使用。', flush=True)\n"
            ),
            markdown_cell(
                "## 7. 查看日志或停止 Web（可选）\n\n"
                "默认只查看状态。要关闭后台面板，把 `STOP_WEB` 改成 `True` 后运行。"
            ),
            code_cell(
                "from pathlib import Path\n"
                "import json\n"
                "STOP_WEB = False\n"
                "process = globals().get('_DUBBER_WEB_PROCESS')\n"
                "if STOP_WEB:\n"
                "    if process is not None and process.poll() is None:\n"
                "        process.terminate()\n"
                "        try:\n"
                "            process.wait(timeout=10)\n"
                "        except Exception:\n"
                "            process.kill()\n"
                "        print('Web 面板已停止。')\n"
                "    else:\n"
                "        print('当前没有由本 Notebook 启动的 Web 进程。')\n"
                "else:\n"
                "    status_path = Path('/content/indextts25_web_status.json')\n"
                "    log_path = Path('/content/indextts25_web.log')\n"
                "    if status_path.exists():\n"
                "        print(json.dumps(json.loads(status_path.read_text(encoding='utf-8')), ensure_ascii=False, indent=2))\n"
                "    print('\\n最近日志：')\n"
                "    print(log_path.read_text(encoding='utf-8', errors='replace')[-8000:] if log_path.exists() else '暂无日志')\n"
            ),
            markdown_cell(
                "## 8. 实时跟随 Web 后台日志（可选）\n\n"
                "在面板中开始分析或生成后，可运行这个单元观察后台日志。默认跟随 10 分钟；"
                "按 Colab 的停止按钮只会停止日志跟随，不会停止 Web 面板或正在执行的任务。"
            ),
            code_cell(
                "from pathlib import Path\n"
                "import time\n"
                "FOLLOW_SECONDS = 600\n"
                "log_path = Path('/content/indextts25_web.log')\n"
                "position = 0\n"
                "started = time.monotonic()\n"
                "next_heartbeat = 0.0\n"
                "print(f'开始跟随 {log_path}，最长 {FOLLOW_SECONDS} 秒。', flush=True)\n"
                "try:\n"
                "    while time.monotonic() - started < FOLLOW_SECONDS:\n"
                "        if log_path.exists():\n"
                "            with log_path.open('r', encoding='utf-8', errors='replace') as handle:\n"
                "                handle.seek(position)\n"
                "                fresh = handle.read()\n"
                "                position = handle.tell()\n"
                "            if fresh:\n"
                "                print(fresh, end='', flush=True)\n"
                "        now = time.monotonic()\n"
                "        if now >= next_heartbeat:\n"
                "            process = globals().get('_DUBBER_WEB_PROCESS')\n"
                "            state = '运行中' if process is not None and process.poll() is None else '进程状态未知或已停止'\n"
                "            print(f'[Notebook] ⏳ Web {state}，日志跟随已运行 {int(now-started)} 秒', flush=True)\n"
                "            next_heartbeat = now + 10\n"
                "        time.sleep(1)\n"
                "except KeyboardInterrupt:\n"
                "    print('已停止日志跟随；Web 面板不受影响。', flush=True)\n"
            ),
            markdown_cell(
                "## 使用提醒\n\n"
                "- 必须拥有视频、人物声音和翻译/输出用途的合法授权。\n"
                "- 多人物视频可安装 Pyannote，或在时间轴中手工修改说话人标签。\n"
                "- 默认检测并保留歌曲人声；检测到唱歌的片段会在时间轴中禁用翻译。对白与歌声完全重叠时仍需人工复核。\n"
                "- 可上传 SRT/VTT/ASS/SSA 作为原文或最终译文；最终译文会直接锁定，不会重复翻译。\n"
                "- 分析后可导出 JSON + 原文/译文 SRT；人工校对后可把 JSON/SRT/VTT/ASS/SSA 重新上传并锁定。\n"
                "- FireRed、Qwen3-ASR、Fun-ASR-Nano 首次被选择时才建立隔离环境并下载模型，进度写入 Web 日志。\n"
                "- HY-MT2-7B 原始权重为 16.1 GB；T4 的 4-bit 只省显存。R7 默认禁用 Xet 临时重建，下载前会做空间预检。\n"
                "- 磁盘仍紧张时可把 HYMT_CACHE_ON_DRIVE=True，或在 Web 选择官方 Hy-MT2-1.8B（约 4 GB）；Qwen ASR/Aligner 无需删除。\n"
                "- ASR 会尝试合并被截断的半句，并优先按句末标点、分句标点和静音切分；上传字幕不自动改动时间轴。\n"
                "- TTS 先保留自然语速并借用下一句前的真实空白；放不下时才压缩，警告只提示明显压缩的句子。\n"
                "- 逐句总时长严格一致不等于逐音素唇形同步。\n"
                "- 详细架构、当前官方分支问题和许可证见项目 `README.md`。"
            ),
        ],
    }


def main() -> None:
    configure_utf8_stdio()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    embedded_bundle = make_zip_bytes()
    notebook_bytes = json.dumps(
        make_notebook(embedded_bundle, notebook_name=NOTEBOOK.name, visible_log_edition=False),
        ensure_ascii=False,
        indent=1,
    ).encode("utf-8")
    visible_log_notebook_bytes = json.dumps(
        make_notebook(
            embedded_bundle,
            notebook_name=VISIBLE_LOG_NOTEBOOK.name,
            visible_log_edition=True,
        ),
        ensure_ascii=False,
        indent=1,
    ).encode("utf-8")
    bundle = make_zip_bytes(
        {
            Path("notebooks") / NOTEBOOK.name: notebook_bytes,
            Path("notebooks") / VISIBLE_LOG_NOTEBOOK.name: visible_log_notebook_bytes,
        }
    )
    validate_zip_bytes(bundle)
    validate_notebook_bytes(notebook_bytes, NOTEBOOK.name)
    validate_notebook_bytes(visible_log_notebook_bytes, VISIBLE_LOG_NOTEBOOK.name)
    checksums_bytes = "".join(
        f"{hashlib.sha256(payload).hexdigest()}  {path.name}\n"
        for path, payload in (
            (PROJECT_ZIP, bundle),
            (VISIBLE_LOG_NOTEBOOK, visible_log_notebook_bytes),
            (NOTEBOOK, notebook_bytes),
        )
    ).encode("utf-8")
    if "--check" in sys.argv[1:]:
        if not PROJECT_ZIP.is_file() or PROJECT_ZIP.read_bytes() != bundle:
            raise SystemExit("项目 ZIP 已过期，请重新运行 scripts/build_artifacts.py")
        if not NOTEBOOK.is_file() or NOTEBOOK.read_bytes() != notebook_bytes:
            raise SystemExit("Colab Notebook 已过期，请重新运行 scripts/build_artifacts.py")
        if (
            not VISIBLE_LOG_NOTEBOOK.is_file()
            or VISIBLE_LOG_NOTEBOOK.read_bytes() != visible_log_notebook_bytes
        ):
            raise SystemExit("可见日志版 Colab Notebook 已过期，请重新运行 scripts/build_artifacts.py")
        if not REPO_NOTEBOOK.is_file() or REPO_NOTEBOOK.read_bytes() != notebook_bytes:
            raise SystemExit("仓库内精简版 Colab Notebook 已过期，请重新运行 scripts/build_artifacts.py")
        if (
            not REPO_VISIBLE_LOG_NOTEBOOK.is_file()
            or REPO_VISIBLE_LOG_NOTEBOOK.read_bytes() != visible_log_notebook_bytes
        ):
            raise SystemExit("仓库内可见日志版 Colab Notebook 已过期，请重新运行 scripts/build_artifacts.py")
        if not CHECKSUMS.is_file() or CHECKSUMS.read_bytes() != checksums_bytes:
            raise SystemExit("SHA256SUMS.txt 已过期，请重新运行 scripts/build_artifacts.py")
        print("交付文件与当前源码一致。")
        return
    PROJECT_ZIP.write_bytes(bundle)
    NOTEBOOK.write_bytes(notebook_bytes)
    VISIBLE_LOG_NOTEBOOK.write_bytes(visible_log_notebook_bytes)
    REPO_NOTEBOOK.write_bytes(notebook_bytes)
    REPO_VISIBLE_LOG_NOTEBOOK.write_bytes(visible_log_notebook_bytes)
    CHECKSUMS.write_bytes(checksums_bytes)
    print(PROJECT_ZIP)
    print(NOTEBOOK)
    print(VISIBLE_LOG_NOTEBOOK)
    print(CHECKSUMS)


if __name__ == "__main__":
    main()
