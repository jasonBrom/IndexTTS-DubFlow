from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IndexTTS-DubFlow Web UI")
    parser.add_argument("--share", action="store_true", help="Create a temporary Gradio share URL")
    parser.add_argument("--server-name", default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument(
        "--auth",
        default=os.environ.get("GRADIO_AUTH", ""),
        help="Optional username:password for the Web UI",
    )
    parser.add_argument(
        "--startup-info",
        default=os.environ.get("DUBBER_STARTUP_INFO", ""),
        help="Optional JSON file used by the Colab launcher to report startup stages",
    )
    return parser.parse_args()


def _write_startup_info(path: str, stage: str, **values: Any) -> None:
    if not path:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pid": os.getpid(),
        "stage": stage,
        "updated_at": time.time(),
        **values,
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(destination)


def _stage(path: str, stage: str, message: str, **values: Any) -> None:
    print(f"[Web] {message}", flush=True)
    _write_startup_info(path, stage, message=message, **values)


def main() -> None:
    args = parse_args()
    try:
        _stage(args.startup_info, "bootstrap", "启动引导程序")
        auth = None
        if args.auth:
            if ":" not in args.auth:
                raise ValueError("--auth 必须是 username:password")
            auth = tuple(args.auth.split(":", 1))
        elif args.share:
            print(
                "警告：正在创建无密码的公开链接。建议通过 GRADIO_AUTH=username:password 设置登录保护。",
                flush=True,
            )

        _stage(args.startup_info, "import_ui", "导入 Gradio 与项目界面")
        from original_dubber.ui import build_app

        _stage(args.startup_info, "build_ui", "构建 Web 面板")
        app = build_app()
        queued_app = app.queue(default_concurrency_limit=1, max_size=8)
        local_probe_url = f"http://127.0.0.1:{args.server_port}"
        if args.share:
            _stage(
                args.startup_info,
                "launching",
                f"监听端口 {args.server_port}；本地服务就绪后还会建立 Gradio 公网隧道",
                local_url=local_probe_url,
            )
        else:
            _stage(
                args.startup_info,
                "launching",
                f"监听端口 {args.server_port}",
                local_url=local_probe_url,
            )
        launch_result = queued_app.launch(
            share=args.share,
            server_name=args.server_name,
            server_port=args.server_port,
            auth=auth,
            show_error=True,
            prevent_thread_lock=True,
        )
        local_url = getattr(app, "local_url", None) or local_probe_url
        share_url = getattr(app, "share_url", None)
        if isinstance(launch_result, tuple):
            if len(launch_result) > 1 and launch_result[1]:
                local_url = str(launch_result[1])
            if len(launch_result) > 2 and launch_result[2]:
                share_url = str(launch_result[2])
        _stage(
            args.startup_info,
            "ready",
            "Web 面板已就绪",
            local_url=local_url,
            share_url=share_url,
        )
        print(f"LOCAL_URL={local_url}", flush=True)
        if share_url:
            print(f"PUBLIC_URL={share_url}", flush=True)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            _stage(args.startup_info, "stopping", "正在关闭 Web 面板")
            app.close()
            _write_startup_info(args.startup_info, "stopped", message="Web 面板已关闭")
    except BaseException as exc:
        _write_startup_info(
            args.startup_info,
            "error",
            message=str(exc),
            error_type=type(exc).__name__,
        )
        print(f"[Web] 启动失败：{type(exc).__name__}: {exc}", flush=True)
        raise


if __name__ == "__main__":
    main()
