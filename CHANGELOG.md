# Changelog

所有重要变更记录在此文件。版本号遵循 Semantic Versioning。

## [0.2.0] - 2026-08-11

### Added

- Linux、Windows PowerShell、WSL2 与 macOS 共用的 Python 安装器；
- 跨平台虚拟环境路径解析；
- 单一运行目录、模型缓存和输出目录约定；
- GitHub Actions 三平台 CI、标签发布与 Issue/PR 模板；
- 完整安装、配置、架构、排障和 GitHub 发布文档；
- 可复现源码 ZIP、R7 Colab Notebook 与 SHA256 清单。

### Changed

- 项目正式命名为 `IndexTTS-DubFlow`，统一仓库、Python 包、Notebook 与 Release 交付件名称；
- README 增加 GitHub Camo 兼容徽章、直接 Colab 入口和规范化项目说明；
- Web 构建号更新为 `2026.08.11-r7-ccd8105`；
- IndexTTS 2.5 正式提交固定为 `ccd81054...`；
- Windows 原生隔离环境使用 `.venv/Scripts/python.exe`；
- 模型下载不再依赖 POSIX 专属 CLI 路径。

### Fixed

- Gradio 可选文本框返回 `None` 时的配置构建错误；
- 官方 2.5 Hub 配置误标 `version: 2.0` 与 `/cubefs` 内部路径；
- HY-MT2/Xet/uv/pip 重复缓存导致的磁盘峰值；
- FFmpeg 旧版 `alimiter` 延迟兼容；
- R6 仅适配 Colab 的绝对路径与虚拟环境路径。
