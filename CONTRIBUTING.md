# 贡献指南

感谢参与。提交代码前请先阅读许可证、第三方声明与安全策略。

## 开发环境

```bash
python -m pip install -e '.[test]'
python -m compileall -q app.py original_dubber scripts tests
pytest -q
ruff check .
```

轻量测试不得下载大模型。需要 GPU/模型的验证应标记为手工 smoke test，并在 PR 中写明 GPU、显存、系统、驱动、CUDA、输入时长和结果。

## Pull Request

- 每个 PR 聚焦一个问题；
- 新平台代码必须使用 `pathlib` 和参数列表调用子进程；
- 不得写死 `/content`、`.venv/bin` 或 Windows 盘符，Colab 专用 Notebook 除外；
- 不得记录 Token、API 密钥、视频路径或人物隐私数据；
- 修改固定 IndexTTS 提交或补丁时，必须验证正向/反向 `git apply --check`；
- 更新用户可见行为时同步 README、CHANGELOG 和测试。

## 提交信息

推荐使用 Conventional Commits，例如：

```text
fix: support Windows venv executable paths
docs: add WSL2 installation guide
feat: add a new ASR backend
```

