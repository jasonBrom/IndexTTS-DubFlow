# GitHub 发布清单

## 首次创建仓库

1. 在 GitHub 创建空仓库，不勾选自动生成 README/License；
2. 解压本项目源码包；
3. 在项目根目录运行：

```bash
git init
git add .
git commit -m "feat: release IndexTTS-DubFlow v0.3.0"
git branch -M main
git remote add origin https://github.com/jasonBrom/IndexTTS-DubFlow.git
git push -u origin main
```

## 仓库设置

- Actions：允许 GitHub Actions；
- Branch protection：要求 `CI / test` 通过后合并；
- Issues：启用；
- Discussions：可选；
- Releases：由标签工作流自动生成；
- 不要提交 `.runtime`、模型、视频、日志、Token 或 API 密钥。

## 发布前检查

```bash
python -m compileall -q app.py original_dubber scripts tests
pytest -q
ruff check .
python scripts/setup_runtime.py --dry-run
python scripts/build_artifacts.py
python scripts/build_artifacts.py --check
```

确认 `artifacts/SHA256SUMS.txt` 与生成文件一致，并在至少一个 NVIDIA CUDA 环境完成最小 GPU smoke test。

## 创建版本

```bash
git status --short
git tag -a v0.3.0 -m "IndexTTS-DubFlow v0.3.0"
git push origin v0.3.0
```

`.github/workflows/release.yml` 会构建并上传：

- GitHub-ready 源码 ZIP；
- R8 可见日志版 Colab Notebook；
- R8 精简版 Colab Notebook；
- SHA256 校验文件。

## 版本策略

- `MAJOR`：配置/任务数据不兼容；
- `MINOR`：新增平台、模型或工作流；
- `PATCH`：兼容性、文档与缺陷修复；
- Web 构建号与 Notebook 文件名必须同步更新；
- 改动固定上游提交时，必须重新验证补丁可正向应用并更新测试。
