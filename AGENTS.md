# AGENTS.md

这个仓库是一个长期维护的 fork。目标是尽量贴近上游，同时允许保留本地定制修改。

## 分支规则

- `main` 视为与上游对齐的分支，不直接在 `main` 上开发。
- 日常开发放在 `dev`，或从 `dev` 切出的短期功能分支上。
- `origin` 应指向 `https://github.com/sc-hua/faster-qwen3-tts`。
- `upstream` 应指向 `https://github.com/andimarafioti/faster-qwen3-tts.git`。

## 同步规则

从上游同步时，使用下面这组流程：

```bash
git fetch upstream
git switch main
git merge --ff-only upstream/main
git push origin main
git switch dev
git rebase main
```

如果 `dev` 是多人共享分支，优先使用 `git merge main`，不要使用 `git rebase main` 改写公共历史。

## 改动规则

- 开始改代码前，先确认当前不在 `main` 分支。
- 改动要聚焦，不要把工作流清理、格式化和功能修改混在同一个变更里，除非任务确实要求。
- 不要随意改动对外行为，除非任务明确要求。
- 尽量保持现有代码风格和结构，除非确实有必要重构。
- 不要回滚和当前任务无关的用户改动。

## PR 规则

- 如果某个改动未来可能提交给上游，请从 `main` 新切一个干净分支，不要从 `dev` 切。
- 面向上游的改动要尽量小、尽量容易审阅。
- fork 专属改动可以保留在 `dev` 或基于 `dev` 的功能分支上。

## 验证要求

- 只运行和当前改动最相关、最小范围的测试。
- 如果没有运行测试，需要明确说明。
- 如果存在风险、后续工作或验证缺口，要直接指出，不要默认一切正确。

## Agent 执行要求

- 优先修复根因，不要只做表面补丁。
- 不要编辑生成文件或无关文件，除非任务确实需要。
- 描述改动时，要说明它属于“同步上游”还是“fork 专属修改”。
- 如果当前分支状态不明确，先检查 `git status --short --branch` 和 `git remote -v`，再开始改动。