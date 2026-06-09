# 1CatQA 开发日志

最后更新：`2026-06-09`

## 1. 项目定位

- 项目正式名称：`1Cat-V100-QA`
- 当前主线版本：`0.2.6`
- 目标硬件：`Tesla V100`
- 主要优化环境：`CUDA 12`
- 当前默认桌面入口：纯文字终端界面 `gpuqa.gui_entry`
- **0.2.6 新增**：HTTP API 服务器 / Agent 集成 / Webhook 回调

说明：
- `0.2.6` 为当前主线，基于 `0.2.5` 增加 Agent/API 支持
- `0.2.5` 保留 `NVLink CRC` 检测并降级为"良好"
- `0.2.5b` 为分支版，继承 `0.2.4b` 的 CRC 移除逻辑

## 2. 分支约定

- `codex/main`
  当前主线，对应 `0.2.4`
- `codex/0.2.4`
  与主线内容一致，便于按版本名切换
- `codex/0.2.4b`
  去掉 `NVLink CRC` 检测的分支版

## 3. 已完成的核心改动

### 3.1 基础定位与发布形态

- 项目名称改为 `1Cat-V100-QA`
- 文档中明确声明该软件是 `V100` 专项 QA 工具
- 默认定位为 `CUDA 12` 环境
- 保留内部 Python 包名 `gpuqa`，避免导入路径和旧脚本失效

目的：
- 对外命名清晰
- 避免内部包名变更引发部署链路大面积断裂

### 3.2 GPU-Burn 工具链

- 删除了自研 `mini-gpu-burn`
- 改为优先使用内置 vendored `gpu-burn`
- 内置了 `Ubuntu 24.04 / CUDA 12` 预编译 `gpu-burn`
- 默认 `gpu-burn` 追加 `-m 100%`，尽量占满显存
- 默认 `gpu-burn` 时长已统一改为 `600 秒`

默认优先级：
1. `src/gpuqa/vendor/gpu-burn/ubuntu24.04-cuda12/usr/sbin/gpu-burn`
2. `/usr/sbin/gpu-burn`
3. `/opt/gpu-burn/bin/gpu_burn`

目的：
- 避免系统自带 `gpu_burn` 损坏或路径不一致
- 保证 `CUDA 12` 机器可直接运行

### 3.3 NVLink 带宽测试

- 默认 `NVLink` 压测时长为 `10 秒`
- 内置 `cuda/p2p_bandwidth_matrix.cu`
- 自动编译缓存到用户目录
- 修复了多 GPU 带宽汇总问题

当前逻辑：
- 先解析完整 `Bandwidth Matrix`
- 每张 GPU 的摘要带宽取“最高有效 peer 带宽”
- 完整矩阵会写入 artifacts：`p2p_bandwidth_matrix`

目的：
- 避免 `4/8 GPU` 时被低速 PCIe peer 平均掉，导致看起来像只测了双卡

### 3.4 测试顺序与门禁

- 默认先跑 `gpu-burn`
- 只有 `gpu-burn` 正常通过，才执行 `NVLink` 带宽测试
- 单卡模式下自动跳过 `NVLink` 带宽测试
- `gpu-burn WARNING/ERROR` 非零会直接视为硬失败

目的：
- 保证先做热稳定和计算稳定性，再做链路验证

### 3.5 ECC / NVLink / CRC 检测

- 增加 ECC 模式检测
- 增加 ECC 单比特 / 双比特启动前、结束后计数采集
- 增加 `NVLink` 通道数检测
- 增加 `NVLink` 错误计数检测
- `0.2.4` 主线保留 `NVLink CRC` 检测
- `0.2.4b` 分支移除 `NVLink CRC` 检测

目的：
- 提前识别机器硬件状态异常
- 保留专项 QA 所需的链路和 ECC 维度

### 3.6 评级逻辑

这是最近一次重点调整。

当前对外显示评级：
- `优秀`
- `良好`
- `不通过`

内部运行状态仍保留：
- `PASS`
- `FAIL`
- `ERROR`
- `NOT_RUN`

当前规则：
- 没有 `NVLink` 错误增量时：`优秀`
- `0.2.4` 中如果发现 `NVLink` 或 `NVLink CRC` 错误增量：不中断检测，评级降为 `良好`
- `0.2.4b` 中如果发现 `NVLink` 错误增量：不中断检测，评级降为 `良好`
- `ECC` 报错、温度超阈值、`gpu-burn` 退出异常、`gpu-burn WARNING/ERROR` 非零、驱动异常、采样失败等仍为硬失败，显示为 `不通过`

说明：
- CLI 返回码和内部自动化逻辑仍依赖 `PASS/FAIL/ERROR/NOT_RUN`
- GUI、TUI、summary 对外显示改成了 `优秀 / 良好 / 不通过`

目的：
- 把链路层软问题和硬失败分开，减少误杀
- 同时保持自动化脚本兼容

### 3.7 文字终端界面

- 桌面默认入口已改为文字 TUI，不再默认走旧 Tk GUI
- 支持自动按 GPU 数量切换布局

布局规则：
- `1-2 GPU`：完整卡片
- `3 GPU`：三列压缩卡片
- `4 GPU`：`2x2` 压缩卡片
- `5-7 GPU`：三列多行压缩卡片
- `8+ GPU`：单行列表

TUI 已支持：
- `R` 开始
- `S` 停止
- `T` 改温度阈值
- `B` 改 `gpu-burn` 时长
- `N` 改 `NVLink` 时长
- `G` 改目标 GPU
- `O` 打开结果目录
- `Q` 退出

显示内容：
- 型号、序列号、UUID、PCI Bus ID
- 温度、当前功耗、峰值功耗、利用率、显存
- `GPU-Burn GF/s`
- `gpu-burn` 告警/错误计数
- ECC 单比特/双比特前后值
- `NVLink` 通道数
- `NVLink` 带宽
- `0.2.4` 中可带 `CRC` 信息
- 最终评级和备注

### 3.8 旧 GUI

- 仓库仍保留 `src/gpuqa/gui.py`
- 已同步适配最终评级显示
- 当前默认发布入口不是它，而是文字终端界面

### 3.9 停止逻辑

- 停止按钮/停止操作会结束整组 `gpu-burn` 进程
- 不仅终止父进程，也会清理子进程

目的：
- 避免 `gpu-burn` 在后台残留继续吃卡

### 3.10 报告与产物

输出：
- `summary.txt`
- `summary.html`
- `summary.md`
- `summary.csv`
- `summary.json`
- `samples.csv`

已完成：
- 报告中文化
- 桌面界面完成后直接显示总结
- summary 中使用最终评级显示
- artifacts 中保留关键环境与运行信息

## 4. 关键默认参数

- 默认 `gpu-burn` 时长：`600`
- 默认 `NVLink` 时长：`10`
- 默认采样间隔：`5`
- 默认温度阈值：`80C`

### 0.2.5b

- 默认温度阈值改为 `80C`
- `gpu-burn` 过程中一旦超温会自动停测，并立即进入失败报告输出
- `NVLink` 带宽测试前移到 `gpu-burn` 之前
- `NVLink` 错误增量和链路不足统一只降级为 `良好`，不影响整机过测
- 修复 TUI 面板顶部边框颜色显示异常
- 默认显存占用：`-m 100%`
- 默认 `NVLink` 期望通道数：`6`

## 5. 当前必须记住的路径

### 本地仓库

- 根目录：`C:\\Users\\24241\\Documents\\1CatAI-GPUQA`

### 远端用户安装结构

统一采用：
- `~/.local/share/1cat-v100-qa/releases/<stamp>`
- `~/.local/share/1cat-v100-qa/current`
- `~/.local/bin/1cat-v100-qa`
- `~/.local/bin/1cat-v100-qa-gui`
- `~/桌面/1Cat-V100-QA`
- `~/桌面/1Cat-V100-QA.desktop`

## 6. 服务器角色记忆

### 测试机

- `192.168.50.32`
- 用户：`onecatai`

### 生产机

- `192.168.50.30`
- 用户：`user`

- `192.168.50.34`
- 用户：`user-test`

安全说明：
- 密码不要写入仓库或日志
- 连接凭据单独管理

## 7. 最近一次版本差异说明

### 0.2.4

- 保留 `NVLink CRC` 检测
- `NVLink / CRC` 错误只降级为 `良好`
- 硬错误仍显示为 `不通过`

### 0.2.4b

- 删除 `NVLink CRC` 检测
- 只保留 `NVLink` 错误软降级逻辑
- 其它硬错误保持与主线一致

## 8. 当前开发建议

- 后续如果再改“评级逻辑”，优先改 `src/gpuqa/models.py` 里的评级辅助函数，再同步 `reporting/gui/tui`
- 后续如果再改多卡 `NVLink` 带宽逻辑，优先检查 `src/gpuqa/core.py` 中：
  - `parse_p2p_bandwidth_matrix`
  - `parse_p2p_bandwidth`
  - `collect_nvlink_bandwidth`
- 同步服务器时优先按分支切换后再推，避免把主线/分支包混装
- `.codex-tmp/` 和 `remote_reports/` 默认不要纳入正式提交

## 9. 建议的同步/验收动作

同步后至少检查：
- `pyproject.toml` 版本
- `src/gpuqa/__init__.py` 版本
- `python3 -m compileall src`
- `~/.local/bin/1cat-v100-qa --help`
- `~/.local/bin/1cat-v100-qa-gui --help`

真实验收建议：
- 至少跑一轮双卡
- 至少在一台 `4 GPU` 或 `8 GPU` 机器上验证 `NVLink` 矩阵汇总
- 单独确认 `优秀 / 良好 / 不通过` 显示是否符合预期

## 10. 当前已知未纳入仓库的目录

- `.codex-tmp/`
- `remote_reports/`

它们主要用于：
- 一次性同步脚本
- 远端报告回传

不建议打进正式分发包。

## 11. 0.2.6 版本变更（2026-06-09）

### 11.1 新增：HTTP API 服务器 (`src/gpuqa/httpd.py`)

- 基于 stdlib `http.server`，零额外依赖
- 端点：
  - `POST /api/run` — Agent 远程触发 GPU 检测（异步后台执行）
  - `GET /api/status` — 当前运行状态与进度
  - `GET /api/results` — 历史结果列表（保留最近 50 条）
  - `GET /api/results/{run_id}` — 指定测试的完整 JSON 结果
  - `GET /api/health` — 健康检查（含 GPU 数量、主机名、版本号）
- 支持 CORS，Agent 可从浏览器或远程调用
- 支持 `webhook_url` 参数，测试完成后自动回调

### 11.2 新增：CLI `serve` 命令

```bash
1cat-v100-qa serve --host 0.0.0.0 --port 8765
```

### 11.3 新增：CLI `--json` / `--quiet` / `--webhook-url`

- `--json`：`run` 命令输出结构化 JSON 到 stdout
- `--quiet`：静默模式，仅退出码表示结果（0=PASS, 1=FAIL, 2=NOT_RUN）
- `--webhook-url`：测试完成后 POST 完整结果到指定 URL
- `--version`：查看版本号

### 11.4 目标：Agent/OpenClaw 接入实现离人可用

0.2.6 的设计目标是让 Agent 能够：

1. **远程触发** — `POST /api/run` 或 `1cat-v100-qa run --json`
2. **查询状态** — `GET /api/status` 轮询进度
3. **获取结果** — `GET /api/results/{run_id}` 获取完整结构化数据
4. **被动通知** — webhook 回调，无需轮询
5. **常驻服务** — systemd 部署 API 服务器，随系统启动

结合 1cat-tunnel 反向代理，可以从公网安全地触发内网 GPU 机器的检测流程。
