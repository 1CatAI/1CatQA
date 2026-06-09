# 1Cat-V100-QA

最后一次更新代码时间：`2026-06-09`

当前分发包版本：`1CatQA-0.2.6`

`1Cat-V100-QA` 是一套面向 `Tesla V100` 的专项 GPU 质检工具，主要优化 `CUDA 12` 环境，当前重点验证对象为 `Tesla V100-SXM2-16GB`。

它不是单一压测脚本，而是一套完整的测试编排与报告工具，包含：
- GPU 发现与身份采集
- `gpu-burn` 压测
- `NVLink / P2P` 带宽测试
- `ECC / NVLink / NVLink CRC` 计数采集
- 中文文本/HTML/Markdown/CSV/JSON 报告输出
- 桌面终端文字界面入口
- **HTTP API 服务器** — Agent/OpenClaw 远程触发与结果查询
- **`--json` / `--quiet` CLI 模式** — 机器可读输出
- **Webhook 回调通知** — 测试完成自动推送结果

## 0.2.6 更新内容

- 新增 `serve` 命令：启动轻量 HTTP API 服务器（stdlib only，零依赖）
  - `POST /api/run` — Agent 远程触发 GPU 检测
  - `GET /api/status` — 查询当前运行状态
  - `GET /api/results` — 列出历史结果
  - `GET /api/results/{run_id}` — 获取指定结果详情
  - `GET /api/health` — 健康检查（含 GPU 数量）
- 新增 `--json` 标志：CLI `run` 命令输出机器可读 JSON
- 新增 `--quiet` 标志：静默模式，仅通过退出码表示结果
- 新增 `--webhook-url` 选项：测试完成后自动 POST 结果到指定 URL
- CLI 支持 `--version` 查看版本号
- 保留 0.2.5 全部功能（温度阈值、NVLink 前置、CRC 降级等）

## 0.2.5 更新内容

- 默认温度阈值从 `85C` 调整为 `80C`
- `gpu-burn` 压测过程中一旦检测到任意 GPU 超过 `80C`，会自动停止测试并立即输出失败报告
- `NVLink` 带宽测试前置到 `gpu-burn` 之前执行，避免高温阶段影响链路检测
- 主线保留 `NVLink CRC` 采集与展示，但 `NVLink` 错误、`NVLink CRC` 错误和 `NVLink` 通道异常都只降级为 `良好`，不再影响整机过测
- 修复 TUI 面板顶部边框颜色异常，顶部线条现在跟随对应面板颜色显示

## 0.2.4b 分支更新内容

- 删除 `NVLink CRC` 检测逻辑，不再采集、展示或以 `CRC` 增量作为失败判定条件
- `summary.txt / html / md / csv / json` 与终端界面同步移除 `NVLink CRC` 相关字段
- 保留原有 `ECC`、`NVLink` 错误计数、`NVLink` 通道数和带宽测试能力，作为 `0.2.4` 的分支包单独维护

## 0.2.4 更新内容

- 调整多 GPU 终端布局：`3 GPU` 现在默认使用“一排三张”压缩卡片布局
- 调整中高密度多卡布局：`5-7 GPU` 现在默认使用“三列多行”压缩卡片布局，避免在 `1080p` 显示器上把界面高度挤爆
- 保留 `4 GPU` 的 `2x2` 压缩卡片布局；只有更高卡数时才继续回退到单行列表布局

## 0.2.3 更新内容

- 新增多 GPU 自适应布局：`4 GPU` 时自动切换为压缩 `2x2` 卡片布局，尽量在原本 `2 GPU` 的主显示区域内显示完整 GPU 概览
- 新增高密度多卡布局：`5+ GPU` 时自动切换为“每张 GPU 单行”的实时状态列表，避免卡片把下半区结论和日志完全挤掉
- 保留原有 `1-2 GPU` 完整卡片布局，不影响双卡机器的显示体验

## 当前默认行为

当前版本默认流程如下：

1. 等待 `NVIDIA` 驱动就绪
2. 发现全部 GPU 并读取基础信息
3. 读取启动前的 `ECC` 单比特/双比特计数
4. 读取启动前的 `NVLink` 通道数、错误计数和 `CRC` 计数
5. 先执行 `NVLink` 带宽测试
6. 再执行 `gpu-burn`
7. 如果 `gpu-burn` 过程中温度超过阈值，会自动停测并直接进入失败报告输出
8. 读取测试完成后的 `ECC / NVLink / NVLink CRC` 计数并计算前后差值
9. 生成报告并写入独立结果目录

默认参数：
- `gpu-burn` 时长：`600` 秒
- `gpu-burn` 显存占用：`-m 100%`
- `NVLink` 带宽测试时长：`10` 秒
- 采样间隔：`5` 秒
- 温度阈值：`80C`

## 当前判定规则

当前版本的重要判定规则如下：

- 无 `NVLink / NVLink CRC` 相关异常时，评级为 `优秀`
- 发现 `NVLink` 错误增量、`NVLink CRC` 错误增量或 `NVLink` 通道数不足 `6/6` 时，不中断检测，评级降为 `良好`
- `ECC` 当前模式不是 `Enabled`，该卡判定为 `FAIL`
- `gpu-burn` 非零退出，判定为 `FAIL`
- `gpu-burn` 输出中如果 `errors:` 段出现非零数字，也判定为 `FAIL`
- `gpu-burn` 输出中如果 `WARNING!` 对应的错误数字非零，也判定为 `FAIL`
- `ECC` 单比特或双比特计数在测试后有增量，判定为 `FAIL`
- `gpu-burn` 压测过程中如果温度超过阈值，会立即停测并输出错误报告
- `ECC` 报错、温度超阈值、`gpu-burn` 异常和其它硬错误仍按原逻辑判定为 `不通过`
- 单卡压测模式下会跳过 `NVLink` 带宽测试，但仍会保留 `ECC / NVLink / NVLink CRC` 计数检查

## 当前桌面形态

当前桌面默认入口已经不是原来的 Tk 图形界面，而是纯文字终端界面。

桌面入口特点：
- 自动在图形会话里拉起终端窗口
- 单实例运行，重复点击不会正常并行启动多份
- 实时显示每张 GPU 的关键指标
- 显示 `gpu-burn` 实时输出
- 检测完成后直接在界面内显示总结，同时落盘到 `summary`
- 会根据 GPU 数量自动切换布局：`1-2 GPU` 完整卡片、`3 GPU` 三列压缩卡片、`4 GPU` `2x2` 压缩卡片、`5-7 GPU` 三列多行压缩卡片、`8+ GPU` 单行列表

当前文字界面会展示：
- 型号、序列号、UUID、PCI Bus ID
- 温度、当前功耗、GPU 峰值功耗、利用率、显存占用
- SM / MEM 频率
- `GPU-Burn GF/s`
- `GPU-Burn` 告警计数 / 错误计数
- `ECC` 单比特 / 双比特前值、后值和差值
- `NVLink` 通道数前值、后值
- `NVLink` 带宽
- `NVLink CRC` 增量
- 每张 GPU 的最终评级：`优秀 / 良好 / 不通过`
- 测试结论、日志、结果目录

说明：
- 仓库中仍保留 `src/gpuqa/gui.py` 的旧图形代码
- 当前默认发布入口是 `src/gpuqa/gui_entry.py` 启动的文字终端界面

## 默认测试工具优先级

`gpu-burn` 默认优先级：

1. `src/gpuqa/vendor/gpu-burn/ubuntu24.04-cuda12/usr/sbin/gpu-burn`
2. `/usr/sbin/gpu-burn`
3. `/opt/gpu-burn/bin/gpu_burn`

说明：
- 仓库已经内置 `Ubuntu 24.04 / CUDA 12` 的 `gpu-burn` 预编译内容
- 运行时会自动带上对应的 `compare.ptx`
- 默认尽量占满显存，避免低载压测

`NVLink` 测试默认工具：
- 优先使用内置/缓存的 `p2p_bandwidth_matrix`
- 找不到时再尝试系统中可用的兼容工具

## 输出结果

每次执行都会生成一个独立目录，例如：

```text
/home/onecatai/桌面/1cat-v100-qa-20260324-151359/
```

默认输出文件：
- `summary.txt`
- `summary.html`
- `summary.md`
- `summary.csv`
- `summary.json`
- `samples.csv`

各文件用途：
- `summary.txt`：纯文本中文摘要，便于终端或文本查看
- `summary.html`：现场查看用的人类可读报告
- `summary.md`：便于贴到工单、Wiki、Issue
- `summary.csv`：便于 Excel 或批量归档
- `summary.json`：保留结构化字段，适合程序对接
- `samples.csv`：记录压测过程中的逐次采样数据

`summary.csv / summary.json` 当前会包含以下关键信息：
- GPU 身份信息
- `ECC` 模式
- `NVLink` 通道数前后值
- 温度/功耗/频率/利用率摘要
- `ECC` 增量
- `NVLink` 错误增量
- `NVLink CRC` 错误增量
- `NVLink` 带宽
- `GPU-Burn GF/s`
- `GPU-Burn` 告警计数
- `GPU-Burn` 错误计数
- 结果和备注

## 运行依赖

### 最小运行依赖

- `Linux`
- `python3 >= 3.10`
- `python3-pip`
- `nvidia-smi`
- 正常工作的 `NVIDIA` 驱动
- 可用终端程序，例如 `x-terminal-emulator` / `gnome-terminal`
- `xdg-open`

### 推荐系统依赖

```bash
sudo apt-get update
sudo apt-get install -y \
  git \
  cmake \
  python3-pip \
  python3-venv \
  python3-dev \
  build-essential \
  g++-12 \
  pciutils \
  xdg-utils
```

### 可选依赖

- `nvcc`
  用于重新编译 `cuda/p2p_bandwidth_matrix.cu`
- `nvtop`
  仅当你希望通过 CLI 额外打开 `nvtop` 观察时使用
- `pyinstaller`
  仅当你需要在目标机上重新打包桌面可执行程序时使用

## Python 依赖

仓库根目录提供了：

```text
requirements.txt
```

当前运行时没有强制的第三方 Python 业务依赖，标准库已经覆盖当前逻辑。  
`requirements.txt` 主要保留 `setuptools`，方便批量部署时直接执行：

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install .
```

## 安装方式

### 方式一：源码直接运行

```bash
cd /path/to/1Cat-V100-QA
python3 -m pip install -r requirements.txt
PYTHONPATH=src python3 -m gpuqa run
```

### 方式二：安装为 Python 包

```bash
cd /path/to/1Cat-V100-QA
python3 -m pip install -r requirements.txt
python3 -m pip install .
1cat-v100-qa run
```

兼容命令 `gpuqa run` 仍然保留。

## 部署到其它机器

下面这套流程适用于当前分发包 `1CatQA-0.2.6`，目标机器建议为 `Linux + Tesla V100 + CUDA 12`。

### 第 1 步：复制分发包到目标机

把以下目录整体复制到目标机：

```text
1CatQA-0.2.6/
```

建议放到目标机用户目录，例如：

```bash
/home/<user>/1CatQA-0.2.6
```

可使用以下任一方式：
- `scp -r 1CatQA-0.2.6 <user>@<ip>:/home/<user>/`
- `WinSCP / FinalShell / MobaXterm` 直接拖拽
- U 盘拷贝后再复制到目标机

### 第 2 步：安装系统依赖

在目标机执行：

```bash
sudo apt-get update
sudo apt-get install -y \
  python3 \
  python3-pip \
  python3-venv \
  python3-dev \
  build-essential \
  g++-12 \
  cmake \
  pciutils \
  xdg-utils
```

如果目标机需要桌面启动入口，建议同时确保存在终端程序：

```bash
sudo apt-get install -y gnome-terminal
```

### 第 3 步：确认 NVIDIA 环境正常

先检查驱动、GPU 和 NVLink 状态：

```bash
nvidia-smi
nvidia-smi nvlink -s
nvidia-smi -q | grep -A3 "Ecc Mode"
```

如果这里已经报错，先修驱动和 CUDA 环境，再继续安装本工具。

### 第 4 步：进入项目目录

```bash
cd /home/<user>/1CatQA-0.2.6
```

### 第 5 步：安装 Python 依赖

```bash
python3 -m pip install -r requirements.txt
```

如果你希望把命令安装到当前 Python 环境，再执行：

```bash
python3 -m pip install .
```

安装完成后可用命令为：
- `1cat-v100-qa`
- `1cat-v100-qa-gui`
- `gpuqa`

### 第 6 步：建议编译 NVLink 带宽测试程序

如果目标机具备 `CUDA 12` 的 `nvcc`，建议执行：

```bash
mkdir -p bin
nvcc -ccbin g++-12 -O3 -std=c++14 cuda/p2p_bandwidth_matrix.cu -o bin/p2p_bandwidth_matrix
```

如果这一步失败，请依次检查：
- `nvcc --version`
- `g++-12 --version`
- CUDA 工具链是否已加入 `PATH`

### 第 7 步：直接运行一轮测试

源码方式运行：

```bash
cd /home/<user>/1CatQA-0.2.6
PYTHONPATH=src python3 -m gpuqa run
```

安装为包后运行：

```bash
1cat-v100-qa run
```

### 第 8 步：安装桌面快捷方式

如果目标机是图形桌面环境，可执行：

```bash
cd /home/<user>/1CatQA-0.2.6
bash scripts/install_desktop_source_app.sh "$(pwd)"
```

安装完成后，目标机真实桌面上会出现：
- `1Cat-V100-QA`
- `1Cat-V100-QA.desktop`

双击即可启动文字终端界面。

### 第 9 步：启动桌面文字界面

源码方式：

```bash
cd /home/<user>/1CatQA-0.2.6
PYTHONPATH=src python3 -m gpuqa.gui_entry
```

安装为包后：

```bash
1cat-v100-qa-gui
```

界面快捷键：
- `R` 开始检测
- `S` 停止检测
- `T` 修改温度阈值
- `B` 修改 `gpu-burn` 时长
- `N` 修改 `NVLink` 压测时长
- `G` 修改目标 GPU，输入 `all` 表示全部
- `O` 打开结果目录
- `Q` 退出

### 第 10 步：如需安装为 systemd 服务

建议先把目录复制到固定路径，例如：

```bash
sudo mkdir -p /opt/1CatQA-0.2.6
sudo cp -a /home/<user>/1CatQA-0.2.6/. /opt/1CatQA-0.2.6/
```

然后执行：

```bash
cd /opt/1CatQA-0.2.6
sudo bash scripts/install_systemd.sh /opt/1CatQA-0.2.6 /var/lib/gpuqa/latest
```

后续可用以下命令管理服务：

```bash
sudo systemctl daemon-reload
sudo systemctl enable gpuqa.service
sudo systemctl start gpuqa.service
sudo systemctl status gpuqa.service
```

### 第 11 步：查看输出结果

每次运行会生成独立目录，默认包含：
- `summary.txt`
- `summary.html`
- `summary.md`
- `summary.csv`
- `summary.json`
- `samples.csv`

如果需要固定输出目录，可手工指定：

```bash
PYTHONPATH=src python3 -m gpuqa run --output-dir ./reports/test-001
```

### 第 12 步：常见问题

- `ECC` 没打开会直接判定为 `FAIL`
- `gpu-burn` 输出中的 `WARNING / ERROR` 只要出现非零数字也会判定为 `FAIL`
- 单卡模式下会跳过 `NVLink` 带宽测试，这是当前设计行为
- 桌面模式依赖图形会话和终端程序，纯 SSH 环境建议直接使用 CLI
- 本项目是 `V100` 专项工具，不建议按通用 GPU 工具部署到其它型号

## Agent 集成 / HTTP API（0.2.6 新增）

### 启动 API 服务器

```bash
cd /path/to/1Cat-V100-QA
PYTHONPATH=src python3 -m gpuqa serve --host 0.0.0.0 --port 8765
```

或安装为包后：

```bash
1cat-v100-qa serve --host 0.0.0.0 --port 8765
```

### API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/health` | 健康检查，返回版本号和 GPU 数量 |
| `GET` | `/api/status` | 当前运行状态（是否在执行测试） |
| `POST` | `/api/run` | 触发一次 GPU 测试（异步） |
| `GET` | `/api/results` | 列出历史测试记录 |
| `GET` | `/api/results/{run_id}` | 获取某次测试的完整结果 |

### Agent 触发测试示例

```bash
# OpenClaw / 小龙虾 / 任意 HTTP Agent 都可以这样调用：
curl -X POST http://gpu-server:8765/api/run \
  -H "Content-Type: application/json" \
  -d '{"burn_seconds": 300, "max_temperature_c": 80}'

# 返回: {"run_id": "run-1717920000", "status": "accepted"}
```

### 轮询结果

```bash
# 查询当前状态
curl http://gpu-server:8765/api/status

# 测试完成后获取结果
curl http://gpu-server:8765/api/results/run-1717920000
```

### CLI 机器可读模式

```bash
# JSON 输出，Agent 可直接解析
1cat-v100-qa run --json

# 静默模式 — 不输出任何内容，只靠退出码：
#   0 = PASS
#   1 = FAIL/ERROR
#   2 = NOT_RUN
1cat-v100-qa run --quiet

# 测试完成后自动回调 Agent
1cat-v100-qa run --webhook-url http://agent-server:9000/callback
```

### Webhook 回调格式

测试完成后，服务端会向 `--webhook-url` 发送 POST 请求，body 为完整 JSON 结果：

```json
{
  "host": "gpu-node-01",
  "started_at": "2026-06-09T10:00:00+00:00",
  "finished_at": "2026-06-09T10:15:00+00:00",
  "overall_result": "PASS",
  "overall_assessment": "EXCELLENT",
  "gpu_count": 4,
  "results": [
    {
      "index": 0,
      "name": "Tesla V100-SXM2-16GB",
      "result": "PASS",
      "avg_temp_c": 68.5,
      "burn_gflops": 4200.0,
      "nvlink_bandwidth_gbps": 38.5
    }
  ],
  "output_files": {
    "summary_json": "/path/to/summary.json",
    "summary_txt": "/path/to/summary.txt"
  }
}
```

### systemd 常驻部署

让 API 服务随系统启动：

```bash
sudo mkdir -p /opt/1CatQA-0.2.6
sudo cp -a /home/<user>/1CatQA-0.2.6/. /opt/1CatQA-0.2.6/
```

创建 `/etc/systemd/system/1catqa-api.service`：

```ini
[Unit]
Description=1CatQA HTTP API Server
After=multi-user.target nvidia-persistenced.service

[Service]
Type=simple
User=root
WorkingDirectory=/opt/1CatQA-0.2.6
Environment=PYTHONPATH=/opt/1CatQA-0.2.6/src
ExecStart=/usr/bin/python3 -m gpuqa serve --host 0.0.0.0 --port 8765
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now 1catqa-api.service
```

## 常用命令

### 默认执行一轮测试

```bash
cd /path/to/1Cat-V100-QA
PYTHONPATH=src python3 -m gpuqa run
```

### 指定输出目录

```bash
PYTHONPATH=src python3 -m gpuqa run --output-dir ./reports/test-001
```

### 指定压测时长

```bash
PYTHONPATH=src python3 -m gpuqa run --burn-seconds 600
```

### 指定 NVLink 压测时长

```bash
PYTHONPATH=src python3 -m gpuqa run --nvlink-seconds 10
```

### 指定采样间隔

```bash
PYTHONPATH=src python3 -m gpuqa run --sample-interval 2
```

### 指定温度阈值

```bash
PYTHONPATH=src python3 -m gpuqa run --max-temperature-c 80
```

### 单独压测某一张 GPU

```bash
PYTHONPATH=src python3 -m gpuqa run --gpu-index 1
```

说明：
- 这里的 `gpu-index` 使用的是 `nvidia-smi` 里的真实 GPU 编号
- 单卡模式下 `gpu-burn` 只会压这一张卡
- 单卡模式下 `NVLink` 带宽测试会自动跳过

### 额外打开 `nvtop`

```bash
PYTHONPATH=src python3 -m gpuqa run --open-nvtop
```

### 指定外部测试工具

```bash
PYTHONPATH=src python3 -m gpuqa run \
  --gpu-burn-command "/usr/sbin/gpu-burn -c /usr/share/gpu-burn/compare.ptx -m 100%" \
  --nvlink-bandwidth-command /usr/local/bin/p2p_bandwidth_matrix
```

## 桌面文字界面

源码方式启动：

```bash
cd /path/to/1Cat-V100-QA
PYTHONPATH=src python3 -m gpuqa.gui_entry
```

安装为包后：

```bash
1cat-v100-qa-gui
```

文字界面快捷键：
- `R` 开始检测
- `S` 停止检测
- `T` 修改温度阈值
- `B` 修改 `gpu-burn` 时长
- `N` 修改 `NVLink` 压测时长
- `G` 修改目标 GPU（输入 `all` 表示全部）
- `O` 打开结果目录
- `Q` 退出

## 编译和桌面部署

### 编译 `p2p_bandwidth_matrix`

```bash
cd /path/to/1Cat-V100-QA
mkdir -p bin
nvcc -ccbin g++-12 -O3 -std=c++14 cuda/p2p_bandwidth_matrix.cu -o bin/p2p_bandwidth_matrix
```

### 手工运行 vendored `gpu-burn`

```bash
src/gpuqa/vendor/gpu-burn/ubuntu24.04-cuda12/usr/sbin/gpu-burn \
  -c src/gpuqa/vendor/gpu-burn/ubuntu24.04-cuda12/usr/share/gpu-burn/compare.ptx \
  -m 100% 600
```

### 安装桌面源码版与快捷方式

```bash
./scripts/install_desktop_source_app.sh /path/to/1Cat-V100-QA
```

### 在 Linux 目标机上重新打包桌面可执行程序

```bash
cd /path/to/1Cat-V100-QA
python3 -m pip install -r requirements.txt
python3 -m pip install ".[build]"
./scripts/build_linux_gui.sh
```

## systemd

项目提供了兼容现有部署的 `systemd` 模板：
- `systemd/gpuqa.service`

说明：
- 文件名暂时保留为 `gpuqa.service`
- 服务描述和内容已经按 `1Cat-V100-QA` 更新

打印模板：

```bash
PYTHONPATH=src python3 -m gpuqa print-service
```

安装脚本：

```bash
sudo bash scripts/install_systemd.sh /opt/gpuqa /var/lib/gpuqa/latest
```

如果希望输出目录固定落在桌面，可设置：

```bash
GPUQA_OUTPUT_ROOT=/home/onecatai/桌面
```

## 仓库结构

```text
1Cat-V100-QA/
  LICENSE
  README.md
  pyproject.toml
  requirements.txt
  cuda/
    p2p_bandwidth_matrix.cu
  scripts/
    build_linux_gui.sh
    gpu_burn_wrapper.sh
    install_desktop_entry.sh
    install_desktop_source_app.sh
    install_systemd.sh
    launch_gui.py
    refresh_vendored_gpu_burn.sh
  src/
    gpuqa/
      __init__.py
      __main__.py
      cli.py
      commands.py
      core.py
      desktop.py
      gui.py
      gui_entry.py
      models.py
      reporting.py
      service.py
      single_instance.py
      tui.py
      vendor/
  systemd/
    gpuqa.service
```

## 已知说明

- 当前项目是 `V100` 专项工具，不追求覆盖所有 GPU 型号
- 当前主要优化目标是 `CUDA 12`
- `gpu-burn` 的 `errors:` 非零计数已经纳入失败判定
- `NVLink` 通道数低于 `6/6` 目前记为告警和备注，不直接单独判失败
- 旧 Tk GUI 代码仍在仓库中，但当前默认发布入口为文字终端界面

## 分发建议

如果你要批量部署，建议分发以下内容：
- `src/`
- `scripts/`
- `cuda/`
- `systemd/`
- `README.md`
- `requirements.txt`
- `pyproject.toml`
- `LICENSE`

不建议把这些内容一起打包给目标机：
- `remote_reports/`
- `__pycache__/`
- 其他本地临时目录

## License

本项目使用 `MIT License`。
