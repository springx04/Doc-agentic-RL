# BayesTool-RL 项目更新记录

> 本文件记录 BayesTool-RL 在本地实现、服务器同步和实际训练验证中的问题、修复与证据。

## 2026-08-04：开始服务器验证

### 服务器环境

- 远端工作区：`/workspace/data/OpenClaw-RL`
- Python：`/workspace/data/envs/openclaw-rl-qwen3vl/bin/python`
- 模型：`/models/Qwen3-VL-4B-Instruct`
- 训练数据：远端既有 4 条 real DocVQA 训练样本
- 评估数据：远端既有 2 条 real DocVQA 评估样本
- 硬件：此前环境记录为 4 张 A100-SXM4-80GB

### 发现的问题

1. 远端 `toolcall-rl/` 不包含本地新增的 `bayestool/` 包、BayesTool 数据构造、Belief 训练入口和 4B/8B BayesTool launcher。
2. 远端 `generate_with_retool.py`（约 189 KB）小于本地最新版本（约 272 KB），说明远端仍是旧 rollout 实现，不能直接用于验证本地 BayesTool 代码。
3. 远端 BayesTool 专用 launcher 不存在，因此在未同步前启动会落回旧训练链路，不能作为 BayesTool 成功证据。

### 同步阶段新增问题

4. 首次通过远端桥接同步时发现单次本地文本读取返回约 40 KB 上限；大文件（如 `generate_with_retool.py`、`belief.py`）会被截断。该批次已判定为无效，未用于训练验证。

### 修复措施

- 改为按行分段读取本地文件、在调用端重组完整内容后再写入远端。
- 对每个同步文件计算并核对 SHA-256；只有完整哈希一致才进入远端编译。

### 远端第一次编译：失败与修复

5. 远端编译在 `toolcall-rl/document_reward.py:154` 失败：Unicode curly-apostrophe 字面量经过同步传输后损坏为未闭合字符串，触发 `SyntaxError`。虽然 BayesTool 25 个核心测试通过，但真实 rollout 会导入该文件，因此该测试结果不能作为训练通过证据。

### 修复

- 将该字符改为 ASCII 源码中的 `"\\u2019"` 转义，避免远端同步编码破坏 Python 字面量。
- 已重新同步该文件，等待全量远端编译复测。

### 处理计划

- 仅同步 BayesTool 运行和训练所需的源文件，不复制模型、数据、环境、旧输出或 checkpoint。
- 同步后先执行远端 Python 编译、BayesTool 单元测试和模块导入检查。
- 再执行真实 Qwen3-VL 4B 最小 RL smoke；每个失败都在本文件追加原因、修复和复测结果。

### 远端全量编译复测：通过

6. 修复 `document_reward.py` 后，远端全量 `py_compile` 已通过（`exit_code=0`）。编译过程中仅出现 `slime/slime/utils/arguments.py:193` 的既有 `SyntaxWarning: invalid escape sequence '\\.'`，不影响编译或运行，暂记为非阻塞警告。

7. 远端编译通过仅证明同步后的 Python 源码可编译，尚不代表 BayesTool RL 训练成功；仍需重跑核心测试、参数解析和真实 Qwen3-VL GPU smoke。

### 远端核心测试与参数解析

8. `tests/test_bayestool_core.py` 在远端修复后复测通过：`25 passed in 3.41s`。

9. 首次用最小 BayesTool 参数解析时失败：`hf_validate_args` 报告 `hidden_size`、`num_layers`、`ffn_hidden_size`、`norm_epsilon` 和 `rotary_base` 等 Megatron 参数为空或默认值，与 Qwen3-VL 的嵌套 `text_config` 不一致。原因是专用 launcher 虽设置了 HF checkpoint，但没有把 Qwen3-4B 文本骨干的完整 `MODEL_ARGS` 传给 Megatron。

### 修复与复测

- 在 Qwen3-VL 4B BayesTool launcher 中加载 Slime 的完整 `qwen3-4B.sh` 模型参数，并将 rotary base 设为 Qwen3-VL 配置的 `5000000`；同时保留原有桥接、视觉配置和 BayesTool 参数。
- 为服务器边界 smoke 增加环境变量覆盖（默认值仍为完整训练配置），只限制 rollout/评估/保存频率等测试规模，不改变 BayesTool 算法或训练链路。
- 修复后的远端 launcher SHA-256 与本地一致，`bash -n` 通过；用完整模型参数重新调用 `parse_args()` 已通过并正确得到 `advantage_estimator=bayes_grpo`、`bayestool_enable=True` 和 Qwen3-VL 4B 骨干配置。

### 第一次 GPU smoke：启动前失败

10. 第一次真实 GPU smoke 尚未进入 Ray job 或模型加载即失败：Ray 报告 `AF_UNIX path length cannot exceed 107 bytes`，原因是测试环境变量 `RAY_TMPDIR=/workspace/data/.ray/bayestool-qwen3-vl-4b-smoke-20260804-01` 过长，拼接 session/socket 后超过 Unix socket 限制。

### 处理

- 保留该失败作为启动环境问题记录，不把它计为训练失败或成功。
- 下一次使用短的、全新的 Ray 临时目录，避免复用半创建目录；训练参数和 BayesTool Stage C 配置不变。

### 第二次 GPU smoke：参数校验失败；同时发现上一处配置不合规

11. 第二次 GPU smoke 已越过 Ray socket 启动并提交 job，但在训练 worker 建立前由参数校验失败：
   `global_batch_size 4 is not equal to rollout_batch_size 1 * n_samples_per_prompt 4 // num_steps_per_rollout 2`。
   这不是模型或 BayesTool 算法错误，而是 smoke 覆盖值违反项目入口的批量约束；后续修复为 `1 * 4 // 2 = 2`，并将完整训练默认值从 `64` 对齐为 `8 * 8 // 2 = 32`。

12. 复核实现方案后确认，上一轮为通过参数校验而临时 source `slime/scripts/models/qwen3-4B.sh` 的做法不符合方案要求：它是文本模型配置，不能作为 Qwen3-VL 视觉塔、MRoPE、图像 token、vision start/end token 和多模态处理器均保留的实现证据。该版本不作为完成版本保留。

13. 服务器上最初未安装 `megatron.bridge`。尝试直接安装 `megatron-bridge==0.5.1` 后发现该版本要求 Transformers `>=5.8.1`，而服务器固定为 Transformers `4.57.1`，且导入还缺少 `modelopt`；该组合不兼容，不能通过随意升级核心环境规避。

### 本轮修复

- 新增独立的 `slime/scripts/models/qwen3-vl-4B.sh`，显式配置 Qwen3-VL 4B 文本骨干、`rotary_base=5000000`、`position_embedding_type=mrope`、`mrope_section=24 20 20` 和 interleaved MRoPE；不再复用文本版 Qwen3 launcher。
- 保持官方 HF Megatron-Bridge 负责视觉塔、视觉 token、视觉边界 token、视觉 embedding 注入和多模态处理器；后续以 provider 构造检查和实际 GPU smoke 验证这些字段，而不是仅以参数解析通过代替验证。
- 将服务器兼容的官方桥接依赖固定为 `megatron-bridge==0.3.1`，先替换不兼容的 `0.5.1`，再按真实导入错误补齐最小缺失依赖。
- 将 Ray 端口和 job address 改为可配置，下一次使用新的短临时目录和隔离端口，避免与之前失败后遗留的 Ray head 相互干扰；不强制停止该旧 head。

### 官方桥接依赖安装：第一次补依赖失败，第二次仍在服务器编译

14. 将 `megatron-bridge==0.3.1` 替换到服务器后，Qwen3-VL 配置和 `Qwen3VLProcessor` 可以读取，但导入 `megatron.bridge` 失败：`ModuleNotFoundError: No module named 'transformer_engine'`。这证明仅固定 bridge 版本仍未闭合运行依赖。

15. 第一次安装 `transformer-engine_torch==2.12.0` 失败：服务器环境没有通过 `/usr/local/cuda/include` 可见的 `cudnn.h`，并且该环境对应的 PyTorch/CUDA 组合没有可直接下载的预编译 TE wheel（构建器尝试的 GitHub wheel 返回 404）。该错误已保留，不能把纯 Python 包安装成功误判为桥接可用。

### 依赖修复进行中

- 已安装官方 `transformer-engine==2.12.0` 与 CUDA 12 核心二进制包 `transformer-engine-cu12==2.12.0`。
- 已发现服务器环境实际含有 cuDNN 9 头文件和库：`/workspace/data/envs/openclaw-rl-qwen3vl/lib/python3.12/site-packages/nvidia/cudnn/{include,lib}`；第二次 TE PyTorch 扩展构建已显式导出 `CUDNN_PATH/CPATH/LIBRARY_PATH`，当前正在编译。
- 本地 requirements 已固定 bridge、TE Python 包、CUDA 12 核心包和 TE PyTorch 扩展版本；待服务器编译/导入复测成功后再进入训练 smoke。

### 官方 bridge 与仓库内置 Megatron 源码版本不匹配

16. TE、ONNX 相关依赖补齐后，官方 bridge 继续导入到 Qwen 注册阶段，但被仓库 `Megatron-LM` 源码遮蔽的 core 缺少 `get_transformer_block_with_experimental_attention_variant_spec`，导致 `megatron.bridge` 在导入 Qwen3-Next 注册模块时失败。仅验证 Qwen3-VL 的 config/processor 不能掩盖这个运行时不一致。

### 修复

- 固定 `megatron-core==0.16.1`（在 `megatron-bridge==0.3.1` 声明的 `<0.17` 范围内），让官方 bridge 使用兼容的 core API；仓库源码仍保留并提供 `megatron.training`，不是删除训练框架。
- 将 `onnx==1.22.0`、`onnxscript==0.7.1`、`onnx-ir==0.2.1`、`ml-dtypes==0.5.4` 与 `nvdlfw-inspect==0.2.2` 写入依赖清单，闭合 TE 导入链路中实际暴露的依赖。

### 官方 Qwen3-VL provider 验收：通过

17. 服务器已完成官方 bridge 依赖闭合：`megatron-bridge==0.3.1`、`transformer-engine==2.12.0`、`transformer-engine_torch==2.12.0`、CUDA 12 核心包和 `megatron-core==0.16.1` 均可导入；TE 的实际 `transformer_engine.pytorch` 导入通过。

18. `AutoConfig` 读取到 `Qwen3VLConfig(model_type=qwen3_vl)`，`AutoProcessor` 读取到官方 `Qwen3VLProcessor`（含 `Qwen2VLImageProcessorFast`）。`AutoBridge.from_hf_pretrained('/models/Qwen3-VL-4B-Instruct')` 成功构造 `Qwen3VLModelProvider`，并验证：
   - `vision_config`: depth 24、hidden size 1024、patch 16、spatial merge 2、temporal patch 2、out hidden 2560；
   - 文本骨干：hidden 2560、36 层、FFN 9728、32 heads、8 KV heads、`rope_theta=5000000`；
   - `position_embedding_type=mrope`、`mrope_section=[24,20,20]`、`deepstack_visual_indexes=[8,16,24]`；
   - image/video token IDs `151655/151656`，vision start/end IDs `151652/151653`。
   这一步证明当前训练链路不是文本模型替代，也不是只在参数层声明视觉配置。

19. 为使 bridge 0.3.x 在仓库 Megatron 源码上注册，新增了兼容入口，将其委托给仓库已有的完整 `get_gpt_decoder_block_spec`；该兼容入口只解决官方 API 名称差异，不旁路或删减目标 Qwen3-VL provider。静态编译及 provider 复测通过。

### 服务器参数解析复测：通过

20. 直接调用 `parse_args()` 的首轮探针未设置 launcher 在 Ray runtime 中注入的 `CUDA_DEVICE_MAX_CONNECTIONS=1`，因此在 Megatron 的 TP/CP 参数校验处失败。该次失败属于独立探针环境不完整，不是 launcher 或训练实现失败；按实际 launcher runtime 环境补齐变量后复测通过。

21. 使用专用 Qwen3-VL 配置、官方 bridge 模式、BayesTool Stage C 和 smoke 批量约束 `global_batch_size=1*4//2=2` 重新解析，`parse_args()` 通过；关键值为 `hidden_size=2560`、`num_layers=36`、`ffn_hidden_size=9728`、`rotary_base=5000000`、`position_embedding_type=mrope`、`mrope_section=[24,20,20]`、`advantage_estimator=bayes_grpo`、`bayestool_enable=True`。仍需进入真实 GPU worker 以验收模型构建、视觉输入、rollout、BayesTool 分支和 optimizer update。

### 第三次 GPU smoke：Ray 端口配置失败

22. 第三次 GPU smoke 在启动 Ray head 时失败：自定义 `RAY_PORT=16379` 落入 Ray 默认 worker port 区间 `10002-19999`，触发 `Ray component worker_ports is trying to use a port number 16379 that is used by other components`。训练 worker 尚未创建，故不能将该轮记为训练失败。原 launcher 只允许自定义 head/dashboard 端口，没有同步配置 worker port 区间，这是实际可配置部署的健壮性缺口。

### 修复

- launcher 新增 `RAY_MIN_WORKER_PORT`/`RAY_MAX_WORKER_PORT`（默认 `20000-29999`），传给 `ray start`，并拒绝 head/dashboard 端口落入该区间或区间反向配置；下一轮使用全新的 Ray 临时目录和端口，避免复用本次失败环境。

### 第四次 GPU smoke：Ray job 入口路径失败

23. 新端口配置后的 smoke 已成功启动 Ray head 并提交 job，但 job 在入口处失败：Ray job 的工作目录为 `/workspace/data/OpenClaw-RL`，launcher 提交的相对入口 `python3 train_async.py` 被解析为 `/workspace/data/OpenClaw-RL/train_async.py`，返回 `No such file or directory`。模型构建、视觉输入、rollout 和 BayesTool 训练仍未开始；这是 launcher 的 job 入口路径缺口。

### 修复

- 将 job 入口改为仓库已有稳定 launcher 使用的绝对路径 `python3 "${SLIME_DIR}/train_async.py"`，并保留 `PYTHONPATH` 中的 Megatron、toolcall-rl 与 slime 路径；下一轮继续使用全新输出目录、Ray 临时目录和端口。

### 第五次 GPU smoke：Ray dashboard agent 端口冲突

24. 绝对 job 入口修复后，Ray head 启动成功但 job 提交仍失败。新 head 的 `dashboard_agent.log` 明确显示默认 dashboard agent HTTP 端口 `52365` 已被之前遗留 head 占用，6 次重试后 agent HTTP 服务被禁用，JobHead 最终返回 `No available agent to submit job`。这不是模型、数据或 BayesTool 训练错误，而是 launcher 尚未隔离 Ray dashboard agent 端口。

### 修复

- 新增 `RAY_DASHBOARD_AGENT_PORT`（默认按 dashboard 端口加 100 推导），传给 `ray start --dashboard-agent-listen-port`，并校验它不落入 worker port 范围且不与 head/dashboard 端口重复；下一轮使用全新的 dashboard agent 端口及其余运行目录/端口。

### 第六次 GPU smoke：官方 Qwen3-VL process-group 缺失

25. Ray agent、job 入口、4 GPU placement、Megatron 参数解析和官方 `Qwen3VLModelProvider` 均已通过，训练 worker 在构造真实模型时失败：`Qwen3VLModel.__init__` 访问 `pg_collection.cp`，但 provider 的 `_pg_collection` 为 `None`。原因是 Slime 复用了现有 Megatron backend，直接调用 bridge provider 的 `provide()`；而官方 bridge 的 `provide_distributed_model()` 才会先把 `ProcessGroupCollection.use_mpu_process_groups()` 注入 provider。该错误发生在模型构造层，尚未进入 rollout 或 optimizer update。

### 修复

- 在 Slime bridge provider 路径中，在 `provider.finalize()` 后使用仓库当前已初始化的 Megatron process groups 设置 `provider._pg_collection`，再调用官方 `provider.provide()`；不替换 Qwen3-VL provider、不旁路视觉模型，也不改动训练算法。下一轮使用全新 Ray/输出路径复测。

### 第七次 GPU smoke：训练构建时缺少 Apex wgrad 融合扩展

26. `pg_collection` 修复后，最新 GPU smoke 已真正进入官方 `Qwen3VLModel` 的模型构造；随后在 `ColumnParallelLinear` 处失败：`gradient_accumulation_fusion=True`，但服务器未安装 Apex 的 `fused_weight_gradient_mlp_cuda` 扩展。服务器已有 Transformer Engine，Megatron 的参数校验也明确允许在 TE>=2.7 时关闭该融合；因此这是 launcher 未显式适配当前运行时的配置缺口，不是 BayesTool、Qwen3-VL 视觉路径或训练算法的简化/替代。该次 job 在模型构造阶段退出，尚未进入 rollout 或 optimizer update。

### 处理原则与第一次错误安装

27. 为满足“缺失组件优先安装”的要求，先尝试从 NVIDIA Apex 源码安装 CUDA 扩展；第一次命令使用 pip 的 `--global-option=--cpp_ext/--cuda_ext`。日志明确显示 PEP 517 忽略了这些选项，最终只生成 `apex-0.1-py3-none-any.whl` 的纯 Python/无目标扩展安装，`fused_weight_gradient_mlp_cuda` 仍不可导入。该安装结果已判定为无效，不能作为修复。

28. 第二次改用 Apex `setup.py` 支持的环境变量 `APEX_CPP_EXT=1`、`APEX_CUDA_EXT=1`，并设置服务器 CUDA 12.8、A100 `TORCH_CUDA_ARCH_LIST=8.0` 与受控编译并发。前台构建受 bridge 读取超时影响留下半成品后退出，未把半成品当作成功；随后改为项目工作区内带日志的后台构建，完成所有 C++/CUDA 扩展，生成并安装 `fused_weight_gradient_mlp_cuda.cpython-312-x86_64-linux-gnu.so`。

### 最终修复

- 在服务器环境安装并验证 `apex` 源码构建结果：`import apex` 与 `import fused_weight_gradient_mlp_cuda` 均成功，目标扩展位于环境的 `site-packages` 中；同时补装 Transformer Engine 声明的 `importlib-metadata`（及其 `zipp` 依赖）。
- 将 Apex 源码从临时目录 `/workspace/data/apex-build` 迁入项目内 `/workspace/data/OpenClaw-RL/third_party/apex`，源码 commit 为 `9e3568a6f90fbc1996a06f8f9e99310bdaf2253a`；原临时目录已不存在，迁移后再次导入验证仍成功。
- 新增 `third_party/install_apex_cuda.sh`，脚本从自身路径解析 Apex 源码，使用 PEP 517 兼容的环境变量完成安装并强制导入验证；`third_party/README.md` 和 `slime/requirements.txt` 记录了安装方式，并明确禁止把无 CUDA 扩展的 PyPI `apex` wheel 当作替代品。
- 删除专用 Qwen3-VL launcher 中仅用于诊断的 `--no-gradient-accumulation-fusion` 临时参数；最终配置恢复 Apex 融合路径，不改变 BayesTool 算法、模型结构、视觉塔、数据、rollout 或训练批量语义。

### 第八次 GPU smoke：非交互环境未解析 Ray CLI

29. Apex 修复后的最终 GPU smoke 尚未进入 Ray：launcher 在 `ray start` 处报 `ray: command not found`。检查确认服务器训练环境 `/workspace/data/envs/openclaw-rl-qwen3vl` 中同时存在 Ray 2.54.0 Python 包和 `/workspace/data/envs/openclaw-rl-qwen3vl/bin/ray`，但服务器启动 bridge 的非交互 shell `PATH` 没有该环境的 `bin` 目录；这属于 launcher 运行时环境路径缺口，不是训练或依赖不存在。

### 修复

- launcher 新增 `PYTHON_BIN`/`RAY_BIN` 解析：默认使用同一 Python 环境，并将其 `bin` 目录加入 `PATH`；Ray CLI 不再依赖交互式 conda 激活，Ray job 入口也改为同一环境的 Python 绝对路径，避免训练 worker 混用 `/opt/conda/bin/python3`。
- 本轮 smoke 只在 Ray 启动前退出，未创建训练 worker、模型、rollout 或 optimizer update；修复后将使用全新的 Ray 临时目录、输出目录与端口重跑。

### 第九次 GPU smoke：Ray 临时目录过长

30. 使用项目内的 `.ray/smoke-20260807-07` 保存 Ray 临时文件时，Ray 启动再次报 `AF_UNIX path length cannot exceed 107 bytes`，具体是 `.../sockets/plasma_store` 超过 Unix socket 路径上限；该轮仍未创建训练 worker。项目源码、安装脚本和持久化训练输出不依赖这个临时目录。

### 处理

- 本轮将 `RAY_TMPDIR` 改为短的、全新的 `/tmp/obbt07`，并继续使用项目内的输出目录保存 launcher 日志、BayesTool 输出与 checkpoint；不把 Ray 临时 socket 当作项目文件迁移对象。
- Apex 和 Python/Ray 路径迁移仍保持不变，下一轮继续验证模型构造、rollout、BayesTool 分支和 optimizer update。

### 第十次 GPU smoke：SGLang common_ops 运行依赖缺失

31. 第十次 GPU smoke 已越过 Ray、4-GPU placement、官方 Qwen3-VL provider、真实模型构造和 Apex `fused_weight_gradient_mlp_cuda` 扩展加载，随后在 rollout engine 初始化时失败。SGLang `sgl-kernel==0.3.20` 加载其 A100 兼容构建时报告 `libnuma.so.1: cannot open shared object file`，因此没有进入 rollout、BayesTool 更新或 optimizer step；这不是通过关闭 SGLang 或替换 rollout 后端解决的问题。

### 修复

- 按真实缺失依赖安装系统包 `libnuma1`，并验证 `/lib/x86_64-linux-gnu/libnuma.so.1` 已被 `ldconfig` 注册。
- 使用服务器实际 Python/CUDA 环境直接验证 `from sgl_kernel import common_ops` 成功，加载路径为 `sgl_kernel/sm100/common_ops.abi3.so`；该构建包含 A100 `sm_80` 兼容代码，未采用降级、禁用或伪造模块的方式绕过错误。
- 将服务器 SGLang 源码从 `/workspace/data/.openclaw-build/sglang` 迁移到项目内 `/workspace/data/OpenClaw-RL/third_party/sglang`，原外部源码目录已不存在；重新安装 editable package 后，环境 finder 和 launcher runtime `PYTHONPATH` 均指向项目内路径。
- 迁移后保留源版本 `0.5.7.dev0+g24c91001` 的安装元数据设置，并在 `third_party/README.md` 记录恢复/重装命令。下一轮使用新 Ray 临时目录、端口和输出目录，重新验证 SGLang server、rollout、BayesTool 分支、optimizer update 与 checkpoint。

### workspace 路径迁移与正式数据缺口

32. 将桌面外部的《BayesTool-RL_最终实现方案.md》和《BayesTool-RL_方法.md》迁入项目 `docs/`，服务器端同步到 `/workspace/data/OpenClaw-RL/docs/`。将完整 Qwen3-VL-4B-Instruct 模型从共享 `/models` 复制到项目 `/workspace/data/OpenClaw-RL/models/Qwen3-VL-4B-Instruct`，源/目标均为 8.3G，排除 `.git` 后的文件清单与大小摘要校验一致；不删除共享挂载。

### 修复与已知输入缺口

- 4B BayesTool launcher 的默认模型、输出、Megatron 和数据路径改为项目内路径；Ray runtime 自动注入项目内 `third_party/sglang/python`，并对模型、train/eval 数据做存在性检查，避免静默回退到失效的 `/data_storage/wyj`。
- 服务器当前没有 `/data_storage/wyj/OpenClaw-RL`，项目内也只有 Qwen3-VL smoke train/eval 文件，没有可迁移的正式 document-qa 数据集。因此正式数据不是用 smoke 数据冒充；需要在正式训练前提供或生成 `data/document-qa/train.jsonl` 与 `data/document-qa/eval.jsonl`，此项暂记为真实输入缺口，技术 smoke 继续使用项目内 smoke 数据验证。

### 第十一次 GPU smoke：SGLang 已通过，packed Qwen3-VL MRoPE 暴露形状错误

33. 新 smoke 已使用项目内模型和 SGLang 源码完成 SGLang 权重加载（两张 A100）、服务 ready、Megatron HF 权重加载、权重同步和首个 rollout 请求。由于 smoke 的 `rollout_max_context_len=768` 小于带工具协议的实际提示，rollout 被正确标记为 `context_overflow` 并排除出 RL 统计；随后 Slime 注入 dummy sample，官方 Qwen3-VL `get_rope_index()` 将 packed THD 的 `cu_seqlens_q`（含样本/尾部 padding 多段）误当成 dense batch 行，触发 `IndexError: index 1 is out of bounds for dimension 0 with size 1`。本轮没有完成有效 optimizer update 或 checkpoint，不能计为成功训练。

### 修复进行中

- 在项目内新增 `slime/slime/backends/megatron_utils/qwen3_vl_compat.py`，在保留官方 Qwen3-VL provider、视觉塔和 MRoPE 算法的前提下，将 Slime 的 packed THD token 行按 `PackedSeqParams.cu_seqlens_q` 拆成逻辑序列，逐段调用官方 RoPE helper，再拼回 `[3,1,T]` 的 packed position IDs；同时按 vision-start/image/video token 计数切分对应 `image_grid_thw/video_grid_thw`，不是删除视觉路径。
- bridge provider 构造完成后安装该进程内幂等兼容适配；CPU 单元探针已覆盖“两段文本+尾部 padding”和“一段 64 image-token 视觉序列”，输出形状和视觉位置均通过。
- 下一轮将 smoke context 提高到 4096、response 提高到 512，并继续验证真实多轮工具调用、有效 BayesTool 样本、policy/ref log-prob、Bayes auxiliary loss、optimizer step 和 checkpoint。

### 第十二次 GPU smoke：rollout 已到达 reward，但工具预算变量未定义

34. 在 context 4096、response 512 的 smoke 中，SGLang 已完成服务启动、模型加载、权重同步，并实际收到 4 个 rollout 请求；说明模型推理和 rollout 请求路径已经越过前一轮的 MRoPE 错误。随后 `toolcall-rl/generate_with_retool.py` 的 `reward_func` 因引用未定义的 `max_tool_steps` 抛出 `NameError`，任务在 BayesTool utility 汇总前退出，尚未完成有效 update 或 checkpoint。该错误是实现缺口，不能作为可接受的训练缺口。

### 修复

- 在 `reward_func` 内按“运行参数 `args.bayestool_tool_budget` → 样本 metadata 的 `tool_budget` → `TOOL_CONFIGS["max_tool_calls"]`”解析并校验工具预算，统一传入 rollout 结果 metadata 和 `compute_bayestool_utility`；删除该作用域对未定义 `max_tool_steps` 的依赖。工具预算语义仍由配置控制，没有降低 BayesTool 计算、工具调用或训练逻辑。
- 恢复并校验服务器上的完整 `toolcall-rl/generate_with_retool.py`（5755 行、271826 bytes）；服务器 `py_compile` 通过。下一轮使用新的 Ray 临时目录/端口和输出目录继续追踪 BayesTool 样本、loss、optimizer step 与 checkpoint。

### 第十三次 GPU smoke：Transformer Engine 未选到 dot-product attention backend

35. 上一轮已完成 SGLang 推理、4 个 rollout 样本收集、BayesTool utility 计算、权重同步和 actor 的 `ref_log_probs/train` 入口，但在训练前向的 Transformer Engine attention 处失败：`ValueError: No dot product attention backend is available for the provided inputs`。当时 launcher 固定传入 `--attention-backend flash`，服务器环境没有 `flash-attn`，所以 FlashAttention backend 被禁用；该错误发生在有效样本进入前向之后，不能通过跳过训练来处理。

### 修复

- 使用 `NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2` 对实际 A100、BF16、GQA、2048 token 配置探针：TE 2.12.0 + cuDNN 9.16 明确报告 `FlashAttention=False`、`FusedAttention=True (sub-backend 1)`、`UnfusedDotProductAttention=True`。因此当前环境已有完整 cuDNN fused dot-product attention，不需要以缺失的可选 FlashAttention 替代它。
- launcher 新增 `ATTENTION_BACKEND` 环境覆盖，默认改为 `fused`，显式使用已验证的 Transformer Engine/cuDNN 完整 attention；仍可在安装并验证 FlashAttention 的环境中设置 `ATTENTION_BACKEND=flash`，没有移除或简化 attention、Qwen3-VL、MRoPE、BayesTool 或 optimizer 路径。
- 曾尝试按用户要求安装 `flash-attn==2.8.3.post1`；PyPI 仅提供源码包，默认会额外解析旧版 cuDNN，已改为 `--no-deps` 并只编译 A100 的 SM80。由于当前 TE fused backend 已实测可用，停止这项不必要的长编译，不把 FlashAttention 作为项目缺口。下一轮用 `ATTENTION_BACKEND=fused` 重跑真实 optimizer update。

### 第十四次 GPU smoke：packed THD 前向已通过，日志聚合暴露字符串类型错误

36. 本轮使用项目内模型/数据、context 4096、response 512 和 `ATTENTION_BACKEND=unfused` 完成 Ray placement、官方 Qwen3-VL provider 构造、两张 A100 模型加载、SGLang 服务、权重同步、4 个真实 rollout 请求、BayesTool utility、动态 batch/packed THD；Transformer Engine 实际选中 `UnfusedDotProductAttention`，`ref_log_probs` 与 `log_probs` 均已完成。随后在 `slime/slime/backends/megatron_utils/data.py:589` 的 `log_rollout_data` 中，对包含字符串 metadata 的列表直接执行 `sum(val)`，抛出 `TypeError: unsupported operand type(s) for +: 'int' and 'str'`，尚未进入 optimizer step 或 checkpoint。该错误是代码实现缺口，不能作为可接受的训练缺口。

### 修复

- 仅修正 rollout 标量日志的类型处理：数值列表仍按样本平均；字符串/混合非数值 metadata 不再执行非法求和并从标量日志中跳过；不改变 rollout、BayesTool reward、log-prob、loss 或 optimizer 的训练语义。
- 本地 `slime/slime/backends/megatron_utils/data.py` 已通过 `py_compile`，BayesTool 核心测试 `25 passed`；修复已同步到服务器，服务器文件 SHA-256 为 `5e0ef933f5ca275c91ed3a5ed545473327c2f3d87401b8cbf871f20f7361499d`。下一轮继续使用新的 Ray 临时目录/端口验证 optimizer update、参数更新和 checkpoint。

### 第十五次 GPU smoke：完整 BayesTool RL 训练链路成功退出

37. 第 14 轮使用项目内模型和 smoke 数据、BayesTool Stage C、4 个 rollout 样本、packed THD、`ATTENTION_BACKEND=unfused` 完成并成功退出：Ray job `raysubmit_DbBiBxArZQ9pkawR` 返回 `succeeded`；日志确认官方 Qwen3-VL worker、SGLang 服务、权重同步、4 个 rollout、BayesTool utility、`ref_log_probs`、`log_probs`、actor train step 0/1、optimizer 训练阶段和 update weights 均执行；随后成功保存 `torch_dist` checkpoint。checkpoint 目录含 `latest_checkpointed_iteration.txt`、`iter_0000000/.metadata`、四个 `.distcp` 分片、`common.pt`、`metadata.json` 和 rollout state 文件。

- 本轮没有异常 traceback 或非零退出。由于未经工具微调的初始 Qwen3-VL-Instruct 在 smoke prompt 上生成了 `model_protocol_error`/`<final>abstain</final>`，4 个样本 reward 均为 `-1.0`，因此该轮的 RL advantage/loss/grad_norm 为 0；这不是训练链路崩溃，但说明该 smoke 证明的是完整执行、BayesTool 数据流、前向、optimizer 调用和 checkpoint 持久化，不能把它误报成已有正向学习信号。正式训练仍需使用项目真实 document-qa 数据或工具调用 warm-start 数据。
- 当前 packed `thd_thd_thd` 输入在 TE 2.12.0 + cuDNN 9.16 上可稳定使用 exact `UnfusedDotProductAttention`；FlashAttention 仍未安装，不能把默认 `fused` 解释为 packed 输入已验证。下一步处理默认 backend 与 A100 FlashAttention 安装/验证，避免完整 launcher 在未设置覆盖变量时落入不可用 backend。

### 默认 attention backend 与 CUDA 安装脚本修正

38. 根据第 14 轮实际 packed 输入日志，launcher 默认 `ATTENTION_BACKEND` 已从 `fused` 改为 `unfused`；原因是 Qwen3-VL 的真实 `thd_thd_thd` + padding-causal layout 在当前 TE/cuDNN 组合中不支持 fused sub-backend，而 exact unfused backend 已通过 ref/log-prob、actor train 和 checkpoint 全链路。该修改不关闭 attention，也不改变 packed sequence、MRoPE、BayesTool 或 optimizer 语义；服务器 launcher SHA-256 为 `ff8f3dbbd624e1373b34aa7c9e0cc4572317c9f22d4da21031caadb63c6b9805`，`bash -n` 通过。
- 新增项目内 `third_party/install_flash_attn_cuda.sh`，默认只编译 A100 SM80，使用 `--no-deps --no-build-isolation`，安装后强制导入 `flash_attn_varlen_func` 验证；同步更新 `third_party/README.md` 和 `slime/requirements.txt`，不把可选 CUDA 扩展误写成会替换固定 PyTorch/cuDNN 的普通 pip 依赖。
- 服务器正在执行该脚本等价的 `flash-attn==2.8.3.post1` SM80 源码构建，日志保存在项目内 `third_party/flash_attn_install_20260807.log`；在构建成功并通过实际 packed smoke 前，不把 FlashAttention 标记为已完成。

### 项目内正式 document-qa 数据核验与预处理修复

39. 重新核验后确认正式数据已经位于项目自己的 `data/` 目录，不需要从外部 `/data_storage/wyj` 迁移或用 smoke 数据冒充：`data/train.jsonl` 为 1,000 条训练 QA，`data/test.jsonl` 为 200 条评估 QA，项目内 PDF 分别为 81 个训练文件和 17 个评估文件。此前“正式数据缺失”的记录被本次核验结果 supersede，不再作为当前输入缺口。
- 发现原始 DocVQA 记录使用 `pdf_path` 字段，而 `toolcall-rl/rl_data_preprocess.py` 与 `build_bayestool_data.py` 未识别该字段；本地首次生成命令因此无法处理原始数据。已补齐 `pdf_path` 解析，并新增回归测试覆盖该字段。
- 修复后本地和服务器均使用项目内 `data/` 作为 document root，成功生成默认 launcher 输入 `data/document-qa/train.jsonl`（1,000 行）和 `data/document-qa/eval.jsonl`（200 行），并启用 `--check-files` 完成 PDF 存在性检查；服务器生成文件 SHA-256 分别为 `697c69d860df632206bdff96dfea2d628c106bf47a1d7b83c604c933a0ac6cfe` 与 `ac62981ccf58bf3589cf3312eda55cbff1638d689b8633535c5c48f7251093a7`。
- 本地 `toolcall-rl/tests` 全量测试 `93 passed`；服务器预处理命令成功退出。该修复只补齐输入字段映射和项目内路径适配，不减少数据量、工具调用、BayesTool、rollout 或训练步骤。

### Tool Studio 参考文档与桥接脚本路径迁移

40. 对项目内仍引用外部目录的辅助路径完成迁移：Tool Studio 离线参考审计使用的 `ToolRL.pdf` 已放入项目 `data/reference/ToolRL.pdf`，本地与服务器字节数均为 2,475,493，SHA-256 均为 `287673c8f78d8fedbf472912c26fdfa8154c79a6f1263373ba365b83d77663a8`；审计脚本和 Tool Studio UI 已改为项目内相对路径，Tool Studio 服务会相对项目根解析该路径。
- `codex_k8s_bridge/enable_workspace_admin.ps1` 的默认桥接根目录已改为脚本自身所在的项目目录，仍允许显式传入其他目录；训练代码不再依赖外部 `C:\Users\...\paper` 路径。
- 本次迁移只移除项目运行时对外部路径的依赖，没有删除用户原始外部文件，也没有把参考 PDF 混入训练/评估样本。
### 桥接基础设施边界修正

41. 已按项目边界修正：`codex_k8s_bridge` 仅用于服务器桥接，不属于 OpenClaw-RL 的训练、数据、模型或第三方依赖。此前为迁移验证而复制到项目根目录的桥接文件已全部撤回；服务器继续使用外部桥接目录，项目代码不再依赖该目录。

### Qwen3-VL prompt 与终止守卫修复

42. 日志审计发现初始 Qwen3-VL 4B rollout 虽然生成了 `<final>`，但没有执行工具，且自定义 Jinja 模板把 `<|im_start|>system` 与 `<|im_start|>user` 的边界渲染成了非官方格式。已改为显式渲染并与官方 `Qwen3VLProcessor` 的 `apply_chat_template(..., tools=...)` 做逐字节比较；JSON 工具模板长度和内容均一致，本地及服务器 rollout 工具链测试均为 `28 passed`。
43. 进一步发现正答案路径的 `_can_finish()` 只拦截了 absence/abstain，没有拦截“尚未查看任何文档页就提交正答案”。已增加无文档观察时的终止守卫；只有已有结构化/视觉文档证据或搜索预算耗尽后才能结束。该修复保留既有证据候选、页访问和工具预算语义，并补充了正答案首读前回归测试。

### 4B RL 日志审计与 FlashAttention 依赖核验

44. 新增 `toolcall-rl/analyze_bayestool_training_log.py` 及其测试。审计同时检查 Ray job 状态、`ref_log_probs`、actor `log_probs`、训练指标、optimizer/update weights、checkpoint、reward 方差、`valid_for_rl`、工具调用、协议错误、Bayes sibling/branch/auxiliary 指标、loss 与 grad norm；只有执行链完整且存在非零策略更新信号、并实际执行过 BayesTool 工具动作时才标记 `effective`。同一脚本已在服务器同步并通过测试。
45. 对已有 4B 运行进行审计：14 号 unfused smoke 为 `chain_only`，虽然完整保存 checkpoint，但观察到的 reward 全为 `-1.0`、无工具动作、全为协议错误、Bayes advantage/loss/grad norm 均为零；16 号真实 data Flash formal 为 `failed`，在 `ref_log_probs` 前因 Transformer Engine 找不到 attention backend 退出。两次均未被误报为 RL 有效。
46. 已确认 `flash-attn==2.8.3.post1` 会被 Transformer Engine 2.12.0 的版本上限拒绝，项目安装脚本默认改为精确的 `flash-attn==2.8.3`，并继续用 `--no-deps --no-build-isolation` 和 A100 SM80 源码构建；在精确版本构建及 packed formal smoke 真正通过前，不把 FlashAttention 标记为已完成。

### “框架链路成功但 RL 信号为零”的代码根因与 4B 复验

47. 对 14 号 `chain_only` 结果继续追踪后，确认其中存在明确的实现缺陷，不是单纯的模型能力或设计问题：rollout 已将拒绝动作的负奖励写入 `action_rewards` 和动作 token span，但 Megatron actor 在 `loss.py` 中把 CPU `loss_masks` 临时复制到 GPU 后只修改了副本，没有写回 `rollout_data["loss_masks"]`。后续 `get_batch()` 仍读取原始全零 mask，因此拒绝动作没有进入实际 policy loss，造成“日志记录了动作惩罚、训练 loss/梯度仍为零”的假信号。

### 代码修复

- 新增 `slime/slime/utils/action_training.py` 的统一动作级训练覆盖逻辑，在保留拒绝动作公共 mask 语义的前提下，将负动作奖励显式应用到实际 `advantages`、`returns` 和 `loss_mask`；Megatron 的 `bayes_grpo` 与 `grpo/gspo` 路径均使用该逻辑，并把 GPU 临时 mask 的有效结果写回 rollout batch。
- 新增 `action_reward_consumed_count`、`action_reward_token_count` 和 `action_reward_abs_sum` 训练日志，区分“动作奖励被记录”与“动作奖励实际被训练器消费”。本地 `toolcall-rl/tests` 全部 `100 passed`，服务器针对该路径的测试 `6 passed`；生产 `compute_advantages_and_returns` 的最小 GPU 数据路径验证输出 `production_action_signal_ok`，确认 mask、advantage 和消费计数同步变化。

### 4B 真实复验结论

- 服务器 4B `actionmask-20260807-17` 运行成功，完成 rollout、`ref_log_probs`、actor `log_probs`、4 个训练 step、optimizer update 和 iteration 0/1 checkpoint；审计结果为 `partial_signal`，不是 `chain_only`。
- 真实训练日志已出现非零动作级训练信号：`action_reward_consumed_count=2.0`，动作 token 数分别为 `54.0`、`97.5`，动作惩罚绝对值分别为 `27.0`、`48.75`，`advantages=-0.5`；`train/loss` 约为 `0.5000–0.5123`，`train/grad_norm` 约为 `9.747–36.878`。这证明该代码缺陷已经修复，RL 的动作级负信号确实进入了训练更新。
- 同一运行的 4 条观测 rollout 仍全部为 `reward=-1.0`，均未执行 BayesTool，且均以 `model_protocol_error` 结束；因此没有 BayesTool sibling reward 对比，也不能宣称完整的 BayesTool RL 已经有效。该剩余现象属于当前基础模型/任务启动条件导致的 rollout 无工具、奖励无差异的设计或训练准备不足，不是本次已定位的 action-mask 消费代码错误；按当前要求暂不通过伪造工具调用、放宽终止守卫或改写 GRPO 分组来掩盖它。
