# BayesTool-RL 必要修复实施规划

> 状态：待实现（本文件只负责规划，不包含代码修改）
> 审计基线：2026-08-07 当前本地版本；服务器代码同步后，实施前需重新核对涉及函数和行号。
> 适用范围：当前项目配置、训练阶段与运行链路。

## 1. 目标与边界

本计划只处理会导致设计语义失真、关键训练信号缺失，或使 Stage C/Stage D 名不副实的问题。实施后应满足：

1. 同一 latent world 的 replica 只具有观测随机性差异，不具有故障结构差异。
2. 四类 coarse world（健康、局部、工具族、非平稳）能按定义生效，并向模型交付合法、可消费的观测。
3. 决策器在执行动作前能比较多个真实 policy candidate，使 regret branching、DVOI、Q replay 和辅助损失获得真实数据。
4. belief 的训练数据、监督目标、在线输入与推理接口一致，不使用 simulator 隐藏真值作为在线特征。
5. Bayes-GRPO 的 sibling 分组、失败惩罚和 rollout world 多样性符合定义。
6. Stage C 不能静默退化为 heuristic-only；Stage D 的 Meta episode 数据和分组必须正确。

以下实现可以保留，不应顺带重写：Bayes-GRPO 不做组内标准差归一化、工具观测 token mask、拒绝动作的负 advantage override、有限 top-k DVOI 假设、mean-field posterior particle、16 步 truncated BPTT、辅助 loss 的反向传播框架、Meta session persistence 和 suffix-return 公式本身。

## 2. 总体实施顺序

必须按依赖关系推进，不能同时无序修改：

1. **M1：world 身份与分组契约**——先修 latent seed、ID、replica 和 sibling 分组。
2. **M2：world 观测语义**——修健康隔离、context/非平稳触发、结构化腐化、失败状态。
3. **M3：多候选决策主链路**——把候选采样移到 controller 和执行动作之前。
4. **M4：belief 数据与模型语义**——统一 replay，修 context readout、teacher 监督和 next-tool prediction。
5. **M5：utility、训练门禁与 rollout 多样性**——修失败惩罚、checkpoint gate、rollout_id。
6. **M6：Q、switch/pre-invariance、stop/reopen**——只在 M3/M4 产出可信数据后启用。
7. **M7：Meta episode / Stage D**——修数据入口、ID、分组和事件切片。

每个里程碑应独立提交和验证。前一里程碑验收失败时，不启用依赖它的后续损失或训练阶段。

---

## 3. M1：修复 latent world、replica 与 sibling 分组

### 3.1 问题

设计要求同一 `world_slot` 的多个 replica 共享完整潜在世界，只改变观测随机性。当前 `sample_tool_world()` 的 seed 包含 `replica_id`，因此两个 replica 会独立采样目标工具、故障工具族、context rule、变化时刻和基础质量；但训练仍把它们放进同一 sibling baseline。

主要位置：

- `docs/BayesTool-RL_最终实现方案.md` 的 world seed 契约。
- `toolcall-rl/bayestool/world.py::sample_tool_world`
- `toolcall-rl/bayestool/world.py::WorldRuntime`
- `toolcall-rl/generate_with_retool.py` 中 `world_id`、`sibling_group_id` 和 BayesTool metadata 构造。
- `slime/slime/backends/megatron_utils/loss.py` 中 Bayes-GRPO 分组。
- `slime/slime/rollout/sglang_rollout.py` 中 auxiliary pair 的 same-world 判断。

### 3.2 目标身份模型

引入并明确区分以下 ID：

```text
coupling_id
  └─ rollout_id
      └─ world_slot
          └─ latent_world_id       # replica 共享
              └─ replica_id
                  └─ trajectory_id # 每条实际 rollout 唯一
```

推荐定义：

```python
latent_seed = stable_seed(coupling_id, rollout_id, world_slot, "latent")
latent_world_id = digest(coupling_id, rollout_id, world_slot, latent_seed)
observation_seed = stable_seed(latent_seed, replica_id, call_id, "observation")
```

`latent_seed` 必须独立决定：

- coarse/fine world type；
- session state；
- 所有工具基础质量；
- 单工具目标、工具族目标；
- context rules；
- regime schedule。

`observation_seed` 只允许决定：

- 某次调用是否采样到 timeout/empty/partial；
- 同一种 corruption 的具体位置、字符、噪声参数；
- 其他不会改变 latent world 定义的随机观测细节。

### 3.3 具体实现步骤

1. 为 `ToolWorldSpec` 增加 `latent_world_id`、`world_slot`；保留唯一 `world_id`，但其语义改为 `latent_world_id + replica_id`。
2. `sample_tool_world()` 用不含 `replica_id` 的 RNG 构造完整 spec。不要在该函数中抽取 replica 特有的腐化随机数。
3. `WorldRuntime.transform()` 按 `latent_seed + replica_id + call_id + tool_name` 构造单次观测 RNG。
4. 非 Meta sibling baseline 使用：

   ```text
   bayes-world:{coupling_id}:rollout={rollout_id}:latent={latent_world_id}
   ```

5. auxiliary pair 排除相同 `latent_world_id` 的 replica；不能再比较完整 `world_id`。
6. branch child 继承 parent 的 `latent_world_id`、replica 和当前 runtime state，不能重新采样 world。
7. 兼容旧 checkpoint/旧日志时：缺少 `latent_world_id` 的记录只允许用于离线诊断，不允许进入新的 sibling 或 auxiliary 分组。

### 3.4 验收条件

- 对至少 1000 个 `coupling_id`，同 slot 两个 replica 的 latent spec 序列化结果 100% 相等。
- 两个 replica 的 `world_id` 不同，但 `latent_world_id` 相同。
- 不同 slot 的 `latent_world_id` 不同。
- 相同 latent world 的观测允许随机不同；固定全部输入后可确定性复现。
- 任意 Bayes-GRPO group 只包含一个 `latent_world_id`，正常 group size 等于 `replicas_per_world`，branch group 按共享 prefix 另行分组。
- 启动时断言 `n_samples_per_prompt == worlds_per_prompt * replicas_per_world`；不满足时直接失败。

必须增加的测试：

- `test_replicas_share_exact_latent_world`
- `test_replica_observation_rng_is_distinct_and_reproducible`
- `test_sibling_group_contains_one_latent_world`
- `test_invalid_samples_per_prompt_fails_fast`

---

## 4. M2：修复四类 world 的实际语义

### 4.1 健康、单工具、局部和工具族故障必须隔离

#### 问题

当前所有非健康 world 会先把全部工具初始化为轻度退化状态；健康 world 的质量范围也足以触发腐化阈值。结果是“单工具故障”可能同时轻度污染其他工具，“健康 world”也可能进入腐化器。

#### 具体实现

1. 新增明确的 clean baseline：所有工具 availability/semantic/structure/calibration 为健康值，relative cost 为 1.0。
2. 所有 world 都从 clean baseline 开始，只应用该 world 声明的 patch：
   - `healthy`：不应用任何 patch，显式 `corruption_enabled=False`；
   - `single_tool_degradation`：只修改目标工具；
   - `context_degradation`：只在 rule 命中时修改目标工具的当前 context；
   - `shared_family_fault`：只修改目标 family；
   - `abrupt/gradual_change`：只在 schedule 生效后修改目标工具或 family。
3. session state 作为独立全局 latent factor 保留，但必须在 metadata 中与工具故障分开记录；健康 world 若强制 healthy session，则不得再引入背景退化。
4. 如果未来需要背景噪声，应新增具名 world/ablation，不能隐式混入现有类别。

#### 验收条件

- 健康 world 的 `corruption_applied` 必须恒为 false。
- single-tool world 在目标外的工具 spec 与 clean baseline 完全相同。
- shared-family world 在 family 外的工具 spec 完全相同。
- 同一干净工具输出在 healthy world 中逐字、逐图路径保持不变。

### 4.2 context world 必须基于公开任务上下文采样且可命中

#### 问题

当前 context world 固定在 1～5 页随机取页，并允许选择不使用 `page_number` 的工具；matcher 又不识别 `page_numbers`，因此部分规则结构上永远不会触发。

#### 具体实现

1. 增加只包含公开信息的 `WorldSamplingContext`，至少提供：

   ```python
   page_count: int | None
   tool_argument_capabilities: dict[str, set[str]]
   tool_budget: int
   ```

   禁止传入答案页、答案 bbox、标签或其他 ground truth。
2. context sampler 只从支持 page/region 的工具中选目标，并只在合法页码内采样。
3. `_extract_page_region()` 同时处理：
   - `page_number`
   - `page`
   - `page_numbers`（列表中任一页命中即生效）
   - `bbox` / `region`
4. 如果 `page_count` 暂时未知：
   - 可以把 rule 标记为 `pending_context_binding`；
   - 第一次获得公开 `page_count` 后再确定性绑定；
   - 不能用固定 1～5 盲采。
5. 每个 rollout 记录 `context_rule_match_count`、`first_context_match_call` 和 `affected_call_count`。

#### 验收条件

- 有 page_count 时，采样页永不越界。
- `parse_document(page_numbers=[...])` 能正确触发 page rule。
- 构造必命中的调用时，rule 激活率 100%。
- 训练统计中单独报告“未命中”而不是把它当成健康观测；长期未命中率超过配置阈值时报警。

### 4.3 非平稳 world 必须在 rollout 预算内产生有效变化

#### 问题

当前 abrupt/gradual 的起始时刻可能太晚，gradual 的第一步又为零，使短 rollout 完全看不到变化。

#### 具体实现

1. schedule sampler 接收 `tool_budget`，保证：
   - 至少一次调用处于变化后状态；
   - 变化后至少还剩一次 agent 决策机会。
2. 推荐 abrupt 的有效起点范围为：

   ```python
   1 <= start_call <= max(1, tool_budget - 2)
   ```

3. gradual progress 使用：

   ```python
   progress = (call_index - start_call + 1) / (end_call - start_call + 1)
   ```

   并裁剪到 `[0, 1]`，确保第一有效步非零。
4. 记录 `first_effective_call`、`last_effective_call`、`affected_call_count`。
5. 若目标工具/family 从未被调用，应记录 `schedule_not_exercised`；采样器或 curriculum 可对这类样本重采/降权，但不能伪装成已发生变化。

#### 验收条件

- 在配置的最小工具预算下，每个 schedule 都至少有一次非零作用。
- abrupt 在起点前后产生预期的离散变化。
- gradual progress 单调不减且第一有效步大于零。

### 4.4 用 schema-aware adapter 取代通用文本腐化

#### 问题

当前通用删行、删字符、附加文本会造成无效 JSON、大量 no-op，并使视觉工具的 `image_path` 无法解析。world 注入失败的普通文本也可能被主循环误判为成功。

#### 具体实现

建立统一接口：

```python
class ObservationCorruptionAdapter:
    def corrupt(payload, quality, rng, runtime_context) -> CorruptionResult: ...

@dataclass
class CorruptionResult:
    observed_result: str
    observation_status: str
    corruption_type: str | None
    corruption_applied: bool
    artifact_paths: list[str]
```

按工具类别实现：

1. **parse_document / OCR**：解析 JSON，只修改 `pages[].markdown/text`、returned page 集合或置信度字段；保持顶层 schema 合法。
2. **detect_layout**：只修改 region 列表、bbox、type、confidence；bbox 必须保持合法顺序和有限数值。
3. **extract_table / chart_to_table**：在 cell/row/column/series 结构上做删除、交换或数值偏移；保持 JSON 合法和必要字段存在。
4. **render/crop/zoom**：真正生成退化图像文件；复制原 payload，只把 `image_path` 指向新文件。不得在 JSON 末尾追加说明文字。
5. 所有 adapter 必须返回 `corruption_applied`；如果随机操作没有改变 payload，应有限次数重试，仍失败则明确标记 no-op。
6. world 注入失败统一返回结构化状态，例如：

   ```json
   {
     "status": "error",
     "failure_origin": "world_injected",
     "execution_succeeded": true,
     "observation_delivered": true,
     "message": "..."
   }
   ```

7. 主循环分别处理：
   - `execution_succeeded=false`：基础设施/真实工具执行失败，可按现有 infra 规则中止或排除；
   - `execution_succeeded=true, observation_status=error`：合法 world 观测，加入 belief 并让 agent 继续；
   - agent 后续使用失败证据作答：由 utility 处罚。

#### 验收条件

- 所有结构化工具 corruption 后 JSON 解析成功率 100%。
- 所有视觉 corruption 的新图像存在、可解码，且会被加入模型视觉输入。
- 非 no-op corruption 的 payload/artifact 与 clean 结果确实不同。
- severity 增大时，定义好的质量指标总体单调恶化；至少不能系统性反向改善。
- world-injected failure 不被计为 infra failure，且 rollout 可以进入下一轮。

### 4.5 移除在线 belief 的隐藏真值泄漏

#### 问题

`semantic_agreement` 目前直接由 simulator 隐藏质量计算，却被当作在线观测特征使用。

#### 具体实现

1. 隐藏 quality 只进入 `ToolStateLabel` / `hidden_supervision`，不得进入 public `WorldEvent`。
2. public agreement 只由可观测结果计算，例如：
   - OCR 与 parse 的归一化文本/数字集合一致性；
   - table 与 parse 的 cell/numeric 一致性；
   - layout region 与可观测表格/图像区域一致性。
3. 没有独立比较来源时，设置相应 missing mask；不要使用 0.5 或隐藏真值假装已观测。
4. `question_term_overlap` 使用公开 question token 计算，不能继续固定为 0。

#### 验收条件

- 删除 hidden label 后，在线 feature extraction 结果不变。
- public event 序列化结果中不存在 latent quality、answer location 或其直接函数。
- agreement 缺失时，模型能通过 missing mask 区分“未知”和“中等一致”。

---

## 5. M3：在执行前建立真实多候选决策闭环

### 5.1 问题

正常主链路先生成并解析一个动作，再把这一个动作交给 `DecisionController`。单候选导致 consensus 恒为 1、regret 恒为 0；而额外候选只在 branch eligible 后才采样，形成循环依赖。最终 regret branching、DVOI、Q replay、switch/pre-invariance 基本没有可靠数据。

主要位置：

- `toolcall-rl/generate_with_retool.py::_sample_bayestool_branch_candidates`
- `toolcall-rl/generate_with_retool.py::_launch_bayestool_branches`
- `toolcall-rl/generate_with_retool.py` 的 assistant turn 解析/执行主链路。
- `toolcall-rl/bayestool/decision.py::DecisionController`

### 5.2 目标执行顺序

```text
固定当前 public prefix / runtime checkpoint
    ↓
从完全相同 prefix 采样 K 个真实 policy completions
    ↓
解析、去重、保留原 token_ids/logprobs
    ↓
DecisionController 对候选集合计算 value/consensus/regret/DVOI/stop
    ↓
执行被选中的真实候选
    ↓
若 regret 达阈值，从同一 checkpoint 启动 sibling continuation
```

### 5.3 具体实现步骤

1. 复用 `_sample_bayestool_branch_candidates()`，但把调用移到执行当前动作之前。
2. 第一个 completion 仍可作为 primary；从相同 prefix 至少再采一个 completion。若前两个不同，再按配置扩到最多 `max_action_candidates`。
3. 候选支持 `tool/final/abstain`，并保存：

   ```text
   canonical_action_key
   parsed_action
   raw_text
   token_ids
   token_logprobs
   source/seed
   parse_status
   ```

4. 只去除 canonical action 完全相同的重复项；不能用重新渲染的 action text 替代原始 completion。
5. `DecisionController.select()` 必须在至少两个不同候选上运行；少于两个时：
   - 标记 `candidate_degenerate=true`；
   - 执行 primary；
   - 禁用本前缀的 regret branch、Q replay、switch 和 pre-invariance；
   - 不得把单候选 consensus 当成有效高置信证据。
6. controller 选中的动作必须直接使用该候选保存的 token/logprob；不能另行合成未由 policy 采样的动作。
7. branch children 从采样候选中选择，继承精确 prefix checkpoint、latent world、replica 和 belief state，只替换第一动作后继续 horizon-H。
8. `best_action_margin` 只有至少两个 action value 时才定义；否则保存为 null，而不是当前动作自身价值。
9. 记录关键指标：
   - `candidate_count_raw/distinct/valid`
   - `candidate_degenerate_rate`
   - `regret_branch_eligible/triggered`
   - `branch_child_count`
   - `q_replay_record_count`

### 5.4 验收条件

- 构造两个价值不同的候选时，controller 的 regret 大于零且能触发 branch。
- sibling 的第一动作 token 与对应候选保存的 token 完全一致。
- branch parent/child 的 prefix hash、latent world、replica、world runtime call index 在分叉点一致。
- 单候选时 pre-invariance loss 不被构造，而不是产生一个恒零 bundle。
- 正常 smoke rollout 中能够观察到非零 branch/Q replay 计数；若策略已塌缩导致没有不同候选，日志必须明确暴露。

---

## 6. M4：统一 belief 训练、上下文状态和在线推理

### 6.1 建立唯一 canonical belief replay

#### 问题

当前 rollout 信息分散在 `tool_execution_trace`、`metadata.tool_execution.calls`、`metadata.bayestool.world_events` 等位置，而 `BeliefEventDataset` 主要读取另一套字段，可能把多次调用读成默认单事件；事件级 task state 也不完整。

#### 具体实现

新增 replay exporter，输出每行一个 trajectory：

```json
{
  "schema_version": 1,
  "trajectory_id": "...",
  "coupling_id": "...",
  "latent_world_id": "...",
  "replica_id": 0,
  "document_hash": "...",
  "events": [
    {
      "call_id": 0,
      "tool": "parse_document",
      "arguments": {},
      "task_state_before": {},
      "observed_result": "...",
      "world_event": {},
      "task_state_after": {},
      "hidden_label": {},
      "next_tool_id": 1
    }
  ]
}
```

实现要求：

1. exporter 以实际 execution trace 为主轴，按 `call_id` 对齐 public event 和 hidden label。
2. 每次调用保存 before/after task state；训练预测下一观测时使用 before-next-action 对应状态。
3. `BeliefEventDataset` 优先且严格读取 canonical schema；旧形状仅放在独立兼容转换器中。
4. schema validator 检查事件数、call_id 单调性、工具名、label 对齐和 next-tool 对齐。
5. chunking 可以继续每 16 步重置 hidden，作为 truncated-BPTT 近似；不能跨 trajectory 混合。

验收：给定 N 次真实工具调用，exporter 和 dataset 必须产生 N 个有序 event；不得静默回退成默认 `tool_id=0`。

### 6.2 让 context hidden 真正影响预测与决策

#### 问题

context GRU 会更新并保存 hidden，但 quality、observation、DVOI 和 Q 使用的 readout 不消费它；local reopen 因而没有决策效果。

#### 具体实现

1. 由候选动作计算 context key：`tool + page/page_numbers + region`。
2. observation readout 改为消费：

   ```text
   session_hidden + context_hidden + tool_embedding + task_projection
   ```

3. quality 采用 global tool posterior 加 context residual：

   ```text
   q(tool, context) = combine(global_tool_quality, context_quality_residual)
   ```

   residual 只作用于当前 tool/context，不能修改全部工具。
4. `predict_observation(candidate_action)`、DVOI hypothetical update 和 Q feature 都使用候选对应 context posterior。
5. local reopen 混合/清除当前 context residual；family/global reopen 按现有层级处理。

验收：保持 session/tool/task 不变，仅改变某一 context hidden 时，该 context 的 observation/quality/action value 必须变化，其他无关 context 不变。

### 6.3 修复 smoother 监督和 HBD 启用条件

#### 问题

smoother 的 session/regime/shared 有直接监督，但 quality/cost head 没有；HBD 却蒸馏这些未校准输出。

#### 具体实现

1. 从 canonical replay 的 `hidden_label` 读取每工具 quality 和 relative cost。
2. smoother quality：使用 Beta NLL/KL，或均值与 concentration 的组合损失；必须覆盖 availability、semantic、structure、calibration 四维。
3. smoother cost：对 `log(relative_cost)` / `log(latency_ratio)` 做回归，并提供 mask。
4. 所有 hidden-target loss 只用于 synthetic world 数据；真实日志缺少 label 时 mask 为零。
5. 只有 teacher 在 validation set 达到配置的 calibration gate 后，才启用对应 HBD 项：

   ```text
   hbd_session_enabled
   hbd_quality_enabled
   hbd_cost_enabled
   hbd_change_enabled
   ```

6. checkpoint manifest 保存每个 head 的 calibration 指标和 enable flag。

验收：单个 synthetic batch 反传后，smoother quality/cost head 都有非零有限梯度；未达 gate 时 filter 不接收对应 teacher loss。

### 6.4 对齐 next-observation 的训练与推理

#### 问题

训练用当前事件更新后的输出预测下一事件，却仍使用当前 tool ID；线上推理使用的是候选下一工具。

#### 具体实现

1. filter 用 event `t` 更新得到 `h_t`。
2. 预测 target `event_{t+1}` 时显式调用：

   ```python
   observation_from_hidden(
       h_t,
       context_hidden=context_for_next_action,
       tool_id=next_tool_id,
       task_projection=task_state_before_next_action,
   )
   ```

3. 最后一个 event 没有 next target，必须 mask。
4. 可选：用初始 prior `h_0` 预测第一事件，但需单独定义，不能与 post-event 输出混用。

验收：交换两个 event 的 `next_tool_id` 会改变对应 observation logits 和 loss；训练和 runtime 使用同一 readout 函数。

---

## 7. M5：修复 utility、rollout world 多样性和 Stage 门禁

### 7.1 统一 failure event 与惩罚来源

#### 问题

utility 从 metadata 顶层读取 failure count，而 runtime 主要写入 `metadata.action_statistics`，导致协议错误、非法参数等可能没有进入失败惩罚。

#### 具体实现

1. 建立 canonical `failure_events[]`：

   ```json
   {
     "type": "invalid_argument",
     "origin": "agent",
     "turn": 2,
     "penalize": true,
     "used_for_final_answer": false
   }
   ```

2. runtime 在事件发生处写入，不在 utility 阶段根据零散计数猜测。
3. 至少覆盖：protocol error、invalid argument、unsupported/premature final、预算耗尽未终止、使用失败证据。
4. `origin=world_injected` 本身不罚 agent；若 agent 后续把失败观测当可靠证据作答，则记录独立 `used_failed_evidence`。
5. infra failure 标记样本不可训练/排除，不转成普通负奖励。
6. `compute_bayestool_utility()` 只消费 canonical events；迁移期可兼容旧 `action_statistics`，但必须记录 compatibility warning。

验收：用真实 rollout metadata 构造每类失败，utility 的 F 与预期一致；world failure 本身 F=0，错误使用后 F>0。

### 7.2 每轮训练应产生新的、可复现的 world

#### 问题

当前训练数据通常没有 `rollout_id`，runtime 回退为 0，导致同一道题每个 epoch 重复相同 world。

#### 具体实现

1. rollout orchestrator 在 prompt 扩展成 8 个 sibling 前分配一个 `rollout_id`。
2. `rollout_id` 必须：
   - 同一组 8 个样本共享；
   - 不同训练 rollout 递增或稳定变化；
   - checkpoint resume 后不回退/冲突；
   - 不依赖 worker/rank 调度顺序。
3. 推荐由全局 rollout step + prompt stable ID 派生，并写入 sample metadata。
4. eval 使用固定 `rollout_id` 或 `fixed_world_specs`，保证可比较；训练不要固定 world spec。

验收：同一训练 step 的 sibling world 可耦合复现；相邻 rollout step 的 latent worlds 有变化；resume 后序列连续。

### 7.3 Stage B/C/D 必须 fail closed

#### 问题

当前 launcher 默认 Stage C，但 belief/Q/risk checkpoint 均可缺失，Stage C 可能静默运行 heuristic belief、无 Q、heuristic risk。

#### 具体实现

建立 capability gate：

| Stage | 必需能力 |
|---|---|
| A | canonical replay + belief/smoother trainer |
| B | 已校准 belief checkpoint；Q/risk 可处于明确 bootstrap 状态 |
| C | belief + risk checkpoint 必需；Q 在 warmup 后必需；多候选/branch 已通过 smoke gate |
| D | C 的全部能力 + 合法 Meta dataset/ID/grouping |

实现要求：

1. launcher 在启动 worker 前加载 checkpoint manifest 并验证 schema/version/calibration gate。
2. 缺少必需能力时直接退出。
3. heuristic ablation 只能通过显式 `--allow-heuristic-*` 运行，并在实验名、metadata 和汇总日志中标明。
4. Stage A 应实际调用 belief 训练入口，而不只是把 policy RL 的 branch probability 设为 0。
5. 若采用手工分阶段训练，也应有统一 manifest，而不是仅依赖环境变量约定。

验收：Stage C 缺任一必需 checkpoint 时启动失败；显式 ablation 可启动但样本全部带 ablation 标记。

---

## 8. M6：在可信多候选和 belief 上修复辅助机制

### 8.1 Q feature 不得通过拼接截断丢失工具状态

#### 问题

当前 particle feature 先拼接所有工具质量再截断到 32 维，后部工具完全不影响 Q；action feature 对 `page_numbers` 等参数表达也不足。

#### 具体实现

采用 factorized encoder，而不是改变工具顺序后硬截断：

```text
global features:
  session/regime/change/shared-family posterior

selected-tool features:
  availability/semantic/structure/calibration 的 mean/std
  relative cost
  candidate context residual

pooled features:
  全工具和各 family 的 min/mean/max/entropy

action features:
  action kind/tool/page/page_numbers/region/remaining budget/last tool
```

通过 learned projection 压到固定宽度。checkpoint 保存 `q_feature_schema_version`；版本不一致拒绝加载。

验收：逐个修改任一工具的 posterior，在该工具作为候选时 Q feature/Q output 都会变化；所有工具通过参数化测试。

### 8.2 switch/pre-invariance 必须使用支持有效的 pair

#### 问题

当前 belief distance 把异质统计量拼成一个 categorical JS；same-world 判断使用包含 replica 的 `world_id`；单候选时 pre-invariance 恒为零；best action 也未必来自真实比较。

#### 具体实现

1. switch pair 必须满足：
   - 相同 public task/content；
   - 不同 `latent_world_id`；
   - 至少两个共同真实候选；
   - 各 world 的 best action 来自 branch return 或已校准 Q；
   - best/runner-up margin 达阈值。
2. belief distance 改为各组成部分分别计算后加权：
   - categorical：session/regime/shared 分别算 JS；
   - Beta：mean 与 log-std 的归一化距离；
   - cost/change：标量距离；
   - missing 项不参与分母。
3. pre-invariance 必须在 `prefix_step < first_distinguishing_step` 的完全相同 public prefix 上，使用一次生成、两边共享的候选集合；候选数至少为 2。
4. pair 构造先按 ID/prefix 建索引，再限额采样，避免对全部状态做无界 O(n²) 枚举。

验收：单候选不生成 bundle；相同 latent world 不生成 switch；共同 prefix 破坏后不生成 pre-invariance；合法 pair 的 loss 对两边 action logprob 都有非零梯度。

### 8.3 stop controller 必须在 terminal action 执行前运行

#### 问题

final/abstain 当前在 controller 前被接受或拒绝，因此 risk controller 无法阻止高风险 final，也无法在 final 和工具候选之间作选择。

#### 具体实现

1. 将 `final`、`abstain`、`tool` 都保留在 M3 的候选集合。
2. 在任何 terminal return 前运行 controller。
3. final risk 使用校准器；continue risk 至少应基于候选工具的预计 cost + 预测观测后的风险，而不是固定常数加 regret。
4. 不安全 final：
   - 若存在可接受工具候选，执行 controller 选择的工具；
   - 若只有 final，返回 recovery observation 并继续下一轮；
   - 保留现有 evidence guard 的负 action span 训练语义。

验收：构造高风险 final + 低风险工具候选时，不得终止；低风险且证据充分的 final 可以正常终止。

### 8.4 reopen 采用层级证据和滞回

#### 具体实现

1. local：单次局部高 surprise，只重置当前 context residual。
2. family：同一 family 连续至少两次高 surprise，且每次都达到 family 阈值。
3. global：`P(change)` 达阈值，或窗口内至少两个不同 family 的高 surprise；单次极端值不直接 global，除非另设明确 hard-failure 条件。
4. 增加 cooldown/hysteresis，防止同一事件在多层连续 reopen。
5. reopen 后清除对应 prediction cache，并记录 cause event IDs。

验收：单一 outlier 只 local；同族两次 high-surprise 触发 family；跨族证据或高 change probability 触发 global；cooldown 内不重复触发。

---

## 9. M7：修复 Meta episode 与 Stage D

### 9.1 统一 Stage-D 数据格式

#### 问题

runtime 读取 `metadata.meta_questions`，builder 输出顶层 `questions`，默认训练数据没有 Meta 入口；`build_meta_episode()` 的 count 逻辑还会把 2～4 题错误压成最多 2 题。

#### 具体实现

1. 单独生成 Stage-D RL JSONL，每行一个 parent episode，至少包含：

   ```json
   {
     "prompt": "...",
     "label": "...",
     "metadata": {
       "episode_content_id": "...",
       "document_path": "...",
       "meta_questions": [
         {"question_id": "...", "question": "...", "label": "..."}
       ]
     }
   }
   ```

2. 修正题数逻辑：可用题少于 minimum 时丢弃/报错；否则在 `[minimum, min(maximum, available)]` 中确定性选择，不能再次用 minimum 截断 maximum。
3. 问题子集和顺序在离线 builder 中确定一次；runtime 直接消费，不能按每个 sample/replica 的 `sample.index` 重新 shuffle。
4. Stage D launcher 必须切换到该专用 dataset，并在启动时验证每行至少 minimum 个问题。

### 9.2 建立 Meta 的两层 ID 和正确分组

定义：

```text
episode_content_id  # 相同文档、问题子集和顺序；跨 world 共享
meta_trajectory_id  # episode_content_id + latent_world_id + replica_id
```

分组规则：

- suffix return：按 `meta_trajectory_id`，只沿同一个 world/replica 的问题序列计算。
- sibling baseline：按 `(episode_content_id, question_index, latent_world_id)`，只比较同 latent world 的 replica。
- switch pair：允许比较相同 episode content、不同 latent world 的状态。
- branch child：继承 parent 的 `meta_trajectory_id`，并另加 branch ID。

### 9.3 每题只保存增量 world event

在每个 meta question 开始时记录：

```text
world_event_start_offset
hidden_label_start_offset
global_session_call_index
```

问题结束时只把对应 slice 写入 child sample，同时保留全局 call index。这样 belief replay 的 execution trace、public event 和 hidden label 才能一一对齐。

### 9.4 验收条件

- 配置 2～4 题时，数据中实际出现 2、3、4 题 episode。
- 8 个 coupled samples 的问题 ID 和顺序完全一致。
- 不同 latent world 的 suffix return 永不混合。
- 同 latent world 两个 replica 的 sibling group 一致，其他 world 不进入该 group。
- 每题 event slice 数量与该题真实工具调用数一致；全局 session call index 连续。

---

## 10. 训练前必须通过的集成门禁

不能只依赖当前局部单元测试。至少新增一个 CPU deterministic integration suite 和一个小规模真实模型 smoke suite。

### 10.1 CPU deterministic integration suite

必须覆盖：

1. latent world/replica 身份与分组。
2. 四类 coarse world 的预期目标和激活。
3. 所有 corruption adapter 的 schema/artifact 合法性。
4. world failure 后主循环继续。
5. 两候选正 regret 触发 branch、产出 sibling 和 Q replay。
6. canonical belief replay 的 event/label/task-state 对齐。
7. smoother quality/cost 非零梯度和 HBD gate。
8. context hidden 对当前 context 决策有影响。
9. failure utility 使用真实 metadata 正确计算。
10. Meta 2/3/4 题、ID、suffix 和 sibling 分组。

### 10.2 小规模真实模型 smoke suite

建议至少 8～32 道题、每题 4 worlds × 2 replicas，报告：

```text
world_type_count
latent_replica_spec_mismatch_count        # 必须为 0
healthy_corruption_count                  # 必须为 0
context_rule_activation_rate
schedule_effective_rate
structured_result_parse_failure_rate      # 必须为 0
visual_artifact_delivery_rate
candidate_distinct_count / degenerate_rate
branch_eligible_rate / branch_trigger_rate
q_replay_record_count
switch_bundle_count / preinv_bundle_count
belief_replay_event_count / hidden_label_coverage
Bayes-GRPO group-size histogram
checkpoint/model versions
failure-event counts by origin
```

Stage C 的启用门槛至少应包括：latent mismatch=0、结构化结果解析失败=0、group size 正确、belief checkpoint 通过 calibration gate、构造的正 regret 测试能触发 branch。Stage D 还必须通过全部 Meta 验收项。

## 11. 实施者交付要求

每个里程碑的代码提交应同时包含：

1. 实现代码和 schema/version 变更。
2. 对应单元测试与集成测试。
3. 旧日志/checkpoint 是否兼容及迁移策略。
4. 新增 runtime 指标和一次 smoke 输出。
5. 明确说明哪些后续机制仍被 gate 禁用。

不得为了让测试通过而降低关键断言、把异常静默改成默认值，或继续让 Stage 名称掩盖缺失能力。
