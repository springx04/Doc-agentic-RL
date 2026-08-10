# BayesTool-RL：面向非平稳工具世界的信念切换元策略优化

## 1. 问题定义

给定多模态文档 \(D\)、问题 \(q\) 和工具集合

\[
\mathcal{T}=\{\texttt{parse\_document},\texttt{detect\_layout},
\texttt{render\_page},\texttt{crop\_region},\texttt{zoom\_region},
\texttt{ocr\_region},\texttt{extract\_table},\texttt{chart\_to\_table}\},
\]

智能体通过多轮工具调用获得观测，并最终输出答案。第 \(t\) 步的交互历史记为

\[
h_t=(D,q,a_{1:t-1},o_{1:t-1}),
\]

其中 \(a_t\) 可以是工具调用、停止并回答或拒答，\(o_t\) 是工具返回结果。

不同工具在不同文档、页面、区域和运行阶段下可能具有不同的可用性、精度、结构保真度和成本。智能体不能直接观测工具的真实运行状态，只能通过历史调用结果进行推断。因此，该任务被建模为具有隐式工具世界状态的部分可观测决策过程。

BayesTool-RL 的目标是学习一个能够：

1. 在线识别当前工具世界；
2. 根据工具世界信念切换工具路由策略；
3. 在必要时主动执行诊断性调用；
4. 在工具状态变化后及时重新规划；
5. 在正确率、工具成本和失败风险之间进行联合优化

的文档工具智能体。

---

## 2. 非平稳工具世界

### 2.1 分层隐变量

第 \(t\) 步的工具世界状态定义为

\[
z_t=
\left(
z^{\mathrm{session}},
z_t^{\mathrm{context}},
z_t^{\mathrm{shared}},
r_t
\right).
\]

其中：

- \(z^{\mathrm{session}}\)：会话级工具状态，描述后端版本、部署环境、全局延迟、服务可用性等跨任务共享因素；
- \(z_t^{\mathrm{context}}\)：当前文档、页面、区域或内容类型相关的局部状态；
- \(z_t^{\mathrm{shared}}\)：多个工具之间的相关故障因素，例如多个视觉工具共同依赖同一渲染结果；
- \(r_t\)：工具状态所处的非平稳阶段，用于描述运行中发生的状态切换。

工具世界按照如下转移过程演化：

\[
p(z_{t+1}\mid z_t,a_t,m_t),
\]

其中 \(m_t\) 为当前任务内容状态。该转移允许工具质量保持稳定，也允许在调用过程中发生突变或渐变。

### 2.2 工具质量向量

对于工具 \(\tau\in\mathcal{T}\)，其状态不是单一的“可用或不可用”，而是由多维质量向量表示：

\[
g_{\tau,t}=
\left[
g_{\tau,t}^{\mathrm{avail}},
g_{\tau,t}^{\mathrm{semantic}},
g_{\tau,t}^{\mathrm{struct}},
g_{\tau,t}^{\mathrm{calib}},
g_{\tau,t}^{\mathrm{cost}}
\right].
\]

各维度分别表示：

- 可用性；
- 语义结果的正确性；
- 版面、表格和图表结构的保真度；
- 工具输出置信度的校准程度；
- 时间、令牌和计算成本。

隐变量 \(z_t\) 决定各工具质量向量的联合分布，并允许多个工具之间存在相关性。

---

## 3. 智能体结构

BayesTool-RL 将智能体划分为任务状态编码器、工具世界过滤器和信念条件策略三个模块。

### 3.1 任务状态

任务状态编码器从文档内容、问题和已有工具结果中构建内容状态：

\[
m_t=f_{\mathrm{task}}(D,q,a_{1:t-1},o_{1:t-1}).
\]

\(m_t\) 描述当前已知的文档结构、候选页面、局部区域、表格和图表信息，以及尚未解决的任务需求。该状态主要回答“当前需要获取什么信息”。

### 3.2 工具世界过滤器

工具世界过滤器根据当前历史维护后验信念：

\[
b_t=q_\phi(z_t\mid h_t).
\]

工具调用后，信念按照

\[
b_{t+1}
=
\mathcal{F}_\phi
\left(
b_t,m_t,a_t,o_t
\right)
\]

进行更新。

过滤器输出包括：

1. 会话级工具状态后验；
2. 当前页面或区域的局部工具状态后验；
3. 工具间相关故障后验；
4. 状态切换概率；
5. 各工具质量维度的预测分布。

工具世界过滤器同时预测下一次工具调用的观测分布：

\[
p_\phi(o_{t+1}\mid h_t,a_t,b_t),
\]

用于信念更新、状态变化检测和诊断调用评估。

### 3.3 信念条件策略

决策策略显式接收任务状态和工具世界信念：

\[
\pi_\theta(a_t\mid m_t,b_t).
\]

任务状态决定需要解决的内容问题，工具世界信念决定采用哪一条工具路径。对于相同的文档内容状态，不同的工具信念应当产生不同的工具选择。

为防止策略绕过信念模块，工具可靠性相关信息通过结构化信念接口提供给策略；原始工具输出用于内容理解，但不直接作为隐式工具质量标签输入路由模块。

---

## 4. 配对工具世界训练数据

### 4.1 耦合工具世界组

对于同一文档问题样本 \((D,q,y)\)，构造 \(M\) 个语义保持的工具世界：

\[
\mathcal{G}(D,q)
=
\left\{
(D,q,y,z^{(1)}),
\dots,
(D,q,y,z^{(M)})
\right\}.
\]

所有工具世界共享：

- 相同文档；
- 相同问题；
- 相同标准答案；
- 相同文档语义。

不同世界仅改变工具观测通道，包括：

- 工具可用性；
- OCR 噪声与遗漏；
- 版面检测偏差；
- 表格结构破坏；
- 图表解析误差；
- 渲染质量；
- 延迟和调用成本；
- 工具间相关故障；
- 运行中的状态切换。

该设计保证策略差异来自工具世界，而不是答案或文档语义变化。

### 4.2 元回合

对于包含多个问题的同一文档，构造元回合：

\[
\mathcal{E}
=
\left[
(D,q_1),
(D,q_2),
\dots,
(D,q_N)
\right].
\]

元回合内共享会话级工具世界 \(z^{\mathrm{session}}\)，但每个问题具有独立的任务状态。完成一个问题后，会话级信念被保留，任务内容状态被重置。

因此，前序问题中的工具调用可以帮助智能体更快地处理后续问题。

---

## 5. 工具世界识别学习

### 5.1 在线过滤与离线平滑

训练阶段同时构建在线过滤后验和完整轨迹平滑后验：

\[
q_\phi^{\mathrm{filter}}(z_t\mid h_t),
\]

\[
q_\psi^{\mathrm{smooth}}(z_t\mid \tau),
\]

其中 \(\tau\) 为完整交互轨迹。平滑器可以利用未来观测判断早期工具状态，而过滤器只能使用时间 \(t\) 之前的信息。

采用后验蒸馏损失：

\[
\mathcal{L}_{\mathrm{HBD}}
=
\sum_t
D_{\mathrm{KL}}
\left(
\operatorname{sg}
\left[
q_\psi^{\mathrm{smooth}}(z_t\mid \tau)
\right]
\parallel
q_\phi^{\mathrm{filter}}(z_t\mid h_t)
\right),
\]

其中 \(\operatorname{sg}\) 表示停止梯度。

该目标使在线过滤器能够从较短前缀中提前推断完整轨迹最终揭示的工具状态。

### 5.2 状态转移预测

过滤器同时学习隐状态转移和下一观测预测：

\[
\mathcal{L}_{\mathrm{transition}}
=
-\sum_t
\log p_\phi(z_{t+1}\mid z_t,a_t,m_t),
\]

\[
\mathcal{L}_{\mathrm{obs}}
=
-\sum_t
\log p_\phi(o_{t+1}\mid h_t,a_t,b_t).
\]

在合成工具世界中，可以使用已知隐状态进行监督；在真实工具日志中，采用变分推断或弱监督训练。

### 5.3 后验校准

对工具可用性、结果正确性和状态切换概率进行校准：

\[
\mathcal{L}_{\mathrm{calibration}}
=
\sum_{\tau,t}
\operatorname{Brier}
\left(
\hat{p}_{\tau,t},
y_{\tau,t}
\right).
\]

其中 \(\hat{p}_{\tau,t}\) 为过滤器预测概率，\(y_{\tau,t}\) 为可获得的工具结果标签或代理标签。

---

## 6. 信念切换策略优化

### 6.1 支持有效的信念配对

从耦合工具世界中选择具有相同任务内容状态、但工具后验不同的状态对：

\[
(m_t,b_t^{(u)}),
\qquad
(m_t,b_t^{(v)}).
\]

信念配对需要满足：

1. 两个状态来自相同文档和问题；
2. 已获取的任务内容具有一致语义；
3. 两个信念均位于训练工具世界的有效后验支持内；
4. 差异主要来自工具观测历史，而不是任务信息量差异。

在两个工具世界中分别估计最优动作：

\[
a_u^*
=
\arg\max_a Q^{(u)}(m_t,b_t^{(u)},a),
\]

\[
a_v^*
=
\arg\max_a Q^{(v)}(m_t,b_t^{(v)},a).
\]

只对满足

\[
a_u^*\neq a_v^*
\]

的状态对施加信念切换目标。

### 6.2 双向动作排序

信念切换损失定义为

\[
\begin{aligned}
\mathcal{L}_{\mathrm{switch}}
=
-\log \sigma
\Big[
&
\log \pi_\theta(a_u^*\mid m_t,b_t^{(u)})
-
\log \pi_\theta(a_v^*\mid m_t,b_t^{(u)})
\\
+
&
\log \pi_\theta(a_v^*\mid m_t,b_t^{(v)})
-
\log \pi_\theta(a_u^*\mid m_t,b_t^{(v)})
\Big].
\end{aligned}
\]

该目标同时要求：

- 在信念 \(b_t^{(u)}\) 下偏好世界 \(u\) 的动作；
- 在信念 \(b_t^{(v)}\) 下偏好世界 \(v\) 的动作。

### 6.3 信念干预

训练时固定任务状态 \(m_t\)，仅替换路由策略接收的工具信念：

\[
\pi_\theta(a\mid m_t,b_t^{(u)}),
\qquad
\pi_\theta(a\mid m_t,b_t^{(v)}).
\]

通过该内部干预，直接检验动作是否随工具信念变化。为避免无效干预，只使用符合后验支持约束的真实配对信念。

### 6.4 信息不足阶段的一致性约束

在尚未获得能够区分工具世界的观测之前，不同世界中的策略不应提前分化。对配对前缀施加：

\[
\mathcal{L}_{\mathrm{pre\text{-}inv}}
=
D_{\mathrm{JS}}
\left(
\pi_\theta(\cdot\mid m_t,b_t^{(u)})
\parallel
\pi_\theta(\cdot\mid m_t,b_t^{(v)})
\right).
\]

当新的工具结果提供了区分信息后，取消一致性约束并启用信念切换目标。

---

## 7. 贝叶斯决策后悔分支

### 7.1 后验世界粒子

从当前工具世界后验中采样 \(K\) 个粒子：

\[
z_t^{(k)}\sim b_t,
\qquad
k=1,\dots,K,
\]

对应权重为 \(w_k\)。

对于每个粒子，估计候选动作价值：

\[
Q_k(m_t,a)
=
\mathbb{E}
\left[
U(\tau;z_t^{(k)})
\mid m_t,a
\right].
\]

粒子 \(k\) 下的最优动作是

\[
a_k^*
=
\arg\max_a Q_k(m_t,a).
\]

### 7.2 动作共识

后验世界对动作的共识程度定义为

\[
C_t
=
\max_a
\sum_{k=1}^{K}
w_k
\mathbf{1}
\left[
a=a_k^*
\right].
\]

当 \(C_t\) 较高时，不同可能工具世界对下一步动作意见一致，智能体可以直接执行共识动作。

### 7.3 决策后悔

后验平均最优动作定义为

\[
a_B
=
\arg\max_a
\sum_{k=1}^{K}
w_k Q_k(m_t,a).
\]

当前决策后悔为

\[
\mathrm{DR}_t
=
\sum_{k=1}^{K}
w_k
\left[
Q_k(m_t,a_k^*)
-
Q_k(m_t,a_B)
\right].
\]

\(\mathrm{DR}_t\) 衡量因无法确定真实工具世界而产生的实际决策损失。只有当工具不确定性会改变最优动作时，该值才会升高。

### 7.4 分支候选

当

\[
\mathrm{DR}_t > c_{\mathrm{branch}}
\]

时，对有限候选动作进行共享前缀分支。候选集合包括：

1. 高权重后验世界对应的最优动作；
2. 预期降低决策后悔最大的诊断动作；
3. 后验最坏情况下表现稳健的动作；
4. 直接回答或拒答动作。

所有分支共享已有前缀，仅对后续轨迹进行展开。

---

## 8. 诊断动作价值

对于候选诊断动作 \(p\)，定义决策价值：

\[
\mathrm{DVOI}(p)
=
\mathrm{DR}_t
-
\mathbb{E}_{o\sim p_\phi(o\mid h_t,p,b_t)}
\left[
\mathrm{DR}_{t+1}
\right]
-
\lambda_c C(p),
\]

其中 \(C(p)\) 为诊断动作成本。

仅当

\[
\mathrm{DVOI}(p)>0
\]

时执行诊断动作。该准则不奖励一般意义上的信息增益，而只奖励能够降低后续决策损失的信息。

---

## 9. 轨迹效用与策略优化

### 9.1 轨迹效用

完整轨迹在工具世界 \(z\) 下的效用定义为

\[
U(\tau;z)
=
S_{\mathrm{task}}(\tau)
-
\lambda_c C(\tau)
-
\lambda_i I(\tau)
-
\lambda_f F(\tau).
\]

其中：

- \(S_{\mathrm{task}}\)：最终任务得分；
- \(C(\tau)\)：工具调用、令牌和时间成本；
- \(I(\tau)\)：无必要调用、重复调用和低效探索成本；
- \(F(\tau)\)：工具失败、结果误用、输出编造和不可恢复错误成本。

训练不为单个工具步骤设置固定正奖励。诊断动作只有在改善最终净效用时才获得正向信用。

### 9.2 相对工具世界预言机后悔

对于工具世界 \(z\)，定义知道真实工具状态的预言机轨迹：

\[
\tau_z^*
=
\arg\max_\tau U(\tau;z).
\]

策略轨迹的相对后悔为

\[
\operatorname{Regret}(\tau,z)
=
U(\tau_z^*;z)-U(\tau;z).
\]

该量用于衡量智能体因工具世界识别和路由错误造成的损失。

### 9.3 贝叶斯动作价值

当前信念下的贝叶斯动作价值为

\[
Q_B(h_t,a)
=
\sum_{k=1}^{K}
w_k
\mathbb{E}
\left[
U(\tau;z_t^{(k)})
\mid h_t,a
\right].
\]

同一前缀下的多个后续分支使用共享基线构造 sibling advantage：

\[
A_{t,j}^{\mathrm{sib}}
=
R_{t,j}
-
\sum_l
\bar{w}_l R_{t,l},
\]

其中 \(R_{t,j}\) 为第 \(j\) 个分支的后续净效用。

策略损失可采用裁剪的重要性采样目标：

\[
\mathcal{L}_{\mathrm{BA\text{-}policy}}
=
-
\mathbb{E}
\left[
\min
\left(
\rho_{t,j}A_{t,j}^{\mathrm{sib}},
\operatorname{clip}
(\rho_{t,j},1-\epsilon,1+\epsilon)
A_{t,j}^{\mathrm{sib}}
\right)
\right],
\]

\[
\rho_{t,j}
=
\frac{
\pi_\theta(a_{t,j}\mid m_t,b_t)
}{
\pi_{\theta_{\mathrm{old}}}(a_{t,j}\mid m_t,b_t)
}.
\]

---

## 10. 总体训练目标

总体目标为

\[
\mathcal{L}
=
\mathcal{L}_{\mathrm{BA\text{-}policy}}
+
\alpha\mathcal{L}_{\mathrm{HBD}}
+
\beta\mathcal{L}_{\mathrm{switch}}
+
\gamma\mathcal{L}_{\mathrm{pre\text{-}inv}}
+
\eta\mathcal{L}_{\mathrm{transition}}
+
\mu\mathcal{L}_{\mathrm{obs}}
+
\xi\mathcal{L}_{\mathrm{calibration}}.
\]

各部分作用如下：

- \(\mathcal{L}_{\mathrm{BA\text{-}policy}}\)：优化工具世界后验下的任务效用；
- \(\mathcal{L}_{\mathrm{HBD}}\)：提高早期工具状态识别能力；
- \(\mathcal{L}_{\mathrm{switch}}\)：学习随工具信念切换动作；
- \(\mathcal{L}_{\mathrm{pre\text{-}inv}}\)：避免在无区分信息时提前产生世界特定策略；
- \(\mathcal{L}_{\mathrm{transition}}\)：学习非平稳工具状态转移；
- \(\mathcal{L}_{\mathrm{obs}}\)：学习工具结果分布；
- \(\mathcal{L}_{\mathrm{calibration}}\)：校准工具质量后验。

---

## 11. 训练流程

### 阶段 A：工具世界识别预训练

使用工具日志、合成扰动和配对工具世界训练：

- 工具状态编码器；
- 在线过滤器；
- 完整轨迹平滑器；
- 工具结果预测模型；
- 状态切换检测器。

主要优化：

\[
\mathcal{L}_{\mathrm{HBD}}
+
\mathcal{L}_{\mathrm{transition}}
+
\mathcal{L}_{\mathrm{obs}}
+
\mathcal{L}_{\mathrm{calibration}}.
\]

### 阶段 B：单任务信念条件策略训练

在单个文档问题上进行多工具世界 rollout，使策略学习：

\[
\pi_\theta(a_t\mid m_t,b_t).
\]

该阶段主要建立工具信念与动作选择之间的基本对应关系。

### 阶段 C：配对世界信念切换训练

从相同任务状态的不同工具世界中构造支持有效的信念对，执行：

- 双向动作排序；
- 信念干预；
- 区分前一致性训练；
- 决策后悔分支；
- sibling advantage 策略更新。

该阶段是策略获得工具世界适应能力的核心阶段。

### 阶段 D：跨任务元策略训练

在同一文档的多个问题之间共享会话级信念，训练：

- 工具知识跨问题迁移；
- 少量调用下的快速工具识别；
- 工具状态突变后的信念重置；
- 长时程调用成本优化。

---

## 12. 推理算法

推理阶段采用 Persistent Consensus–Probe–Reopen 策略。

### 12.1 持久化信念

新问题开始时：

- 重置任务内容状态 \(m_0\)；
- 保留会话级工具信念；
- 初始化新的局部上下文信念。

### 12.2 共识执行

从后验中采样工具世界粒子并计算动作共识 \(C_t\)。

当

\[
C_t\ge \delta_{\mathrm{consensus}}
\]

时，直接执行后验共识动作。

### 12.3 决策性探测

当共识不足且决策后悔较高时，计算候选诊断动作的 \(\mathrm{DVOI}\)。

若存在

\[
\max_p \mathrm{DVOI}(p)>0,
\]

则执行价值最大的诊断动作。

### 12.4 稳健提交

当诊断动作收益不足或预算受限时，执行稳健动作：

\[
a_{\mathrm{robust}}
=
\arg\max_a
\operatorname{CVaR}_{z\sim b_t}
Q_z(m_t,a),
\]

或采用后验最坏世界最大化：

\[
a_{\mathrm{robust}}
=
\arg\max_a
\min_{z\in\operatorname{supp}(b_t)}
Q_z(m_t,a).
\]

### 12.5 预测惊异与重新打开

工具调用后计算预测惊异：

\[
S_{t+1}
=
-\log
p_\phi(o_{t+1}\mid h_t,a_t,b_t).
\]

根据惊异来源执行不同级别的重新打开：

- 局部重新打开：重置当前页面或区域状态；
- 工具族重新打开：重置相关工具的共享状态；
- 全局重新打开：检测到会话级状态切换时重置全局工具信念。

### 12.6 停止与拒答

比较立即停止风险和继续调用风险：

\[
R_{\mathrm{stop}}
=
\mathbb{E}_{z\sim b_t}
\left[
L(\hat{y}_t,y;z)
\right],
\]

\[
R_{\mathrm{continue}}
=
\min_a
\left[
C(a)
+
\mathbb{E}
\left[
R_{t+1}\mid h_t,a
\right]
\right].
\]

当

\[
R_{\mathrm{stop}}
\le
R_{\mathrm{continue}}
\]

时停止调用并输出答案。

若答案风险超过拒答阈值，则输出拒答或不确定性声明。

---

## 13. 推理伪代码

```text
Input:
    document D
    question q
    persistent session belief b_session
    tool budget B

Initialize:
    task state m_0
    local belief b_local
    combined belief b_0 = Merge(b_session, b_local)
    history h_0

for t = 0, 1, ..., B:

    1. Encode task state:
       m_t = TaskEncoder(D, q, h_t)

    2. Update tool-world posterior:
       b_t = ToolWorldFilter(h_t, m_t)

    3. Sample posterior world particles:
       {z_t^(k), w_k}_{k=1}^K ~ b_t

    4. Estimate particle-specific action values:
       Q_k(m_t, a)

    5. Compute:
       action consensus C_t
       Bayes action a_B
       decision regret DR_t

    6. If stopping risk <= continuation risk:
           return final answer or abstention

    7. If C_t >= consensus threshold:
           a_t = posterior consensus action

       Else if DR_t >= regret threshold:
           evaluate diagnostic actions with DVOI

           if max DVOI > 0:
               a_t = best diagnostic action
           else:
               a_t = robust action

       Else:
           a_t = Bayes-optimal action

    8. Execute a_t and receive observation o_t

    9. Compute predictive surprise:
       S_t = -log p(o_t | h_t, a_t, b_t)

   10. If surprise indicates change point:
           reopen local, tool-family, or global belief

   11. Update:
           h_{t+1} = h_t ∪ {(a_t, o_t)}
           persistent session belief

Return:
    answer, abstention, or budget-exhausted result
```
