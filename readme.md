VJEPA-Gym 核心架构与详细设计规范 (v2.0)

本文档是 Vibe Coding 的全局系统上下文。系统被高度浓缩为 6 个核心文件：main.py, configs.yaml, core_models.py, environment.py, latent_planner.py, trainer_engine.py。

一、 core_models.py: 网络结构与张量流

该模块包含特征提取器与动作条件预测器。核心难点在于Token拼接（Interleaving）与因果注意力掩码（Causal Mask）。

1. VJEPA 2.1 Encoder (特征提取 - 完全对齐官方规范)

版本映射：vjepa2_1_vit_large_384

模型输入：[B, T, 3, 384, 384]，像素值归一化至 [0.0, 1.0]。

输出尺寸：由于 Patch Size 为 16，单帧切分为 $24 \times 24 = 576$ 个 Token。输出 Shape 为 [B, T, 576, 1024]。

API 用法警示：官方 V-JEPA 期望输入具有时序维度，即便推理单帧，也必须显式增加 T=1 维度。

2. AC-Predictor (动作条件预测器)

网络规模：12层 Transformer Block，12头注意力，内部维度 D_pred = 768。

各投影层参数：

visual_proj: Linear(1024, 768)

action_proj: Linear(7, 768)

state_proj: Linear(7, 768)

output_proj: Linear(768, 1024)

Token 组合规则 (Interleaving)：
每一帧拼接为 [Action_Token, State_Token, Visual_Token_1, ..., Visual_Token_576]。
单帧 Token 总数 = $1 + 1 + 576 = 578$。

时间位置编码 (RoPE)：
仅对每帧前两个 Token（Action 和 State）注入基于相对时间的 Rotary Position Embedding。视觉 Token 保留 Encoder 的原图空间位置编码。

Block-causal Attention Mask：
序列总长为 $T \times 578$。注意力掩码必须保证时间步 $t$ 的 token 无法 attend 到 $t+1$ 及之后的 token，但允许同一帧内的 token 相互可见。

二、 environment.py: 可回滚物理状态机

封装 MuJoCo，实现“物理世界的分支预测”。

1. 核心状态栈设计

必须同时维护两条历史线，保证潜空间观测与物理绝对状态的一致性：

physics_stack: 存储 env.physics.get_state().copy() (扁平化的 qpos, qvel)。

frame_buffer_stack: 存储历史渲染图像的 deque，尺寸为 [T_context, 384, 384, 3]。

2. 标准回滚 API 范本

# 压栈操作 (用于开启一条假想轨迹)
def push_state(self):
    self._phys_stack.append(self.physics.get_state().copy())
    self._frame_stack.append(copy.deepcopy(self.frame_buffer))

# 弹栈操作 (用于恢复真实物理世界)
def pop_state(self):
    self.physics.set_state(self._phys_stack.pop())
    with self.physics.reset_context():  # 致命易错点：不调用此项将导致后续触碰传感器失效
        pass
    self.frame_buffer = self._frame_stack.pop()


3. 动作与状态空间 (7D)

Action (7维连续)：[Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper_force]。范围 [-1.0, 1.0]。

State (7维连续)：绝对坐标 [x, y, z, roll, pitch, yaw, gripper_width]。

三、 latent_planner.py: CEM 纯潜空间规划

全程不调用物理引擎的 step()，只在特征空间推演。

1. Latent Rollout (自回归展开)

输入：历史上下文 $Z_{ctx}$ [1, T_ctx, 576, 1024]，动作序列 $A$ [1, H, 7]，状态序列 $S$ [1, H, 7]。

过程：循环 $H$ 次（Horizon）。每次预测 $\hat{z}_{t+1}$，并将其追加到 $Z_{ctx}$ 的末尾，剔除最早的一帧，形成新的滑动窗口。

能量函数 (距离度量)：计算预测的终态 $\hat{z}_{t+H}$ 与目标图像特征 $z_{goal}$ 之间的距离：


$$Cost = || \text{Mean}_{patches}(\hat{z}_{t+H}) - \text{Mean}_{patches}(z_{goal}) ||^2$$

2. 交叉熵方法 (CEM) 参数

采样候选数 (N_candidates): 512 （利用潜空间高并发优势）。

规划步长 (Horizon): 15 ~ 30 步。

精英数量 (N_elites): 64。

迭代次数 (N_iters): 10。

物理验证机制 (Physics Verification)：对最终优化出的 Top 5 动作序列，调用 environment.py 的 push_state()，在真实 MuJoCo 中执行并计算真实特征差异，随后 pop_state()，选出最优。

四、 trainer_engine.py: 训练与微调流

1. HDF5 数据流格式

存储格式必须对齐以下结构，以保证 I/O 效率：

frames: [Episode_Len, 384, 384, 3], uint8

actions: [Episode_Len - 1, 7], float32

states: [Episode_Len - 1, 7], float32

2. AC-Predictor 无监督损失函数

使用 Teacher Forcing 和长程 Rollout 结合的策略：

Teacher Forcing Loss ($L_{TF}$): 输入真实 $z_{t}$，预测 $\hat{z}_{t+1}$。


$$L_{TF} = \text{SmoothL1}(\hat{z}_{1:T}, z_{target})$$

Rollout Loss ($L_{roll}$): 输入真实 $z_0$，自回归预测 K 步。


$$L_{roll} = \frac{1}{K} \sum_{k=1}^{K} \text{SmoothL1}(\hat{z}_{k}, z_{target, k})$$

Total Loss: $L = L_{TF} + 0.5 \times L_{roll}$

3. VJEPA Encoder 微调机制 (EMA 指导)

为防止表征坍塌，微调阶段禁止直接使用对比学习，需遵循官方架构：

实例化 EMA Target Encoder，动量 $\tau = 0.996$。

掩码策略：对原图施加 75% 的 Block Masking。Online Encoder 仅看未遮挡区域，Target Encoder 看全图。

计算 Online Encoder 对掩码区域预测特征与 Target Encoder 提取特征的 Smooth L1 距离。

解冻策略：仅解冻 ViT-Large 的最后 2~8 层 Transformer Block。