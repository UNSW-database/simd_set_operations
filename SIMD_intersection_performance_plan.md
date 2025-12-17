# 2-set SIMD 交集性能分析与实验计划

结合 `setops` 代码的仪器化与可观测成本模型，给出实验目标、模型要点、变量设计与实验矩阵，便于复现与论文撰写。

## 目标与范围
- 覆盖排序整型数组的 2-set SIMD 交集路径，比较标量 vs SIMD、线性 vs 搜索、分支 vs 无分支、过滤 vs 无过滤、SSE/AVX2/AVX-512。
- 按三阶段模型（推进 / 预过滤 / 微核+写回）对齐计数与 perf，确保成本项可由测量验证。

## 仪器化与可观测量
- Stage1：`linear_steps`、`advance_a/advance_b`，以及搜索族的 `search_probes`（倍增/块跳探测）和 `search_binary_steps`（二分步数）。
- Stage2：`stage2.lowbyte_*` / `stage2.bytegate_*` 的 `probes/hits`（统一视为过滤通过率 p = H/P）。
- Stage3：交集大小 `|A∩B|`（真实输出或 `σ·n_b` 估计），写回路径依 ISA 而异。
- Perf（Linux）：`cycles`、`instructions`、`l1d/l1i/ll` miss、`branch_misses`。

## 成本模型（概要）
- Stage1：线性 `Cost_S1_lin ≈ L*c_iter + A_a*c_adv_a + A_b*c_adv_b`；搜索 `Cost_S1_search ≈ S_p*c_probe_search + S_b*c_bin_step`（用计数替代 log 近似）；分支惩罚 `branch_misses*c_mispredict`。
- Stage2：`Cost_S2 ≈ P*c_probe + H*c_pass`（无过滤 P=H=L，可将 `c_pass` 并入 Stage3）。
- Stage3：`I3≈L*(H/P)`（线性）或 `I3≈σ*n_b/W_cmp`（搜索）；`Cost_S3≈I3*c_kernel + |A∩B|*c_write`（写回常数区分 movmsk 路径与 AVX-512 compressstore）。  
- Miss 残差：`α*l1d_miss + β*ll_miss + γ*branch_misses` 视作回归残差/干扰项，承认 OoO/MLP 重叠，不做物理叠加解释。

## 控制变量与取值
- 选择率 σ：{1e-4, 1e-3, 1e-2, 0.1, 0.5}。
- 尺寸比 s（skew）：对数刻度 {1, 1e-1, 1e-2, 1e-3}。
- 密度/值域 ρ：{0.1%, 1%, 5%, 20%}。
- 工作集大小（max_len）：{2^12(L1), 2^17(L2), 2^20–2^22(LLC/更大)}。
- ISA：SSE、AVX2、AVX-512；分支版/无分支版；过滤开/关。
- 数据类型：合成（可控 σ/s/ρ/max_len），真实（WebDocs/Graph 等对齐档位）。

## 实验矩阵（示例）
- **E1 Pipeline 切换**：σ ∈ {1e-4..0.5}，s ∈ {1e-3..1}，ρ=0.1%，max_len ∈ {2^17,2^20}；对比线性（shuffling/broadcast）vs 搜索（lbk/galloping），输出 `search_probes/binary_steps` vs 性能。
- **E2 过滤收益**：σ ∈ {1e-4,1e-3,1e-2}，s ∈ {1e-3,1e-2}，ρ 低/中，ISA={AVX2,AVX-512}；对比 `_prefilter` / bytegate vs baseline。
- **E3 微核对比**：σ ∈ {1e-4..0.5}，ρ=1%，s≈1，max_len ∈ {2^17,2^20}；Rotate vs Broadcast vs Byte gate，跨 ISA。
- **E4 写回策略**：σ ∈ {1e-4..0.5}，ρ=1%，s≈1，max_len L2/L3；AVX2 (movmsk) vs AVX-512 (mask_compressstore)。
- **E5 分支 vs 无分支**：σ ∈ {1e-3,0.1}，max_len {2^12,2^20}，ISA={SSE,AVX2,AVX-512}；对比 `_branch` vs 非 branch（每元素 instructions/CPI/branch_misses 展示）。
- **E6 真实数据校准**：WebDocs/Graph，选择与合成匹配的 σ/ρ 档；对比合成预测与实测。
- **E7 ISA/频率敏感**：固定算法（如 shuffling/broadcast），σ≈0.01，max_len L2/L3；AVX2-only vs AVX-512，必要时锁频。

## 观测与绘图
- Stage1：`linear_steps` vs σ/s（控制其他变量）以及搜索族的 `search_probes/binary_steps`，观察线性/搜索交叉点。
- Stage2：`skip = 1 - H/P` vs σ/ρ，判断过滤收益。
- Stage3：`(cycles - miss_penalty)` vs `|A∩B|`，比较写回路径与微核（Rotate/Broadcast/Byte gate），写回区分 movmsk vs AVX-512 compressstore。
- 分支：`branch_misses` 与性能对比 `_branch` vs 非 branch。

## 结论基线（期望方向）
- 低 σ、高 skew → 搜索 + 过滤（lbk/galloping + prefilter）；低 σ、中熵 → 过滤线性族（bmiss/qfilter）；σ 中高、s≈1 → 线性无过滤；高 σ 下 AVX-512 写回更优，低 σ 差异小。

## 标定与拟合流程
1. 按代表性档位跑实验，采集 `L/A/P/H`、`|A∩B|`、perf。
2. 按算法族/ISA 回归 `c_iter/c_adv/c_probe/c_kernel/c_write/α/β/γ/c_mispredict`。
3. 用模型预测其它配置相对优劣，缩小实验矩阵；验证偏差大的点，迭代常数。
