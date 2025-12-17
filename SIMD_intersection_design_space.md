# SIMD 交集设计空间（排序数组专用）与成本模型

面向 `setops` 库中排序整型数组的 2-set SIMD 交集路径，复刻原设计空间文档的结构，并以代码可观测的三阶段成本模型替换旧公式，便于实验与论文撰写。

## 研究范围与假设
- 数据结构：排序整型数组；本稿不覆盖 BSR/Roaring/FESIA。
- 平台：x86-64，ISA 覆盖 SSSE3/SSE4.2、AVX2、AVX-512F/CD；Rust 端主要使用 `std::simd`，少量 `std::arch` intrinsic。
- 目标：Pipeline / Prefilter / Microkernel 三层抽象下，模型与计数一一对应，支撑顶会级的可验证性。

## 三层架构概览
1) **Pipeline（推进层）**  
   - 线性扫描：同步推进 A/B 游标（`shuffling_*`、`broadcast_*`、`bmiss`/`qfilter`、`vp2*`、`conflict*`）。  
   - 搜索/跳跃：倍增/二分驱动小集合（`lbk_v*`、`galloping_*`）。  
   - 分支 vs 无分支：`*_branch` 在掩码稀疏、分支可预测时指令更少；无分支在分支失误高时更稳。
2) **Stage2 预过滤**  
   - 无过滤：直接入微核。  
   - 统一过滤接口：低字节探针或 byte gate（`bmiss`/`qfilter`/`sttni`、`*_prefilter`），计数暴露为 `probes/hits`。
3) **Stage3 微核与写回**  
   - Rotate-and-Compare（`shuffling_*`）；Broadcast-and-Compare（`broadcast_*`、所有搜索族）；Byte Gate 联核（`bmiss`/`qfilter`）；Block-compare（`galloping`/`lbk_v3`）；特殊核 VP2/Conflict（AVX-512）。  
   - 写回：SSE/AVX2 用 `movmsk+scatter/LUT`；AVX-512 用 `mask_compressstore`。

## 可观测计数（代码钩子）
- Stage1：`linear_steps`、`advance_a/advance_b`、`search_probes`（倍增/块跳探测）和 `search_binary_steps`（二分步数），用于线性与搜索族的真实事件计数。  
- Stage2：`lowbyte_*` / `bytegate_*` 的 `probes/hits/skipped`。  
- Perf（Linux）：`cycles`、`instructions`、`l1d/l1i/ll` miss、`branch_misses`。  
- 真实输出：交集大小 `|A∩B|`（或 `σ·n_b` 估计）。

## 成本模型（可观测、可回归）
符号：`n_a≥n_b`，`σ=|A∩B|/n_b`，`s=n_b/n_a`，向量宽 `W`；计数 `L=stage1.linear_steps`，`A_a/A_b=advance_a/b`，`S_p=search_probes`，`S_b=search_binary_steps`；`P/H` 为过滤探测/通过（无过滤时 `P=H=L`）；`T=|A∩B|`（或 `σ·n_b` 估计）；perf：`cycles/instructions/l1d_miss/ll_miss/branch_misses`。

- **Stage1（推进）**  
  线性：`Cost_S1_lin ≈ L * c_iter + A_a * c_adv_a + A_b * c_adv_b`，`L≈n_b/W_lin`。  
  搜索：`Cost_S1_search ≈ S_p * c_probe_search + S_b * c_bin_step`（通过计数而非 log 估计），`L` 仍可用于 enter 次数近似；分支惩罚 `Cost_branch≈branch_misses * c_mispredict`。
- **Stage2（预过滤，统一公式）**  
  `Cost_S2 ≈ P * c_probe + H * c_pass`（如需粗化，可将 `c_pass` 并入 Stage3）。
- **Stage3（微核 + 写回）**  
  进入次数：线性 `I3≈L*(H/P)`；搜索可用 `I3≈σ*n_b/W_cmp`。  
  `Cost_S3 ≈ I3 * c_kernel + T * c_write`，`c_write` 区分 SSE/AVX2（movmsk+scatter）与 AVX-512（mask_compressstore）。  
  外部惩罚（回归残差项）：`miss_penalty≈α*l1d_miss + β*ll_miss + γ*branch_misses`，用于工程拟合，**不**声称可物理叠加，需承认 OoO/MLP 重叠。
- **总成本**  
  `Cost_total ≈ Cost_S1 + Cost_S2 + Cost_S3 + Cost_branch + miss_penalty`  
  常数按算法族/ISA 拟合，少量代表性点做线性回归即可。

## 阶段 3 指令微核明细
### Rotate-and-Compare
- 128-bit：`pshufd/pshufb` + `pcmpeqd` + `vpmovmskb`；Rust API：`simd_swizzle!`、`Simd::simd_eq`、`Mask::to_bitmask`；调用：`shuffling_sse`。  
- 256-bit：`vpermps/vpermd` + `vpcmpeqd` + `vpmovmskb`；调用：`shuffling_avx2`、`lbk_v3_avx2`。  
- 512-bit：`permutexvar/vpermd` + `vpcmpd` + `kmovw`；调用：`shuffling_avx512`、`galloping_avx512`。

### Broadcast-and-Compare
- 128-bit：`pbroadcastd` + `pcmpeqd` + `vpmovmskb`；调用：`broadcast_sse{,_branch}`、`lbk_v1x4_sse`。  
- 256-bit：`vpbroadcastd` + `vpcmpeqd` + `vpmovmskb`；调用：`broadcast_avx2{,_branch}`、`lbk_v1x16_avx2`。  
- 512-bit：`vpbroadcastd` + `vpcmpd` + `kmovw`；调用：`broadcast_avx512{,_branch}`、`lbk_v1x32_avx512`。

### Byte Gate（过滤与比较一体）
- 128-bit：`pshufb/pcmpeqb/ptest`、`pcmpestrm`；调用：`bmiss`、`qfilter`、`shuffling_sse_bsr` 等。  
- 256-bit：`vmovdqu/vpand/vpcmpeqd/vpmovmskb`；调用：`lbk_v1x16_avx2_prefilter`、`lbk_v3_avx2_prefilter`、`galloping_avx2_prefilter`。  
- 512-bit：`vmovdqu64/vpandd/vpcmpd/kmovw`；调用：`lbk_v1x32_avx512_prefilter`、`lbk_v3_avx512_prefilter`、`galloping_avx512_prefilter`。

### Mask Compare（基础）
- 128-bit：`pcmpeqd/pcmpgtd/ptest`；调用：`bmiss`、`broadcast_sse`、`lbk_v3_sse`、`simd_galloping_sse`。  
- 256-bit：`vpcmpeqd/vpcmpgtd/vpmovmskb`；调用：`broadcast_avx2`、`lbk_*_avx2`、`galloping_avx2`。  
- 512-bit：`vpcmpd/kvptest/kmovw`；调用：`broadcast_avx512`、`lbk_*_avx512`、`galloping_avx512`、`vp2intersect_emulation`。

### Mask Compression / 写回
- SSE：`pshufb + movdqu` 或 `movmsk+LUT`。  
- AVX2：`movmsk+shuffle/scatter`（纯 AVX2 无 compressstore；如使用 `_mm256_mask_compressstoreu_epi32` 需 AVX-512VL 扩展）。  
- AVX-512：`_mm512_mask_compressstoreu_epi32` / `_mm512_mask_compress_epi32`。

### Block Compare
- 128-bit：多次 `pcmpeqd+por`；调用：`simd_galloping_impl::<i32,4>`。  
- 256-bit：多次 `vpcmpeqd+vpor`；调用：`simd_galloping_impl::<i32,8>`、`lbk_v3_avx2`。  
- 512-bit：多次 `vpcmpd+kor`；调用：`simd_galloping_impl::<i32,16>`、`lbk_v3_avx512`。

### VP2 Emulation（AVX-512）
- 指令：`alignr+shuffle+cmpneq` + `mask_cmpneq`；调用：`vp2intersect_emulation{,_branch}`。

### Conflict Detection（AVX-512）
- 指令：`vinserti32x8`、`_mm512_conflict_epi32`、`mask_compressstoreu`；调用：`conflict_intersect{,_branch}`。

## 算法路径映射（Pipeline → Stage2 → Stage3）
- `shuffling_{sse,avx2,avx512}{,_branch}`：线性 / 无过滤 / Rotate-and-Compare → Mask → 写回。  
- `broadcast_{sse,avx2,avx512}{,_branch}`：线性 / 无过滤 / Broadcast-and-Compare → Mask → 写回。  
- `bmiss{,_sttni}{,_branch}`、`qfilter{,_branch}`：线性 / Byte gate 过滤 / 联核（过滤+比较+写回）。  
- `lbk_v1/v3_*`、`galloping_*`：搜索 / 可选过滤（`_prefilter`）/ Broadcast 或 Block-compare + 写回。  
- `vp2intersect_emulation{,_branch}`、`conflict_intersect{,_branch}`（AVX-512）：线性 / 无过滤 / VP2 仿真或 Conflict 检测 + 压缩写回。

## 算法族说明与适用场景
- **线性 Pipeline**：`shuffling`/`broadcast`，适合 `s≈1` 或中高 σ；`*_branch` 在掩码稀疏、可预测时指令更少，无分支在分支失误高时更稳。  
- **Byte Gate（bmiss/qfilter/sttni）**：线性 + 多级门卫，伪阳性低，适合低 σ、中等熵；指令较重但过滤强。  
- **搜索族（LBK/Galloping）**：倍增/二分 + 可选预过滤；收益依赖极低 σ、高度 skew（s≪1），否则探测/随机访存成本会抵消收益。Block 版本在更宽 ISA 吞吐更高。  
- **AVX-512 特殊核**：VP2 仿真、Conflict 检测用于低 σ 且需压缩写回的场景。

## 实验与调优要点
- 自变量：σ、s、密度/值域、工作集大小（max_len 控制驻留层级）、ISA、分支版/无分支版、过滤开关。  
- 观测：`L/A/P/H`、`skip = 1 - H/P`、交集大小 `T`、perf（cycles/instructions/miss/branches）。  
- 绘图建议：Stage1 `L` vs σ/s；Stage2 `skip` vs σ/密度；Stage3 `(cycles - miss_penalty)` vs `T` 比较写回路径和微核（Rotate vs Broadcast）。  
- 经验决策：低 σ、高 skew → 搜索+过滤；低 σ、中熵 → 过滤线性族（bmiss/qfilter）；σ 中高、s≈1 → 线性无过滤；高 σ 下 AVX-512 写回优势更明显，低 σ 差异小。

## 标定流程（建议）
1. 选少量代表性配置（σ=1e-4/1e-2/0.1/0.5，s=1/1e-1/1e-3，低/中密度，工作集 L1/L3 档）跑实验，收集 `L/A/P/H/T` 与 perf。  
2. 按算法族/ISA 回归常数 `c_iter/c_adv/c_probe/c_kernel/c_write/α/β/γ/c_mispredict`。  
3. 用模型预测其它配置的相对优劣，优先测试模型给出的候选，缩小实验矩阵。
