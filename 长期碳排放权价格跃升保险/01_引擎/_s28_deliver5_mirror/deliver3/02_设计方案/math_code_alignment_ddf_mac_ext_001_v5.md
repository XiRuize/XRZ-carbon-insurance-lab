# 数学-代码对齐矩阵 —— ddf_mac_ext_001 2026–2030 情景外推设计 v5

**任务 ID**：`ddf_mac_ext_001`　｜　**产出方**：铸模·构建者 V1.4L　｜　**生成时间**：2026-10-04
**版本**：v5（在 v4 基础上增补 2026–2030 前望段与自洽性约束；v4 快照 `math_code_alignment_ddf_mac_ext_001_v4.md` 保留，未修改）
**用途**：逐条把 2026–2030 设计文档（`ddf_mac_ext_001_forge_2026-2030情景外推设计_v1.md`）的数学对象映射到代码骨架的位置与上游只读来源，供 Tester 实现与 Critic 审计。

> **只读声明**：v1/v2/v4 及无版本版对齐矩阵、既有设计 v1–v5、既有契约与 CSV **一律不改**；本文件为新建 v5。上游路径动态引用，禁复制。

---

## 一、核心公式 → 代码位置（M-1 ~ M-28）

### 1.1 回望段（2023–2025，承接 v4，逐位不变）

| 编号 | 数学对象 | 公式 | 代码位置 | 上游/来源 | 校验方式 |
|:---|:---|:---|:---|:---|:---|
| M-1 | 归一化输入 | `yᵢₜ=Y_raw/ȳᵢ`（k,l,e,b 同） | `ext_common.frozen_baselines` + 链接后除基准 | 上游面板 xlsx | 与上游量纲化面板逐位一致 |
| M-2 | HC-2 冻结基准 | `ȳᵢ=mean_{1995..2022}Y_raw` | `ext_common.frozen_baselines` | 上游面板 | 北京 1995 反推 ȳ=7396.985761776982 |
| M-3 | dDdy | `b1+b2(y+b)+dK·k+dL·l+dE·e` | `ext_common.mac_orig` | C_LP_v1.json | 2022 复现 190.53058207373442 |
| M-4 | dDdb | `b1+1+b2(y+b)+dK·k+dL·l+dE·e` | `ext_common.mac_orig` | 同上 | 同上 |
| M-5 | MAC_scaled | `−dDdb/dDdy` | `ext_common.mac_orig` | — | T-01 派生字段比对 |
| M-6 | MAC_orig | `MAC_scaled·(ȳᵢ/b̄ᵢ)·10⁴` | `ext_common.mac_orig` | — | 元/吨；840 条逐位 |
| M-7 | 偏离度 dev | `MAC_scaled·(ȳᵢ/b̄ᵢ)·C_raw/Y_raw` | `ext_common.dev_share` | — | dev≤1 谓词 O4 |
| M-8 | 增长率链接 | `x_{t,norm}=x_{t−1,norm}(1+g_t)` | `ext_stage2_link.chain_var` | 勘探产出增速 | 与 `x_t,raw/x̄` 恒等 |
| M-9 | K 链接（甲/乙） | `k_{t,norm}=k_{t−1,norm}(1+g_K)` | `ext_stage3_gk_band` | T-06 §七 g_K 分位 | 三档端点自洽 |
| M-10 | W* 定义 | `W*={c:max_i dev_i(c)≤1}` | `ext_stage1_Wstar` | deliver2 s07_exact_window | 两段端点逐位 |
| M-11 | b1 联动 | `b1=−1+cβ` | `s00_common.run_at` | deliver2 | β=1 时 b1=−1+c |
| M-12 | 年度函数 | `f_t(c)=nanmean_i MAC_orig(i,t;c)` | `ext_stage5_report.two_dim_report` | deliver2 s14 | 2022 复现 f-区间 |
| M-13 | 合法占比 | `D_t=\|{Λ_t=1}\|/\|W*\|` | `ext_stage5_report.legality_fraction` | — | 阈值 0.95 |
| M-14 | 前沿外占比 | `φ_t=#{D<0}/30` | `ext_stage5_report.out_of_frontier_fraction` | — | 交付锚点 c0 |
| M-15 | 误差传导速估 | `ΔMAC_rel≈Ση_v·r_v` | `ext_stage4_cband` | T-05 η | **仅诊断**；K 带禁 η_k 乘推（P-10） |
| M-16 | ∂D/∂b 变号（O6） | `λ_b=1 ⟺ dDdb>0`；`B_t=#{λ_b=1}/30` | `ext_stage6_O6_regularity` | v3 §3 | 逐 (i,t,c) 记录；禁静默（P-9） |
| M-17 | 2025 C 降级 | `C(i,2025)=C(i,2024)·(1+g_C^{extrap})` | `ext_stage7_dg2025` | Scout-I | 连乘截断 2024；双分支并列 |
| M-18 | 外推额外加宽比 | `halfwidth_t=(U.hi−U.lo)/2`；`R_widen=halfwidth_t/halfwidth_ref` | `ext_stage8_widen_ratio` | v4 §2 / 用户裁决 A | 尺度不变；不含 naive_err；ref 同口径 |

### 1.2 前望段（2026–2030）新增映射（M-19 ~ M-28）

| 编号 | 数学对象（设计 §） | 公式 | 代码位置 | 上游/来源 | 校验方式 |
|:---|:---|:---|:---|:---|:---|
| **M-19** | **前望段起点承接（§1.3）** | `x_{2025,norm}` 直接读入 T-09；`x_{2026,norm}=x_{2025,norm}·(1+g_{2026})` | `ext_stage9_prospect_link` | `T09_output.json`（只读，逐位） | 前望段起点与 T-09 2025 逐位相等 |
| **M-20** | **情景参数映射（§2.1，P1–P8）** | `g_Y,t/g_L,t/g_E,t` 取 P1/P7/P3 档标量（全国代分省）；`g_C,t=(1+g_Y,t)(1+g_s,t)−1`（P2） | `ext_scenario_params.load` | Route M §2 | 与 parameters_handover 2026-2030 组一致 |
| **M-21** | **K 前望段（层丙）** | `g_K,t=r·(1+g_I,t)−δ_t`，`r=0.12890728324110395`；`k_{t,norm}=k_{t−1,norm}(1+g_K,t)` | `ext_stage9_prospect_link`（K 支） | T-10 §6 / Route M P6 | 逐档 g_K 与 §2.3 表一致 |
| **M-22** | **K 前望段（层甲/乙）** | `k_{t,norm}=k_{2025,norm}·(1+g_K)^{t−2025}`（标量档幂） | `ext_stage9_prospect_link`（K 支） | T-10 §5 / D-005 | 端点 low/mid/high 自洽 |
| **M-23** | **碳强度锚年化（§3.1, D3-1）** | `∏(1+g_s,t)=0.83`；`g_s*=(0.83)^{1/5}−1=−0.03657030` | `ext_stage10_anchor_consistency` | 官方硬锚（政府工作报告/碳达峰方案） | 残差 ≤1e-6 |
| **M-24** | **非化石锚年化（§3.1, D3-2）** | `(1−p):0.783→0.750`；`g_{(1−p)}*=(0.750/0.783)^{1/5}−1=−0.0085746` | `ext_stage10_anchor_consistency` | 官方硬锚 | 残差 ≤1e-6 |
| **M-25** | **排放分解恒等式（§3.2, D3-3）** | `(1+g_s,t)=(1+g_{(1−p)},t)(1+g_E,t)(1+g_κ,t)/(1+g_Y,t)`；基准 `g_κ=0` | `ext_stage10_anchor_consistency` | 设计 §3.2 | 逐 t 逐省 |
| **M-26** | **双锚联立解（§3.2, D3-4/D3-5）** | `g_E*(g_Y)=0.9717642(1+g_Y)−1`；`g_Y*(g_E)=1.0290566(1+g_E)−1` | `ext_stage10_anchor_consistency` | 设计 §3.2 | 中档解 1.5494% |
| **M-27** | **一致性残差（§3.3, D3-6）** | `r_E=g_E^{scen}−g_E*(g_Y^{scen})`；判据 `\|r_E\|≤0.5pp` ⇒ 一致 | `ext_stage10_anchor_consistency` | 设计 §3.3 | 逐情景登记偏离 |
| **M-28** | **分式失效正向指标（§5.3, O6+）** | `F_t=#{dDdb≤0}/30`（按档分列） | `ext_stage6_O6_regularity`（扩） | T-12 §三 | 2022 复现 1/30 |

---

## 二、口径对象 → 报告轴映射

| 对象 | 数学定义 | 归属轴 | 代码 | 禁并规则 |
|:---|:---|:---|:---|:---|
| 识别区间 `I_t` | `{f_t(c):c∈W*}` | 轴1（标度识别不确定性, HC-1） | `ext_stage5_report.interval_by_enumeration` | 禁与轴2 合并（P-4） |
| 合法性受限区间 `I_t^leg` | `{f_t(c):c∈W*,Λ_t=1}` | 轴1 次对象 | 同上 | 不替代 `I_t` |
| C 双路径带 | `η_b·\|C_p1−C_p3\|/C` 精确代入 | 轴2（数据不确定性, O2） | `ext_stage4_cband` | 禁入轴1 |
| K 口径带 | 三档 g_K 链接 → 精确代入 | 轴2 | `ext_stage3_gk_band` / `ext_stage9_prospect_link` | 禁入轴1；禁 η_k 乘推（P-10） |
| Y 修订/代用带 | 全国代分省偏差（前望段） | 轴2 | `ext_stage9_prospect_link` | 禁入轴1 |
| O4 O5 O6 标记 | `D_t / φ_t / B_t / F_t` | 审计注 | `ext_stage5_report` / `ext_stage6_O6_regularity` | 披露/降级标记，不改区间 |
| **D3 锚一致性残差** | **`r_E`（M-27）** | **审计注（非轴）** | **`ext_stage10_anchor_consistency`** | **披露偏离，不改区间** |
| R_widen 判据量 | `半宽_t/半宽_ref` | 报告层判据 | `ext_stage8_widen_ratio` | 尺度不变、不含 naive_err；与 R_half 并列 |

---

## 三、只读来源（路径动态引用，禁复制）

| 来源 | 用途 | 方式 |
|:---|:---|:---|
| `deliver/06_复现归档/outputs/C_LP_v1.json` | 冻结 15 参数 | `json.load` 只读 |
| `deliver/04_数据/原始数据_*.xlsx` | Y/L/E/C/K_raw 面板 | pandas 只读；基准冻结 |
| `deliver3/03_实验记录/T09_output.json` | **前望段起点（2025 归一化值）** | json 只读，逐位承接 |
| `deliver3/03_实验记录/T10_output.json` / `T10_output_corr.json` | 路径丙 r、路径甲窗分位 | json 只读 |
| `deliver3/03_实验记录/T11_output.json` / `T11b_output.json` | 三层权威归属 | json 只读 |
| `deliver3/01_勘探产出/ddf_mac_ext_001_scout_路线M_2026-2030情景依据_v1.md` | P1–P8 情景参数 | 只读 |
| `deliver3/01_勘探产出/ddf_mac_ext_001_scout_路线J_K高档宏观参照_v1.md` | K 高档反证 | 只读 |
| `deliver3/04_批判审计/ddf_mac_ext_001_主智能体_终裁决定书_v1.md` | O6 处置/红线 | 只读 |

> **铁律**：导入上游模块须 `PYTHONDONTWRITEBYTECODE=1 python3 -B`，禁止在只读目录生成 `__pycache__` 或任何文件。

---

## 四、前望段接口 / 自洽性 / 判据映射

| 项 | 依赖 | 接口 | 说明 |
|:---|:---|:---|:---|
| 前望段链接 | M-8/M-19~M-22 | `ext_stage9_prospect_link` | 起点=T-09 2025；Y/L/E/C 全国代分省；K 走三层 |
| 情景参数 | M-20 | `ext_scenario_params.load` | P1–P8 → g_Y/g_L/g_E/g_s/g_I；P8 不入链 |
| 自洽约束 | M-23~M-27 | `ext_stage10_anchor_consistency` | 双锚联立 + 逐情景残差 + 偏离登记 |
| O6 处置 | M-16/M-28 | `ext_stage6_O6_regularity` | 定义保留；新增退化披露 + `F_t` 正向指标 |
| 判据 | M-18 | `ext_stage8_widen_ratio` | R_widen（1.5）+ R_half（3.0, retained_failed）并列 |

**映射完整性**：M-1~M-28 共 28 条（回望段 18 条沿用 v4，前望段/自洽性新增 10 条 M-19~M-28）。前望段映射可先用 2023–2025 数据干跑自检（以前望段模块跑回望段，应与 T-09 逐位一致）。

---

## 五、版本与快照登记

- 本文件 **v5**：在 v4（M-1~M-18）基础上新增前望段与自洽性映射（M-19~M-28）。
- v4 快照 `math_code_alignment_ddf_mac_ext_001_v4.md`（8702 B）**保留未改**；v1/v2 及无版本版保留未改。
- 号位冲突登记：既有工作版契约文件名无版本后缀但内部 `version="v5"`；本 v5 为 **2026–2030 情景外推代次**（对象区隔见设计文档 §0.1）。如实登记，未静默修正。
