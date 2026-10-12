# -*- coding: utf-8 -*-
"""s28 —— deliver5（ddf_mac_ext_001 2023–2030 MAC 外推）离线复现核验

用途
----
deliver5 的 T-09 / T-14 脚本写死了 Linux 绝对路径 `/home/user/1699367/...`。
本脚本在本机建一个**镜像目录**（口径不变、只重写路径常量），把两支脚本原样跑一遍，
再与 `deliver5/04_复现归档/` 的归档 JSON 做逐字段数值比对，回答一个问题：

    deliver5 的 2023–2030 MAC 数字，能不能在本地独立重算出来？

纪律
----
- **只读** deliver5：不修改、不新建、不删除其任何文件（含 __pycache__）。
- 全部产物落在 `02_产物/` 与本脚本同目录的 `_s28_deliver5_mirror/`。
- 运行 T-09/T-14 时置 `PYTHONDONTWRITEBYTECODE=1`，避免污染只读目录。

输出
----
  02_产物/s28_deliver5_复现核验.json
  02_产物/s28_deliver5_复现核验.csv
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # .../长期碳排放权价格跃升保险/01_引擎
PROJ = HERE.parent                              # .../长期碳排放权价格跃升保险
ROOT = PROJ.parent                              # .../碳排放权保险
PAPER = ROOT / "背景材料与论文" / "中国省际碳排放影子价格测算与碳修正绿色GDP核算——基于二次型方向距离函数的实证研究（1995–2022）"
D5 = PAPER / "deliver5"
OUT = PROJ / "02_产物"
MIRROR = HERE / "_s28_deliver5_mirror"

LINUX_ROOT = "/home/user/1699367"
IGNORE_KEYS = {"elapsed_s", "platform", "env", "input_hashes", "python",
               "numpy", "scipy", "pandas", "cpu_count"}


# ---------------------------------------------------------------- 镜像构建
def build_mirror() -> Path:
    """建镜像：只拷必需输入，并重写脚本内的 Linux 绝对路径前缀。"""
    if MIRROR.exists():
        shutil.rmtree(MIRROR)
    d3 = MIRROR / "deliver3" / "03_实验记录"
    d3.mkdir(parents=True)
    (MIRROR / "deliver" / "04_数据").mkdir(parents=True)
    (MIRROR / "deliver" / "06_复现归档" / "outputs").mkdir(parents=True)
    des = MIRROR / "deliver3" / "02_设计方案"
    des.mkdir(parents=True)

    deliver = PAPER / "deliver"
    shutil.copy2(deliver / "04_数据" / "原始数据_修正LGDP_Kbase_WJP_Lofficial_v1.xlsx",
                 MIRROR / "deliver" / "04_数据" / "原始数据_修正LGDP_Kbase_WJP_Lofficial_v1.xlsx")
    shutil.copy2(deliver / "06_复现归档" / "outputs" / "C_LP_v1.json",
                 MIRROR / "deliver" / "06_复现归档" / "outputs" / "C_LP_v1.json")

    contract = D5 / "08_设计与契约"
    pairs = [
        # (镜像目标名, 源文件)
        ("validation_contract_ddf_mac_ext_001.json",
         contract / "validation_contract_ddf_mac_ext_001.json"),
        ("validation_contract_ddf_mac_ext_001_v5.json",
         contract / "validation_contract_ddf_mac_ext_001_v5.json"),
        ("ddf_mac_ext_001_forge_外推方案设计_v5.md",
         contract / "ddf_mac_ext_001_forge_外推方案设计_v5.md"),
        ("ddf_mac_ext_001_forge_2026-2030情景外推设计_v1.md",
         contract / "ddf_mac_ext_001_forge_2026-2030情景外推设计_v1.md"),
        ("parameters_handover_ddf_mac_ext_001.csv",
         contract / "parameters_handover_ddf_mac_ext_001.csv"),
        ("parameters_handover_ddf_mac_ext_001_v5.csv",
         contract / "parameters_handover_ddf_mac_ext_001_v5.csv"),
        ("math_code_alignment_ddf_mac_ext_001_v5.md",
         contract / "math_code_alignment_ddf_mac_ext_001_v5.md"),
    ]
    for dst, src in pairs:
        if not src.exists():                      # 裸名版在 历史版本/ 下
            alt = contract / "历史版本" / dst
            assert alt.exists(), f"缺源文件：{src} / {alt}"
            src = alt
        shutil.copy2(src, des / dst)

    arch = D5 / "04_复现归档"
    for name in ["ddf_mac_ext_001_tester_T08_外推与回测_v1.md",
                 "ddf_mac_ext_001_tester_T13_勘误登记_v1.md",
                 "T10_output.json", "T10_output_corr.json",
                 "T11b_output.json", "T12_output.json"]:
        shutil.copy2(arch / name, d3 / name)

    code = D5 / "03_代码"
    for name in ["run_T09_extrapolation.py", "run_T14_scenario_extrapolation.py"]:
        txt = (code / name).read_text(encoding="utf-8")
        assert LINUX_ROOT in txt, f"{name} 内未找到 {LINUX_ROOT}"
        txt = txt.replace(LINUX_ROOT, MIRROR.as_posix())
        (d3 / name).write_text(txt, encoding="utf-8")

    # 归档 T09_output.json：不放入镜像（避免被本次重跑覆盖而误作"原件"），
    # 本次运行由 T-09 自行生成。
    return MIRROR


# ---------------------------------------------------------------- 运行
def run_script(name: str) -> dict:
    d3 = MIRROR / "deliver3" / "03_实验记录"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run([sys.executable, "-B", name], cwd=str(d3), env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    return dict(returncode=p.returncode, stdout=p.stdout[-4000:], stderr=p.stderr[-4000:])


# ---------------------------------------------------------------- 数值比对
def walk_diff(a, b, path="", out=None):
    """递归比对两个 JSON 结构，返回 (最大相对偏差, 偏差>tol 的条目清单, 比对项数)。"""
    if out is None:
        out = []
    if isinstance(a, dict) and isinstance(b, dict):
        n = 0
        worst = 0.0
        for k in sorted(set(a) | set(b)):
            if k in IGNORE_KEYS:
                continue
            if k not in a or k not in b:
                out.append((f"{path}/{k}", "KEY_MISSING", None, None))
                continue
            w, nsub = walk_diff(a[k], b[k], f"{path}/{k}", out)
            worst = max(worst, w)
            n += nsub
        return worst, n
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((path, "LEN_DIFF", len(a), len(b)))
            return 0.0, 0
        worst = 0.0
        n = 0
        for i, (x, y) in enumerate(zip(a, b)):
            w, nsub = walk_diff(x, y, f"{path}[{i}]", out)
            worst = max(worst, w)
            n += nsub
        return worst, n
    if isinstance(a, bool) or isinstance(b, bool):
        if a != b:
            out.append((path, "BOOL_DIFF", a, b))
        return 0.0, 1
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a == b:
            return 0.0, 1
        denom = max(abs(a), abs(b), 1e-30)
        rel = abs(a - b) / denom
        if rel > 1e-9:
            out.append((path, "NUM_DIFF", a, b))
        return rel, 1
    if a != b:
        out.append((path, "STR_DIFF", a, b))
    return 0.0, 1


def main() -> int:
    print("[1/4] 建镜像 …")
    build_mirror()
    print("      mirror =", MIRROR)

    print("[2/4] 运行 T-09（2023–2025 真外推）…")
    r09 = run_script("run_T09_extrapolation.py")
    print("      rc =", r09["returncode"])
    if r09["returncode"] != 0:
        print(r09["stderr"])
        return 2

    print("[3/4] 运行 T-14（2026–2030 情景外推）…")
    r14 = run_script("run_T14_scenario_extrapolation.py")
    print("      rc =", r14["returncode"])
    if r14["returncode"] != 0:
        print(r14["stderr"])
        return 3

    print("[4/4] 与归档 JSON 比对 …")
    d3 = MIRROR / "deliver3" / "03_实验记录"
    arch = D5 / "04_复现归档"
    report = {"tasks": {}}
    for tag, new_f, old_f in [
        ("T09", d3 / "T09_output.json", arch / "T09_output.json"),
        ("T14", d3 / "T14_output.json", arch / "T14_output.json"),
    ]:
        newj = json.loads(new_f.read_text(encoding="utf-8"))
        oldj = json.loads(old_f.read_text(encoding="utf-8"))
        diffs: list = []
        worst, n = walk_diff(oldj, newj, "", diffs)
        report["tasks"][tag] = dict(
            n_compared=n, max_rel_diff=worst, n_diff_gt_tol=len(diffs),
            diffs=diffs[:80],
            new_bytes=os.path.getsize(new_f), old_bytes=os.path.getsize(old_f))
        print(f"      {tag}: 比对 {n} 项，最大相对偏差 {worst:.3e}，超 1e-9 项 {len(diffs)}")
        for d in diffs[:15]:
            print("        ·", d)

    # 关键锚点单列（便于人工复核）
    newj = json.loads((d3 / "T09_output.json").read_text(encoding="utf-8"))
    report["key_anchors"] = dict(
        MAC_2022=newj["anchor_check"].get("MAC_2022"),
        SigmaD=newj["anchor_check"].get("SigmaD"),
        I_2022=newj.get("I_2022"),
        R_widen_c0_2023_2025=newj.get("R_widen_c0"),
        R_widen_max_2023_2025=newj.get("R_widen_max"),
    )
    new14 = json.loads((d3 / "T14_output.json").read_text(encoding="utf-8"))
    report["key_anchors"]["ref_halfwidth_enumerated"] = new14.get("ref_halfwidth_enumerated")
    report["key_anchors"]["start_check"] = new14.get("start_check")
    report["runtime"] = dict(t09=r09, t14=dict(returncode=r14["returncode"],
                                               stdout=r14["stdout"][-2000:]))
    report["env"] = dict(python=sys.version.split()[0],
                         mirror=str(MIRROR))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "s28_deliver5_复现核验.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    lines = ["任务,比对项数,最大相对偏差,超1e-9项数,重跑字节,归档字节"]
    for tag, v in report["tasks"].items():
        lines.append(f"{tag},{v['n_compared']},{v['max_rel_diff']:.3e},"
                     f"{v['n_diff_gt_tol']},{v['new_bytes']},{v['old_bytes']}")
    (OUT / "s28_deliver5_复现核验.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n关键锚点：")
    for k, v in report["key_anchors"].items():
        print("  ", k, "=", v)
    print("\nwritten", OUT / "s28_deliver5_复现核验.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
