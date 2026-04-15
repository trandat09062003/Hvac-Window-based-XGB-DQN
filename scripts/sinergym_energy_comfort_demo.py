#!/usr/bin/env python3
"""
Một pipeline mô phỏng: Sinergym (Gymnasium) + EnergyPlus.

Trước khi import sinergym, script đổi CWD sang thư mục `sinergym_runs/`
trong repo để mọi thư mục `*-resN` và `episode-*` nằm gọn, không làm bẩn gốc project.
"""

from __future__ import annotations

import os
import sys

# --------------------------------------------------------------------------- #
# Phải chạy TRƯỚC khi `import sinergym` — Sinergym lấy CWD một lần lúc import.
# --------------------------------------------------------------------------- #
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
_RUN_ROOT = os.path.join(_REPO_ROOT, "sinergym_runs")
os.makedirs(_RUN_ROOT, exist_ok=True)
os.chdir(_RUN_ROOT)
os.environ.setdefault("TQDM_DISABLE", "1")

import argparse
import shutil
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

import sinergym  # noqa: E402 — sau chdir

# Giảm log ASCII-art của Sinergym (vẫn giữ WARNING/ERROR)
for _name in ("ENVIRONMENT", "MODEL", "SIMULATOR", "REWARD"):
    try:
        sinergym.set_logger_level(_name, "WARNING")
    except Exception:
        pass


VI_EXPLANATION = """
================================================================================
GIẢI THÍCH MÔ PHỎNG (đọc một lần cho báo cáo)
================================================================================

1) MÔI TRƯỜNG (RL / Gymnasium)
   - Sinergym cung cấp env kiểu Gymnasium: reset() → quan sát đầu episode;
     step(action) → quan sát tiếp theo, reward, info.
   - Phía sau là EnergyPlus chạy song song qua Python API: mỗi bước tương ứng
     một (hoặc vài) bước thời gian mô phỏng tòa nhà (timestep có trong log MODEL).

2) NĂNG LƯỢNG
   - Biến HVAC_electricity_demand_rate: công suất điện HVAC tức thời (W), lấy từ
     Output:Variable của EnergyPlus (Facility Total HVAC Electricity Demand Rate).
   - Tổng kWh in cuối script ≈ Σ (P_W × Δt) với Δt lấy từ timesteps_per_hour của
     env (ví dụ 1 step = 1 h hoặc 15 phút tùy cấu hình).

3) TIÊU CHUẨN / TIỆN NGHI
   a) LinearReward (trong Sinergym): phạt khi nhiệt độ phòng ra khỏi dải mùa đông
      hoặc mùa hè (range_comfort_winter / range_comfort_summer trong YAML).
      Đây là tiêu chí vận hành thực dụng, gần thói quen setpoint / dải ASHRAE.
   b) PMV & PPD (ISO 7730): tính thêm bằng pythermalcomfort từ T_air, RH với
      giả định đơn giản (tr ≈ ta, vr cố định, met/clo theo mùa). Đây là chỉ số
      “cảm giác nhiệt” chuẩn học thuật; muốn khớp sát mô hình E+ cần đồng bộ
      MET, CLO, MRT từ mô hình người / bức xạ trong IDF.

4) FILE XUẤT RA (sau mỗi lần chạy)
   - Thư mục làm việc: sinergym_runs/<env_name>-resN/episode-M/output/
   - eplusout.eso / eplusout.err / eplustbl.htm … là output chuẩn EnergyPlus;
     env_config.pyyaml trong *-resN là snapshot cấu hình env.

5) LỆNH GỢI Ý
   .venv/bin/python scripts/sinergym_energy_comfort_demo.py --max-steps 48
================================================================================
""".strip()


def _ensure_eplus_path() -> None:
    if os.environ.get("EPLUS_PATH"):
        return
    exe = shutil.which("energyplus")
    candidates: List[str] = []
    if exe:
        candidates.append(os.path.dirname(os.path.realpath(exe)))
    candidates.extend(
        [
            "/usr/local/EnergyPlus-25-2-0",
            "/usr/local/EnergyPlus-24-2-0",
            "/opt/EnergyPlus-25-2-0",
        ]
    )
    for root in candidates:
        idd = os.path.join(root, "Energy+.idd")
        if os.path.isfile(idd):
            os.environ["EPLUS_PATH"] = root
            return
    raise RuntimeError(
        "Chưa đặt EPLUS_PATH và không tìm thấy Energy+.idd. "
        "Ví dụ: export EPLUS_PATH=/usr/local/EnergyPlus-25-2-0"
    )


def _obs_dict(env: gym.Env, obs_vec: np.ndarray) -> Dict[str, float]:
    raw = env.unwrapped
    names: List[str] = list(raw.observation_variables)
    return dict(zip(names, np.asarray(obs_vec, dtype=np.float64)))


def _default_action(env: gym.Env) -> np.ndarray:
    space = env.action_space
    if isinstance(space, gym.spaces.Box):
        low = np.asarray(space.low, dtype=np.float32)
        high = np.asarray(space.high, dtype=np.float32)
        return ((low + high) / 2.0).astype(np.float32)
    raise TypeError("Cần action_space Box cho demo này.")


def _pmv_ppd(
    t_air_c: float,
    rh_pct: float,
    month: float,
) -> Optional[Tuple[float, float]]:
    try:
        from pythermalcomfort.models import pmv_ppd_iso
    except ImportError:
        return None
    met = 1.1
    clo = 0.5 if 6 <= month <= 9 else 1.0
    tr = t_air_c
    vr = 0.1
    out = pmv_ppd_iso(tdb=t_air_c, tr=tr, vr=vr, rh=rh_pct, met=met, clo=clo)
    return float(out["pmv"]), float(out["ppd"])


def _step_hours(env: gym.Env) -> float:
    step_h = 1.0
    bc = env.unwrapped.building_config
    if isinstance(bc, dict):
        tph = bc.get("timesteps_per_hour")
        if tph:
            step_h = 1.0 / float(tph)
    return step_h


def run_episode(
    env_id: str,
    max_steps: int,
    seed: int,
    *,
    verbose_steps: bool,
    print_theory: bool,
) -> int:
    _ensure_eplus_path()

    if print_theory:
        print(VI_EXPLANATION)
        print()

    print(f"[Thư mục làm việc Sinergym] {os.getcwd()}")
    print(f"[Repo] {_REPO_ROOT}")
    print()

    env = gym.make(env_id)
    obs, info = env.reset(seed=seed)
    action = _default_action(env)

    raw = env.unwrapped
    print(f"Env ID        : {env_id}")
    print(f"Env name      : {raw.name}")
    print(f"Quan sát ({len(raw.observation_variables)}): {raw.observation_variables}")
    print(f"Hành động     : {env.action_space}  (demo dùng điểm giữa Box — không học RL)")
    if isinstance(raw.building_config, dict):
        print(f"building_config: {raw.building_config}")
    print(f"Δt (giờ/step) : {_step_hours(env)}")
    if verbose_steps:
        print("---")

    rows = 0
    total_hvac_wh = 0.0
    pmv_vals: List[float] = []
    ppd_vals: List[float] = []

    while rows < max_steps:
        obs, reward, terminated, truncated, info = env.step(action)
        od = _obs_dict(env, obs)

        month = od.get("month", 1.0)
        t_air = od.get("air_temperature", float("nan"))
        rh = od.get("air_humidity", float("nan"))
        p_hvac = od.get("HVAC_electricity_demand_rate", float("nan"))
        t_out = od.get("outdoor_temperature", float("nan"))

        step_h = _step_hours(env)
        if not np.isnan(p_hvac):
            total_hvac_wh += float(p_hvac) * step_h

        pmv_pp = _pmv_ppd(t_air, rh, month)
        if pmv_pp is not None and not np.isnan(t_air):
            pmv_vals.append(pmv_pp[0])
            ppd_vals.append(pmv_pp[1])

        if verbose_steps:
            line = (
                f"t={info.get('timestep', rows + 1):3d} | "
                f"T_out={t_out:5.1f}°C T_air={t_air:5.2f}°C RH={rh:4.1f}% | "
                f"P_HVAC={p_hvac:8.1f} W | r={reward:10.5f}"
            )
            if pmv_pp is not None:
                line += f" | PMV={pmv_pp[0]:+.2f} PPD={pmv_pp[1]:.1f}%"
            print(line)
            et = info.get("energy_term", "")
            ct = info.get("comfort_term", "")
            cp = info.get("comfort_penalty", "")
            if et != "" or ct != "" or cp != "":
                print(f"         → energy_term={et} | comfort_term={ct} | comfort_penalty={cp}")

        rows += 1
        if terminated or truncated:
            break

    workspace = getattr(raw.model, "workspace_path", None)
    episode_path = getattr(raw.model, "episode_path", None)
    env.close()

    kwh = total_hvac_wh / 1000.0
    print("---")
    print(f"Đã chạy       : {rows} bước")
    print(f"Tổng HVAC     : ~{kwh:.4f} kWh (Σ P[W]×Δt[h])")
    if pmv_vals:
        import statistics

        print(
            f"PMV (trung bình ± lệch chuẩn): {statistics.mean(pmv_vals):+.3f} ± {statistics.pstdev(pmv_vals):.3f}"
        )
        print(
            f"PPD (trung bình ± lệch chuẩn): {statistics.mean(ppd_vals):.2f}% ± {statistics.pstdev(ppd_vals):.2f}%"
        )
    else:
        print("PMV/PPD: cài pythermalcomfort để có thống kê (pip install pythermalcomfort)")

    out_dir = os.path.join(episode_path, "output") if episode_path else None
    print(f"Output E+     : {out_dir or '(không xác định)'}")
    print(f"Workspace     : {workspace}")

    summary_path = os.path.join(_RUN_ROOT, "last_run_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"env_id={env_id}\nsteps={rows}\nkwh_hvac_approx={kwh:.6f}\n")
        f.write(f"workspace={workspace}\nepisode={episode_path}\n")
        f.write(f"eplus_output={out_dir}\n")
    print(f"Đã ghi tóm tắt: {summary_path}")

    return 0


def main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(
        description="Mô phỏng Sinergym + EnergyPlus: môi trường, năng lượng, tiện nghi."
    )
    p.add_argument("--env", default="Eplus-demo-v1")
    p.add_argument("--max-steps", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--no-theory",
        action="store_true",
        help="Không in khối giải thích tiếng Việt đầu chương trình.",
    )
    p.add_argument(
        "--quiet-steps",
        action="store_true",
        help="Chỉ in tóm tắt cuối, không in từng bước.",
    )
    p.add_argument(
        "--progress",
        action="store_true",
        help="Bật thanh tqdm (mặc định tắt: TQDM_DISABLE=1).",
    )
    args = p.parse_args(argv)

    if args.progress:
        os.environ.pop("TQDM_DISABLE", None)

    try:
        return run_episode(
            args.env,
            args.max_steps,
            args.seed,
            verbose_steps=not args.quiet_steps,
            print_theory=not args.no_theory,
        )
    except Exception as e:
        print(f"Lỗi: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    # argv[0] là đường dẫn script; argparse dùng argv[1:]
    raise SystemExit(main(sys.argv[1:]))
