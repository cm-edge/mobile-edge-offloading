from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import heapq
import math


@dataclass
class Task:
    img_size_bits: float
    t_cpu: float
    t_gpu: float
    t_npu: float
    t_edge: float


@dataclass
class PowerParams:
    P_cpu: float
    P_gpu: float
    P_npu: float

    P_tx: float
    P_idle: float  # still kept (may be used elsewhere)

    P_tail: float
    tail_time: float

    P_edge: float
    alpha: float

    B_up: float


def precompute_base_profiles(tasks: List[Task], params: PowerParams) -> List[Dict[str, Dict[str, float]]]:
    """
    Base profiles:
      - cpu/gpu/npu: base_energy = P_mode * duration
      - offload: base_energy = E_up (tx during upload) + E_wait (tail during edge) + E_edge (optional remote energy)
                where E_wait uses P_tail (NOT P_idle) to avoid double-counting tail later.
    """
    profiles = []
    for t in tasks:
        cpu = {"duration": t.t_cpu, "base_energy": params.P_cpu * t.t_cpu}
        gpu = {"duration": t.t_gpu, "base_energy": params.P_gpu * t.t_gpu}
        npu = {"duration": t.t_npu, "base_energy": params.P_npu * t.t_npu}

        t_up = t.img_size_bits / params.B_up
        E_up = params.P_tx * t_up

        t_edge = t.t_edge

        # Edge-side energy (keep as in your original model; set P_edge=0 if you don't want to count it)
        E_edge = params.P_edge * t_edge

        # During edge execution, you want to charge at tail power (radio kept active)
        E_wait = params.P_tail * t_edge

        off = {
            "duration": t_up + t_edge,
            "base_energy": E_up + E_wait + E_edge,
        }

        profiles.append({"cpu": cpu, "gpu": gpu, "npu": npu, "offload": off})

    return profiles


def round_value(x: float, step: Optional[float]) -> float:
    if step is None or step <= 0:
        return x
    return round(x / step) * step


def solve_energy_min_with_tail(
    tasks: List[Task],
    params: PowerParams,
    T_max: float,
    time_round_step: Optional[float] = 1e-3,
    tail_round_step: Optional[float] = 1e-3,
) -> Tuple[float, float, List[str]]:
    """
    Tail model you requested:
      - Offload: edge-wait energy already uses P_tail, so do NOT add extra tail energy for the offload task itself.
                But tail_rem is still refreshed to tail_time after an offload.
      - Local compute (cpu/gpu/npu): if tail is active, energy in the overlap window is max(P_mode, P_tail)*overlap.
                Since base_energy already counts P_mode*duration, we add only the positive difference:
                   extra = max(0, P_tail - P_mode) * overlap
    """
    n = len(tasks)
    profiles = precompute_base_profiles(tasks, params)
    MODES = ("cpu", "gpu", "npu", "offload")

    start = (0, 0.0, 0.0)  # (task_index, time_used, tail_remaining)
    dist: Dict[Tuple[int, float, float], float] = {start: 0.0}
    prev: Dict[Tuple[int, float, float], Tuple[Optional[Tuple], Optional[str]]] = {start: (None, None)}
    heap: List[Tuple[float, Tuple[int, float, float]]] = [(0.0, start)]

    best_final_state: Optional[Tuple[int, float, float]] = None
    best_final_energy: float = math.inf

    mode_power = {"cpu": params.P_cpu, "gpu": params.P_gpu, "npu": params.P_npu}

    while heap:
        E, state = heapq.heappop(heap)
        if E > dist.get(state, math.inf) + 1e-12:
            continue

        i, t_used, tail_rem = state

        if i == n:
            if E < best_final_energy - 1e-12:
                best_final_energy = E
                best_final_state = state
            continue

        for mode in MODES:
            duration = profiles[i][mode]["duration"]
            base_energy = profiles[i][mode]["base_energy"]

            t2 = t_used + duration
            if t2 > T_max + 1e-12:
                continue

            # How much of this task overlaps with remaining tail time?
            tail_overlap = min(tail_rem, duration)

            # Tail energy handling
            if mode == "offload":
                # Already charged P_tail during edge-wait inside base_energy, so no extra tail add-on here
                extra_tail_energy = 0.0
            else:
                P_mode = mode_power[mode]
                # overlap energy should be max(P_mode, P_tail)*overlap
                # base_energy already includes P_mode*duration => add only the positive diff over overlap
                extra_tail_energy = max(0.0, params.P_tail - P_mode) * tail_overlap

            # Update remaining tail after spending 'duration' seconds
            tail2 = max(0.0, tail_rem - duration)
            if mode == "offload":
                # Offload refreshes tail timer
                tail2 = params.tail_time

            # Rounding for state-space control
            t2 = round_value(t2, time_round_step)
            tail2 = round_value(tail2, tail_round_step)

            new_state = (i + 1, t2, tail2)
            E2 = E + base_energy + extra_tail_energy

            if E2 < dist.get(new_state, math.inf) - 1e-12:
                dist[new_state] = E2
                prev[new_state] = (state, mode)
                heapq.heappush(heap, (E2, new_state))

    if best_final_state is None:
        raise RuntimeError("No feasible solution under the given T_max")

    schedule: List[str] = [None] * n
    cur = best_final_state
    for idx in range(n, 0, -1):
        p, mode = prev[cur]
        schedule[idx - 1] = mode
        cur = p

    total_time = best_final_state[1]
    return best_final_energy, total_time, schedule


# --- Demo ---
if __name__ == "__main__":
    tasks = [
        Task(img_size_bits=1024 * 1024 * 8, t_cpu=0.21, t_gpu=0.78, t_npu=0.20, t_edge=0.108),
        Task(img_size_bits=1024 * 1024 * 8, t_cpu=0.21, t_gpu=0.78, t_npu=0.20, t_edge=0.108),
        Task(img_size_bits=1024 * 1024 * 8, t_cpu=0.23, t_gpu=0.82, t_npu=0.24, t_edge=0.183),
        Task(img_size_bits=1024 * 1024 * 8, t_cpu=0.23, t_gpu=0.82, t_npu=0.24, t_edge=0.183),
        Task(img_size_bits=1024 * 1024 * 8, t_cpu=0.27, t_gpu=0.83, t_npu=0.26, t_edge=0.378),
        Task(img_size_bits=1024 * 1024 * 8, t_cpu=0.27, t_gpu=0.83, t_npu=0.26, t_edge=0.378),
        Task(img_size_bits=1024 * 1024 * 8, t_cpu=0.29, t_gpu=0.97, t_npu=0.28, t_edge=0.599),
        Task(img_size_bits=1024 * 1024 * 8, t_cpu=0.29, t_gpu=0.97, t_npu=0.28, t_edge=0.599),
        Task(img_size_bits=1024 * 1024 * 8, t_cpu=0.29, t_gpu=0.97, t_npu=0.28, t_edge=0.599),
        Task(img_size_bits=1024 * 1024 * 8, t_cpu=0.29, t_gpu=0.97, t_npu=0.28, t_edge=0.599),
        Task(img_size_bits=1024 * 1024 * 8, t_cpu=0.21, t_gpu=0.78, t_npu=0.20, t_edge=0.108),
        Task(img_size_bits=1024 * 1024 * 8, t_cpu=0.21, t_gpu=0.78, t_npu=0.20, t_edge=0.108),
        Task(img_size_bits=1024 * 1024 * 8, t_cpu=0.23, t_gpu=0.82, t_npu=0.24, t_edge=0.183),
        Task(img_size_bits=1024 * 1024 * 8, t_cpu=0.23, t_gpu=0.82, t_npu=0.24, t_edge=0.183),
        Task(img_size_bits=1024 * 1024 * 8, t_cpu=0.27, t_gpu=0.83, t_npu=0.26, t_edge=0.378),
        Task(img_size_bits=1024 * 1024 * 8, t_cpu=0.27, t_gpu=0.83, t_npu=0.26, t_edge=0.378),
        Task(img_size_bits=1024 * 1024 * 8, t_cpu=0.29, t_gpu=0.97, t_npu=0.29, t_edge=0.599),
        Task(img_size_bits=1024 * 1024 * 8, t_cpu=0.29, t_gpu=0.97, t_npu=0.29, t_edge=0.599),
        Task(img_size_bits=1024 * 1024 * 8, t_cpu=0.29, t_gpu=0.97, t_npu=0.29, t_edge=0.599),
        Task(img_size_bits=1024 * 1024 * 8, t_cpu=0.29, t_gpu=0.97, t_npu=0.29, t_edge=0.599),
    ]

    params = PowerParams(
        P_cpu=2.56,
        P_gpu=2.46,
        P_npu=2.6,
        P_tx=1.8,
        P_idle=1.2,
        P_tail=1.4,
        tail_time=10,
        P_edge=0.8,
        alpha=1,
        B_up=5*1024*1024*1024,
    )

    T_max = 7
    E_min, T_used, schedule = solve_energy_min_with_tail(tasks, params, T_max)

    print("Min energy (J):", E_min)
    print("Total time (s):", T_used)
    print("Schedule:", schedule)
