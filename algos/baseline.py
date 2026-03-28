from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import heapq
import math


# =========================
# Data models
# =========================

@dataclass
class Task:
    img_size_bits: float
    t_cpu: float
    t_edge: float


@dataclass
class PowerParams:
    P_cpu: float

    P_tx: float
    P_tail: float
    tail_time: float

    P_edge: float   # keep if you want to count edge energy
    B_up: float


# =========================
# Base profiles
# =========================

def precompute_base_profiles(
    tasks: List[Task],
    params: PowerParams
) -> List[Dict[str, Dict[str, float]]]:
    """
    Base energy definitions:

    CPU:
      E = P_cpu * t_cpu

    Offload:
      - upload:   P_tx   * t_up
      - edge wait (radio active): P_tail * t_edge
      - edge compute (optional):  P_edge * t_edge
    """
    profiles = []

    for t in tasks:
        # CPU
        cpu = {
            "duration": t.t_cpu,
            "base_energy": params.P_cpu * t.t_cpu
        }

        # Offload
        t_up = t.img_size_bits / params.B_up
        E_up = params.P_tx * t_up

        E_wait = params.P_tail * t.t_edge
        E_edge = params.P_edge * t.t_edge

        offload = {
            "duration": t_up + t.t_edge,
            "base_energy": E_up + E_wait + E_edge
        }

        profiles.append({
            "cpu": cpu,
            "offload": offload
        })

    return profiles


def round_value(x: float, step: Optional[float]) -> float:
    if step is None or step <= 0:
        return x
    return round(x / step) * step


# =========================
# Solver
# =========================

def solve_energy_min_with_tail(
    tasks: List[Task],
    params: PowerParams,
    T_max: float,
    time_round_step: Optional[float] = 1e-3,
    tail_round_step: Optional[float] = 1e-3,
) -> Tuple[float, float, List[str]]:

    profiles = precompute_base_profiles(tasks, params)
    n = len(tasks)
    MODES = ("cpu", "offload")

    # state = (task_index, time_used, tail_remaining)
    start = (0, 0.0, 0.0)
    dist = {start: 0.0}
    prev = {start: (None, None)}

    heap = [(0.0, start)]

    best_final_state = None
    best_final_energy = math.inf

    while heap:
        E, state = heapq.heappop(heap)
        if E > dist.get(state, math.inf) + 1e-12:
            continue

        i, t_used, tail_rem = state

        if i == n:
            if E < best_final_energy:
                best_final_energy = E
                best_final_state = state
            continue

        for mode in MODES:
            duration = profiles[i][mode]["duration"]
            base_energy = profiles[i][mode]["base_energy"]

            t2 = t_used + duration
            if t2 > T_max + 1e-12:
                continue

            tail_overlap = min(tail_rem, duration)

            # -------- energy accounting --------
            if mode == "cpu":
                # overlap energy = max(P_cpu, P_tail)
                extra_tail_energy = max(
                    0.0, params.P_tail - params.P_cpu
                ) * tail_overlap
            else:
                # offload already paid P_tail during edge wait
                extra_tail_energy = 0.0

            E2 = E + base_energy + extra_tail_energy

            # -------- tail evolution --------
            tail2 = max(0.0, tail_rem - duration)
            if mode == "offload":
                tail2 = params.tail_time

            t2 = round_value(t2, time_round_step)
            tail2 = round_value(tail2, tail_round_step)

            new_state = (i + 1, t2, tail2)

            if E2 < dist.get(new_state, math.inf) - 1e-12:
                dist[new_state] = E2
                prev[new_state] = (state, mode)
                heapq.heappush(heap, (E2, new_state))

    if best_final_state is None:
        raise RuntimeError("No feasible solution")

    # reconstruct schedule
    schedule = [None] * n
    cur = best_final_state
    for idx in range(n, 0, -1):
        p, mode = prev[cur]
        schedule[idx - 1] = mode
        cur = p

    return best_final_energy, best_final_state[1], schedule


# =========================
# Demo
# =========================

if __name__ == "__main__":
    tasks = [
        Task(img_size_bits=1024 * 1024 * 8, t_cpu=0.21, t_edge=0.108),
        Task(img_size_bits=1024 * 1024 * 8, t_cpu=0.21, t_edge=0.108),
        Task(img_size_bits=1024 * 1024 * 8, t_cpu=0.23, t_edge=0.183),
        Task(img_size_bits=1024 * 1024 * 8, t_cpu=0.23, t_edge=0.183),
        Task(img_size_bits=1024 * 1024 * 8, t_cpu=0.27, t_edge=0.378),
        Task(img_size_bits=1024 * 1024 * 8, t_cpu=0.27, t_edge=0.378),
        Task(img_size_bits=1024 * 1024 * 8, t_cpu=0.29, t_edge=0.599),
        Task(img_size_bits=1024 * 1024 * 8, t_cpu=0.29, t_edge=0.599),
        Task(img_size_bits=1024 * 1024 * 8, t_cpu=0.29, t_edge=0.599),
        Task(img_size_bits=1024 * 1024 * 8, t_cpu=0.29, t_edge=0.599),
        Task(img_size_bits=1024 * 1024 * 8, t_cpu=0.21, t_edge=0.108),
        Task(img_size_bits=1024 * 1024 * 8, t_cpu=0.21, t_edge=0.108),
        Task(img_size_bits=1024 * 1024 * 8, t_cpu=0.23, t_edge=0.183),
        Task(img_size_bits=1024 * 1024 * 8, t_cpu=0.23, t_edge=0.183),
        Task(img_size_bits=1024 * 1024 * 8, t_cpu=0.27, t_edge=0.378),
        Task(img_size_bits=1024 * 1024 * 8, t_cpu=0.27, t_edge=0.378),
        Task(img_size_bits=1024 * 1024 * 8, t_cpu=0.29, t_edge=0.599),
        Task(img_size_bits=1024 * 1024 * 8, t_cpu=0.29, t_edge=0.599),
        Task(img_size_bits=1024 * 1024 * 8, t_cpu=0.29, t_edge=0.599),
        Task(img_size_bits=1024 * 1024 * 8, t_cpu=0.29, t_edge=0.599),
    ]

    params = PowerParams(
        P_cpu=2.56,
        P_tx=1.8,
        P_tail=1.4,
        tail_time=10,
        P_edge=0.8,
        B_up=5*1024*1024*1024,
    )

    T_max = 7
    E_min, T_used, schedule = solve_energy_min_with_tail(tasks, params, T_max)

    print("Min energy (J):", E_min)
    print("Total time (s):", T_used)
    print("Schedule:", schedule)
