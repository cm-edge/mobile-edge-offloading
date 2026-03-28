from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from ortools.linear_solver import pywraplp
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

def precompute_base_profiles(tasks: List[Task], params: PowerParams) -> List[Dict[str, Dict[str, float]]]:
    """
    Base profiles:
      - cpu/gpu/npu: base_energy = P_mode * duration
      - offload: base_energy = E_up (tx during upload) + E_wait (tail during edge) + E_edge
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
        E_edge = params.P_edge * t_edge
        E_wait = params.P_tail * t_edge

        off = {
            "duration": t_up + t_edge,
            "base_energy": E_up + E_wait + E_edge,
        }
        profiles.append({"cpu": cpu, "gpu": gpu, "npu": npu, "offload": off})
    return profiles


def solve_energy_min_milp_with_tail(
    tasks: List[Task],
    params: PowerParams,
    T_max: float,
    solver_name: str = "CBC",
) -> Tuple[float, float, List[str]]:
    profiles = precompute_base_profiles(tasks, params)
    n = len(tasks)
    MODES = ("cpu", "gpu", "npu", "offload")

    solver = pywraplp.Solver.CreateSolver(solver_name)
    if solver is None:
        raise RuntimeError(f"Could not create MILP solver '{solver_name}'. Try solver_name='CBC'.")

    INF = solver.infinity()

    d = {(i, m): profiles[i][m]["duration"] for i in range(n) for m in MODES}
    ebase = {(i, m): profiles[i][m]["base_energy"] for i in range(n) for m in MODES}

    delta_power = {
        "cpu": max(0.0, params.P_tail - params.P_cpu),
        "gpu": max(0.0, params.P_tail - params.P_gpu),
        "npu": max(0.0, params.P_tail - params.P_npu),
        "offload": 0.0,
    }

    max_d = max(d.values()) if n > 0 else 0.0
    M_min = max(params.tail_time, max_d, 1e-9)     # for min(y_i, d_i)
    M_prod = max(params.tail_time, 1e-9)           # for w = o_i * x
    M_reset = params.tail_time                     # for tail reset disjunction

    # Decision variables
    x = {(i, m): solver.BoolVar(f"x_{i}_{m}") for i in range(n) for m in MODES}

    # Realized duration per task
    d_i = {i: solver.NumVar(0.0, INF, f"d_{i}") for i in range(n)}

    # Tail remaining at start of task i (i=0..n)
    y = {i: solver.NumVar(0.0, params.tail_time, f"y_{i}") for i in range(n + 1)}

    # Overlap o_i = min(y_i, d_i)
    o = {i: solver.NumVar(0.0, params.tail_time, f"o_{i}") for i in range(n)}
    b = {i: solver.BoolVar(f"b_{i}") for i in range(n)}  # min selector

    # Product linearization w[i,m] = o_i * x[i,m]
    w = {(i, m): solver.NumVar(0.0, params.tail_time, f"w_{i}_{m}") for i in range(n) for m in MODES}

    # --- Constraints ---

    # y_0 = 0
    solver.Add(y[0] == 0.0)

    # Exactly one mode per task
    for i in range(n):
        solver.Add(sum(x[i, m] for m in MODES) == 1)

    # d_i definition
    for i in range(n):
        solver.Add(d_i[i] == sum(d[i, m] * x[i, m] for m in MODES))

    # Total time constraint
    solver.Add(sum(d_i[i] for i in range(n)) <= T_max)

    # o_i = min(y_i, d_i) via big-M
    for i in range(n):
        solver.Add(o[i] <= y[i])
        solver.Add(o[i] <= d_i[i])
        solver.Add(o[i] >= y[i] - M_min * (1 - b[i]))
        solver.Add(o[i] >= d_i[i] - M_min * b[i])

    # Tail evolution with RESET on offload:
    # if x_offload=0 -> y_{i+1} = y_i - o_i
    # if x_offload=1 -> y_{i+1} = tail_time
    for i in range(n):
        xoff = x[i, "offload"]

        # Case: not offload => y_{i+1} = y_i - o_i
        solver.Add(y[i + 1] <= (y[i] - o[i]) + M_reset * xoff)
        solver.Add(y[i + 1] >= (y[i] - o[i]) - M_reset * xoff)

        # Case: offload => y_{i+1} = tail_time
        solver.Add(y[i + 1] <= params.tail_time + M_reset * (1 - xoff))
        solver.Add(y[i + 1] >= params.tail_time - M_reset * (1 - xoff))

    # Product linearization w[i,m] = o_i * x[i,m]
    for i in range(n):
        for m in MODES:
            solver.Add(w[i, m] <= o[i])
            solver.Add(w[i, m] <= M_prod * x[i, m])
            solver.Add(w[i, m] >= o[i] - M_prod * (1 - x[i, m]))

    # --- Objective ---
    objective = solver.Objective()
    objective.SetMinimization()

    for i in range(n):
        for m in MODES:
            objective.SetCoefficient(x[i, m], ebase[i, m])
            objective.SetCoefficient(w[i, m], delta_power[m])

    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        # Helpful debug info
        st = {solver.OPTIMAL: "OPTIMAL",
              solver.FEASIBLE: "FEASIBLE",
              solver.INFEASIBLE: "INFEASIBLE",
              solver.UNBOUNDED: "UNBOUNDED",
              solver.ABNORMAL: "ABNORMAL",
              solver.NOT_SOLVED: "NOT_SOLVED"}.get(status, str(status))
        raise RuntimeError(f"MILP solver status: {st} (no optimal solution).")

    # Extract schedule
    schedule: List[str] = []
    for i in range(n):
        chosen = None
        for m in MODES:
            if x[i, m].solution_value() > 0.5:
                chosen = m
                break
        if chosen is None:
            raise RuntimeError(f"Internal error: no mode selected for task {i}.")
        schedule.append(chosen)

    total_time = sum(d_i[i].solution_value() for i in range(n))
    total_energy = objective.Value()
    return total_energy, total_time, schedule


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
        B_up=5 * 1024 * 1024 * 1024,
    )

    T_max = 7
    """E_min, T_used, schedule = solve_energy_min_with_tail(tasks, params, T_max)

    print("Min energy (J):", E_min)
    print("Total time (s):", T_used)
    print("Schedule:", schedule)"""

    E_min, T_used, schedule = solve_energy_min_milp_with_tail(tasks, params, T_max)
    print("MILP Min energy (J):", E_min)
    print("MILP Total time (s):", T_used)
    print("MILP Schedule:", schedule)
