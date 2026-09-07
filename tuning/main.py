import sys
import os
import io
import random
import logging
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict


def _parse_args(argv) -> tuple[str, int, Dict[str, str]]:
    instance_path = None
    seed = 1
    params: Dict[str, str] = {}

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg.lower().endswith((".txt", ".dat")) and os.path.isfile(arg):
            instance_path = arg
            i += 1
            continue
        if arg in ("--seed", "-seed") and i + 1 < len(argv):
            try:
                seed = int(float(argv[i + 1]))
            except ValueError:
                seed = 1
            i += 2
            continue
        if arg.startswith("--"):
            key = arg[2:].strip()  # irace puede enviar "--alpha " con espacio
            # Handle --param VALUE
            if key and i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                params[key] = argv[i + 1]
                i += 2
                continue
            # Try to split key into name+value if no space was used (--paramVALUE)
            for name in (
                "alpha",
                "max_depth",
                "learning_rate",
                "max_iter",
                "l2_regularization",
            ):
                if key.startswith(name) and len(key) > len(name):
                    params[name] = key[len(name) :].strip()
                    break
            i += 1
            continue
        if "=" in arg:
            key, value = arg.split("=", 1)
            if key:
                params[key.lstrip("-").strip()] = value.strip()
        i += 1

    if instance_path is None:
        raise SystemExit("No instance path provided to target runner.")

    return instance_path, seed, params


def _as_float(params: Dict[str, str], key: str, default: float) -> float:
    value = params.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _as_int(params: Dict[str, str], key: str, default: int) -> int:
    value = params.get(key)
    if value is None:
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def main():
    instance_path, seed, params = _parse_args(sys.argv)

    with open(instance_path, "r") as file:
        lines = file.readlines()
        file_name = lines[0].strip()

    file_path = "original_instances/" + file_name

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # Silence logging
    logging.disable(logging.CRITICAL)
    cost = 0  # default en caso de error
    random.seed(seed)
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    with redirect_stdout(buf_out), redirect_stderr(buf_err):
        from Algorithms.SimLearnHeuristicAlgorithm import SimLearnHeuristicAlgorithm
        from Instance import Instance

        # Valores por defecto si irace no envía el parámetro (evitar None)
        alpha = _as_float(params, "alpha", 0.5)
        max_depth = _as_int(params, "max_depth", 5)
        learning_rate = _as_float(params, "learning_rate", 0.05)
        max_iter = _as_int(params, "max_iter", 300)
        l2_regularization = _as_float(params, "l2_regularization", 1.0)
        instance = Instance(file_path)

        if len(instance.list_actual_covariables) > 0:
            index = random.randint(0, len(instance.list_actual_covariables) - 1)
            instance.config_current_instance(index)
            algorithm_metaheuristic = SimLearnHeuristicAlgorithm(
                instance,
                max_depth=max_depth,
                learning_rate=learning_rate,
                max_iter=max_iter,
                l2_regularization=l2_regularization,
                seed=seed,
            )
            solutions_metaheuristic = algorithm_metaheuristic.build_solution(
                time_limit=30, alpha=alpha, seed=seed
            )
            score = solutions_metaheuristic.get_score()
            cost = -score
    # print(cost)
    sys.stdout.write(f"{cost}\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
