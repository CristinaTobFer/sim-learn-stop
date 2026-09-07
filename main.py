from Instance import Instance
from Algorithms.SimHeuristicAlgorithm import SimHeuristicAlgorithm
from Algorithms.SimLearnHeuristicAlgorithm import SimLearnHeuristicAlgorithm
from Algorithms.MetaheuristicAlgorithm import MetaheuristicAlgorithm
import csv
import os
import time
import random

if __name__ == "__main__":
    instances_paths = [
        "original_instances/Set_21_234/p2.3.g",
        "original_instances/Set_21_234/p2.3.k",
        "original_instances/Set_32_234/p1.3.k",
        "original_instances/Set_32_234/p1.4.b",
        "original_instances/Set_33_234/p3.2.i",
        "original_instances/Set_33_234/p3.2.j",
        "original_instances/Set_64_234/p6.4.a",
        "original_instances/Set_64_234/p6.4.n",
        "original_instances/Set_100_234/p4.3.r",
        "original_instances/Set_100_234/p4.3.i",
        "original_instances/Set_102_234/p7.2.j",
        "original_instances/Set_102_234/p7.3.j",
    ]

    rows_metrics = [
        [
            "Instance",
            "Scenario",
            "Algorithm",
            "Score Simulation",
            "Score Validation",
            "Number of Iterations",
        ]
    ]

    scenarios = [
        {
            "time_of_day": "night",
            "day_of_week": "weekend",
            "weather": "dry",
            "accidents": "no",
            "special_events": "no",
            "season": "normal",
            "roadworks": "no",
        },
        {
            "time_of_day": "night",
            "day_of_week": "workday",
            "weather": "rain",
            "accidents": "no",
            "special_events": "no",
            "season": "normal",
            "roadworks": "no",
        },
        {
            "time_of_day": "peak",
            "day_of_week": "workday",
            "weather": "dry",
            "accidents": "no",
            "special_events": "no",
            "season": "normal",
            "roadworks": "no",
        },
        {
            "time_of_day": "peak",
            "day_of_week": "workday",
            "weather": "rain",
            "accidents": "no",
            "special_events": "no",
            "season": "normal",
            "roadworks": "no",
        },
        {
            "time_of_day": "peak",
            "day_of_week": "workday",
            "weather": "rain",
            "accidents": "no",
            "special_events": "no",
            "season": "normal",
            "roadworks": "yes",
        },
    ]

    alpha = 0.9964
    max_depth = 7
    learning_rate = 0.168
    max_iter = 264
    l2_regularization = 0.9441
    time_limit = 5 * 60
    seed = 54
    random.seed(seed)

    for instance_path in instances_paths:
        # Get all subdirectories (folders) in the instance path
        print(f"Processing instance: {instance_path}")
        print("=========================================")
        file_path = instance_path
        print(f"File path: {file_path}")
        # Create Instance object
        instance = Instance(file_path)
        print("========== METAHEURISTIC ALGORITHM ==========")
        print("================================================")
        algorithm_metaheuristic = MetaheuristicAlgorithm(instance)
        solutions_metaheuristic = algorithm_metaheuristic.build_solution(
            alpha=alpha, time_limit=time_limit
        )
        solutions_metaheuristic.print_solution()

        print("========== SIM HEURISTIC ALGORITHM ==========")
        print("=========================================")
        algorithm_sim_heuristic = SimHeuristicAlgorithm(instance)
        solutions_sim_heuristic = algorithm_sim_heuristic.build_solution(
            alpha=alpha, time_limit=time_limit
        )
        solutions_sim_heuristic.print_solution()

        for scenario in scenarios:
            print(f"Scenario: {scenario}")
            instance.config_current_instance(scenario)
            print("========== SIM LEARN HEURISTIC ALGORITHM ==========")
            print("================================================")
            start_time = time.time()
            algorithm_sim_learn_heuristic = SimLearnHeuristicAlgorithm(
                instance,
                max_depth=max_depth,
                learning_rate=learning_rate,
                max_iter=max_iter,
                l2_regularization=l2_regularization,
                seed=seed,
            )
            print(f"Predicting values: {time.time() - start_time} seconds")
            solutions_sim_learn_heuristic = (
                algorithm_sim_learn_heuristic.build_solution(
                    alpha=alpha, time_limit=time_limit
                )
            )
            solutions_sim_learn_heuristic.print_solution()

            H_SCORE = instance.compute_validation_score(solutions_metaheuristic)
            SH_SCORE = instance.compute_validation_score(solutions_sim_heuristic)
            SLH_SCORE = instance.compute_validation_score(solutions_sim_learn_heuristic)
            print(f"H_SCORE: {H_SCORE}, SH_SCORE: {SH_SCORE}, SLH_SCORE: {SLH_SCORE}")
            rows_metrics.append(
                [
                    instance.name,
                    scenario,
                    "Metaheuristic",
                    solutions_metaheuristic.score,
                    H_SCORE,
                    solutions_metaheuristic.n_iterations,
                ]
            )
            rows_metrics.append(
                [
                    instance.name,
                    scenario,
                    "Sim Heuristic",
                    solutions_sim_heuristic.score,
                    SH_SCORE,
                    solutions_sim_heuristic.n_iterations,
                ]
            )
            rows_metrics.append(
                [
                    instance.name,
                    scenario,
                    "Sim Learn Heuristic",
                    solutions_sim_learn_heuristic.score,
                    SLH_SCORE,
                    solutions_sim_learn_heuristic.n_iterations,
                ]
            )

            # Write all metrics to CSV file with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            if not os.path.exists("metrics"):
                os.makedirs("metrics")
            with open(
                f"metrics/metrics_{timestamp}.csv", "w", newline="", encoding="utf-8"
            ) as f:
                writer = csv.writer(f, delimiter=";")
                # Format decimal numbers with commas
                formatted_rows = []
                for row in rows_metrics:
                    formatted_row = []
                    for cell in row:
                        if isinstance(cell, (int, float)):
                            # Format floats as decimals with comma as decimal separator
                            if isinstance(cell, float):
                                formatted_row.append(f"{cell:.6f}".replace(".", ","))
                            else:
                                formatted_row.append(cell)
                        else:
                            formatted_row.append(str(cell))
                    formatted_rows.append(formatted_row)
                writer.writerows(formatted_rows)

    # Write all metrics to CSV file with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    with open(f"final_metrics_{timestamp}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        # Format decimal numbers with commas
        formatted_rows = []
        for row in rows_metrics:
            formatted_row = []
            for cell in row:
                if isinstance(cell, (int, float)):
                    # Format floats as decimals with comma as decimal separator
                    if isinstance(cell, float):
                        formatted_row.append(f"{cell:.6f}".replace(".", ","))
                    else:
                        formatted_row.append(cell)
                else:
                    formatted_row.append(str(cell))
            formatted_rows.append(formatted_row)
        writer.writerows(formatted_rows)
