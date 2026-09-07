from Algorithms.Algorithm import Algorithm
from Solution import Solution
import numpy as np


class SimLearnHeuristicAlgorithm(Algorithm):
    def __init__(
        self,
        instance,
        max_depth=5,
        learning_rate=0.05,
        max_iter=300,
        l2_regularization=1.0,
        seed=42,
    ):
        super().__init__(instance)

        deterministic_time_matrix = self.instance.compute_predicted_deterministic_time(
            max_depth=max_depth,
            learning_rate=learning_rate,
            max_iter=max_iter,
            l2_regularization=l2_regularization,
            seed=seed,
        )
        self.instance.set_deterministic_time(deterministic_time_matrix)

        self.instance.set_simulation_time_matrices(
            self.instance.simulation_time_matrices_simlearnheuristic
        )

    def compute_score(self, solution: Solution):
        scores = []
        for matrix in solution.instance.simulation_time_matrices:
            score = 0
            for path in solution.paths:
                last_point = solution.instance.start_point
                route_time = 0
                for point in path.points[1:]:
                    if (
                        route_time
                        + matrix[last_point["index"]][point["index"]]
                        + matrix[point["index"]][solution.instance.end_point["index"]]
                        <= self.instance.tmax
                    ):
                        route_time += matrix[last_point["index"]][point["index"]]
                        score += point["score"]
                    else:
                        break
                    last_point = point
                route_time += matrix[last_point["index"]][
                    solution.instance.end_point["index"]
                ]
            scores.append(score)

        solution.score = np.mean(scores)
        return solution.score
