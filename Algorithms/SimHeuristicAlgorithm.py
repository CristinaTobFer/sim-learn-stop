from Algorithms.Algorithm import Algorithm
from Solution import Solution
import numpy as np


class SimHeuristicAlgorithm(Algorithm):
    def __init__(self, instance):
        super().__init__(instance)

        deterministic_time_matrix = self.instance.compute_mean_deterministic_time()
        self.instance.set_deterministic_time(deterministic_time_matrix)

        self.instance.set_simulation_time_matrices(
            self.instance.simulation_time_matrices_simheuristic
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
