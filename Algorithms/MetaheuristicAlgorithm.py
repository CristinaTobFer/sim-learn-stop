from Algorithms.Algorithm import Algorithm
from Solution import Solution


class MetaheuristicAlgorithm(Algorithm):
    def __init__(self, instance):
        super().__init__(instance)

        deterministic_time_matrix = self.instance.compute_mean_deterministic_time()
        self.instance.set_deterministic_time(deterministic_time_matrix)

    def compute_score(self, solution: Solution):
        score = 0
        end_point = solution.instance.end_point
        for path in solution.paths:
            last_point = solution.instance.start_point
            route_time = 0
            for point in path.points[1:]:
                path.time += self.instance.get_deterministic_time(last_point, point)
                if (
                    route_time
                    + self.instance.get_deterministic_time(last_point, point)
                    + self.instance.get_deterministic_time(point, end_point)
                    <= self.instance.tmax
                ):
                    route_time += self.instance.get_deterministic_time(
                        last_point, point
                    )
                    score += point["score"]
                else:
                    break
                last_point = point
            route_time += self.instance.get_deterministic_time(last_point, end_point)

        solution.score = score
        return solution.score
