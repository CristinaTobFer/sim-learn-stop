class Path:
    def __init__(self, instance, path_id):
        self.instance = instance
        self.path_id = path_id
        self.points = [instance.start_point]
        self.score = 0.0
        self.time = 0

    def copy(self):
        new_path = Path(self.instance, self.path_id)
        new_path.points = [point for point in self.points]
        new_path.score = self.score
        new_path.time = self.time
        return new_path

    def add_point(self, point):
        self.score += point["score"]
        self.time += self.instance.get_deterministic_time(self.points[-1], point)
        self.points.append(point)

    def is_feasible(self, tmax):
        return (
            self.time <= tmax
            and self.points[0] == self.instance.start_point
            and self.points[-1] == self.instance.end_point
        )


class Solution:
    """
    Clase principal para la solución del Team Oriented VRP
    """

    def __init__(self, instance):
        self.instance = instance
        self.paths = [
            Path(instance, i) for i in range(instance.m)
        ]  # Lista de listas con los puntos entregados en cada ruta
        self.unassigned_points = [
            point for point in instance.points[1:-1]
        ]  # Puntos no asignados a ninguna ruta
        self.total_score = 0.0
        self.total_time = 0.0
        self.feasible = True

        self.score = 0.0

    def copy(self):
        new_solution = Solution(self.instance)
        new_solution.paths = [path.copy() for path in self.paths]
        new_solution.unassigned_points = [point for point in self.unassigned_points]
        new_solution.total_score = self.total_score
        new_solution.total_time = self.total_time
        new_solution.feasible = self.feasible
        new_solution.score = self.score
        return new_solution

    def add_path(self, path_id: int, path: Path):
        self.paths[path_id] = path
        self.unassigned_points = [
            point for point in self.unassigned_points if point not in path.points
        ]
        self._update_solution_metrics()

    def _update_solution_metrics(self):
        """Actualiza las métricas de la solución completa"""
        self.total_score = sum(path.score for path in self.paths)
        self.total_time = sum(path.time for path in self.paths)

        # Verificar factibilidad
        self.feasible = all(path.is_feasible(self.instance.tmax) for path in self.paths)

    def get_score(self):
        return self.total_score

    def print_solution(self):
        """Imprime la solución de forma legible"""
        print("=== INSTANCE TEAM ORIENTED VRP ===")
        print(f"Número de vehículos: {self.instance.m}")
        print(f"Número de puntos: {self.instance.n - 2}")
        print(f"Tiempo máximo: {self.instance.tmax}")
        print()
        print("=== SOLUCIÓN TEAM ORIENTED VRP ===")
        print(f"Score total: {self.total_score:.2f}")
        print(f"Tiempo total: {self.total_time:.2f}")
        print(f"Factible: {self.feasible}")
        print(f"Puntos no asignados: {len(self.unassigned_points)}")
        print()

        for path_id, path in enumerate(self.paths):
            print(f"Ruta {path_id}:")
            print(f"  Secuencia: {[p['index'] for p in path.points]}")
            print(f"  Score de la ruta: {path.score:.2f}")
            print(f"  Tiempo de la ruta: {path.time:.2f}")

            print()
