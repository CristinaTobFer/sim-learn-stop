from Solution import Solution, Path
from typing import Optional, List
import logging
import random
import time

logging.basicConfig(level=logging.INFO)


class Algorithm:
    def __init__(self, instance):
        self.instance = instance

    def build_solution(
        self,
        max_iterations: int = 10000000,
        time_limit: float = None,
        alpha: float = 0.3,
        seed: int = 42,
    ) -> Solution:
        """
        Construye una solución usando diferentes estrategias constructivas.

        Args:
            max_iterations: Número máximo de iteraciones
            time_limit: Tiempo límite en segundos (None = sin límite)
        """
        random.seed(seed)
        best_solution = None
        start_time = time.time()

        for iteration in range(max_iterations):
            # Verificar tiempo límite
            if time_limit is not None:
                elapsed_time = time.time() - start_time
                if elapsed_time >= time_limit:
                    logging.info(
                        "Time limit reached (%.2f s) after %d iterations",
                        elapsed_time,
                        iteration,
                    )
                    break
            if elapsed_time % 10 == 0:
                logging.info(
                    "Constructing grasp score solution iteration %s and elapsed time %s",
                    iteration,
                    elapsed_time,
                )
            solution = self._construct_grasp_solution(alpha=alpha)
            if iteration % 10 == 0:
                logging.info(
                    "Local search swap points iteration %s and elapsed time %s",
                    iteration,
                    elapsed_time,
                )
            solution = self._local_search_swap_points(solution, 100)

            self.compute_score(solution)

            # Agregar solución a la lista
            if (
                best_solution is None
                or solution.get_score() > best_solution.get_score()
            ):
                best_solution = solution

        best_solution.n_iterations = iteration + 1
        total_time = time.time() - start_time
        logging.info(
            "Build solution completed in %.2f s with %d iterations",
            total_time,
            iteration + 1,
        )

        return best_solution

    # GRASP #########################################################################################
    def _create_grasp_route(
        self,
        path_id: int,
        solution: Solution,
        available_points: List,
        alpha: float = 0.3,
    ) -> Optional[Path]:
        """
        Crea una ruta usando GRASP con RCL.
        alpha controla el tamaño de la RCL:
            - alpha=0 -> greedy puro
            - alpha=1 -> elección aleatoria total
        """
        route = Path(self.instance, path_id)
        current_point = self.instance.start_point

        while available_points:
            candidates = []

            for point in available_points:
                time_to_point = self.instance.get_deterministic_time(
                    current_point, point
                )
                time_to_end = self.instance.get_deterministic_time(
                    point, self.instance.end_point
                )

                if route.time + time_to_point + time_to_end <= self.instance.tmax:
                    value = point["score"]
                    candidates.append((point, value))

            if not candidates:
                break

            # Ordenar candidatos por valor
            candidates.sort(key=lambda x: x[1], reverse=True)
            best_value = candidates[0][1]
            worst_value = candidates[-1][1]

            # Definir umbral para la RCL
            threshold = best_value - alpha * (best_value - worst_value)
            rcl = [p for p, v in candidates if v >= threshold]

            # Selección aleatoria desde RCL
            chosen_point = random.choice(rcl)

            # Añadir punto a la ruta
            route.add_point(chosen_point)
            current_point = chosen_point
            available_points.remove(chosen_point)

        # Cerrar ruta
        route.add_point(self.instance.end_point)
        return route if len(route.points) > 2 else None

    def _construct_grasp_solution(
        self, iterations: int = 10, alpha: float = 0.3
    ) -> Solution:
        """Construye una solución mediante GRASP"""
        best_solution = None

        for _ in range(iterations):
            solution = Solution(self.instance)
            available_points = sorted(
                solution.unassigned_points, key=lambda x: x["score"], reverse=True
            )

            for path_id in range(self.instance.m):
                route = self._create_grasp_route(
                    path_id, solution, available_points, alpha
                )
                if route:
                    solution.add_path(path_id, route)

            if (
                best_solution is None
                or solution.total_score > best_solution.total_score
            ):
                best_solution = solution

        return best_solution

    # Búsquedas Locales #############################################################################

    def _local_search_swap_points(
        self, solution: Solution, max_iterations: int = 100
    ) -> Solution:
        """Realiza búsqueda local intercambiando puntos entre rutas y puntos no asignados"""
        best_solution = solution
        best_score = solution.get_score()

        for _ in range(max_iterations):
            improved = False

            # Intentar intercambiar puntos de las rutas con puntos no asignados
            for path_id, path in enumerate(best_solution.paths):
                if len(path.points) <= 2:  # Solo ruta con inicio y fin
                    continue

                for point_idx in range(1, len(path.points) - 1):  # Excluir inicio y fin
                    point_in_route = path.points[point_idx]

                    for unassigned_point in best_solution.unassigned_points:
                        if unassigned_point["score"] < point_in_route["score"]:
                            continue

                        # Crear una copia de la solución para probar el intercambio
                        new_solution = self._swap_points(
                            best_solution, path_id, point_idx, unassigned_point
                        )

                        if new_solution and new_solution.feasible:
                            new_score = self.compute_score(new_solution)

                            if new_score > best_score:
                                best_solution = new_solution
                                best_score = new_score
                                improved = True
                                break

                    if improved:
                        break

                if improved:
                    break

            # Si no se encontró mejora, intentar intercambiar entre rutas
            if not improved:
                for path1_id in range(len(best_solution.paths)):
                    for path2_id in range(path1_id + 1, len(best_solution.paths)):
                        path1 = best_solution.paths[path1_id]
                        path2 = best_solution.paths[path2_id]

                        if len(path1.points) <= 2 or len(path2.points) <= 2:
                            continue

                        for point1_idx in range(1, len(path1.points) - 1):
                            for point2_idx in range(1, len(path2.points) - 1):
                                new_solution = self._swap_between_paths(
                                    best_solution,
                                    path1_id,
                                    point1_idx,
                                    path2_id,
                                    point2_idx,
                                )

                                if new_solution and new_solution.feasible:
                                    new_score = self.compute_score(new_solution)

                                    if new_score > best_score:
                                        best_solution = new_solution
                                        best_score = new_score
                                        improved = True
                                        break

                            if improved:
                                break

                        if improved:
                            break

                    if improved:
                        break

            # Si no se encontró ninguna mejora, terminar
            if not improved:
                break

        return best_solution

    def _swap_points(
        self, solution: Solution, path_id: int, point_idx: int, new_point
    ) -> Optional[Solution]:
        """Intercambia un punto de una ruta con un punto no asignado"""
        try:
            # Crear una copia profunda de la solución
            new_solution = solution.copy()
            path = new_solution.paths[path_id]
            old_point = path.points[point_idx]

            # Remover el punto de la ruta
            path.points.pop(point_idx)
            path.score -= old_point["score"]

            # Recalcular tiempo de la ruta
            path.time = 0
            for i in range(len(path.points) - 1):
                path.time += self.instance.get_deterministic_time(
                    path.points[i], path.points[i + 1]
                )

            # Insertar el nuevo punto en la mejor posición
            best_position = -1
            best_time_increase = float("inf")

            for pos in range(1, len(path.points)):
                # Calcular el aumento en tiempo si se inserta en esta posición
                prev_point = path.points[pos - 1]
                next_point = path.points[pos]

                original_time = self.instance.get_deterministic_time(
                    prev_point, next_point
                )
                new_time = self.instance.get_deterministic_time(
                    prev_point, new_point
                ) + self.instance.get_deterministic_time(new_point, next_point)

                time_increase = new_time - original_time

                # Verificar factibilidad
                if (
                    path.time
                    + time_increase
                    + self.instance.get_deterministic_time(
                        new_point, self.instance.end_point
                    )
                    <= self.instance.tmax
                ):
                    if time_increase < best_time_increase:
                        best_time_increase = time_increase
                        best_position = pos

            if best_position != -1:
                # Insertar el nuevo punto
                path.points.insert(best_position, new_point)
                path.score += new_point["score"]
                path.time += best_time_increase

                # Actualizar puntos no asignados
                new_solution.unassigned_points.remove(new_point)
                new_solution.unassigned_points.append(old_point)

                # Actualizar métricas de la solución
                new_solution._update_solution_metrics()

            return new_solution

        except Exception as e:
            print(f"Error en _swap_points: {e}")

        return None

    def _swap_between_paths(
        self,
        solution: Solution,
        path1_id: int,
        point1_idx: int,
        path2_id: int,
        point2_idx: int,
    ) -> Optional[Solution]:
        """Intercambia puntos entre dos rutas diferentes"""
        try:
            # Crear una copia profunda de la solución
            new_solution = solution.copy()
            path1 = new_solution.paths[path1_id]
            path2 = new_solution.paths[path2_id]

            point1 = path1.points[point1_idx]
            point2 = path2.points[point2_idx]

            # Remover puntos de ambas rutas
            path1.points.pop(point1_idx)
            path2.points.pop(point2_idx)

            path1.score -= point1["score"]
            path2.score -= point2["score"]

            # Recalcular tiempos de ambas rutas
            for path in [path1, path2]:
                path.time = 0
                for i in range(len(path.points) - 1):
                    path.time += self.instance.get_deterministic_time(
                        path.points[i], path.points[i + 1]
                    )

            # Insertar point2 en path1 en la mejor posición
            best_position1 = -1
            best_time_increase1 = float("inf")

            for pos in range(1, len(path1.points)):
                prev_point = path1.points[pos - 1]
                next_point = path1.points[pos]

                original_time = self.instance.get_deterministic_time(
                    prev_point, next_point
                )
                new_time = self.instance.get_deterministic_time(
                    prev_point, point2
                ) + self.instance.get_deterministic_time(point2, next_point)

                time_increase = new_time - original_time

                if (
                    path1.time
                    + time_increase
                    + self.instance.get_deterministic_time(
                        point2, self.instance.end_point
                    )
                    <= self.instance.tmax
                ):
                    if time_increase < best_time_increase1:
                        best_time_increase1 = time_increase
                        best_position1 = pos

            # Insertar point1 en path2 en la mejor posición
            best_position2 = -1
            best_time_increase2 = float("inf")

            for pos in range(1, len(path2.points)):
                prev_point = path2.points[pos - 1]
                next_point = path2.points[pos]

                original_time = self.instance.get_deterministic_time(
                    prev_point, next_point
                )
                new_time = self.instance.get_deterministic_time(
                    prev_point, point1
                ) + self.instance.get_deterministic_time(point1, next_point)

                time_increase = new_time - original_time

                if (
                    path2.time
                    + time_increase
                    + self.instance.get_deterministic_time(
                        point1, self.instance.end_point
                    )
                    <= self.instance.tmax
                ):
                    if time_increase < best_time_increase2:
                        best_time_increase2 = time_increase
                        best_position2 = pos

            # Si ambos puntos pueden ser insertados, realizar el intercambio
            if best_position1 != -1 and best_position2 != -1:
                # Insertar point2 en path1
                path1.points.insert(best_position1, point2)
                path1.score += point2["score"]
                path1.time += best_time_increase1

                # Insertar point1 en path2
                path2.points.insert(best_position2, point1)
                path2.score += point1["score"]
                path2.time += best_time_increase2

                # Actualizar métricas de la solución
                new_solution._update_solution_metrics()

                return new_solution

        except Exception as e:
            print(f"Error en _swap_between_paths: {e}")

        return None
