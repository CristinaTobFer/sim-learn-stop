import os
import numpy as np
import hdbscan
import scipy.stats as st
from sklearn.ensemble import HistGradientBoostingRegressor
from joblib import Parallel, delayed


class Instance:
    def __init__(
        self,
        path,
        n_clusters=50,
        compute_mahalanobis_distance=False,
        probability_of_occurrence=False,
        seed=100,
    ):
        """
        Initialize the ExtendInstance class with a file path.

        Args:
            path (str): Path to the text file to be read
        """
        self.path = path
        self.name = path.split("/")[-1]
        self.n = None  # number of vertices
        self.m = None  # number of paths
        self.tmax = None  # available time budget per path
        self.points = []  # list of points with coordinates and scores
        self.start_point = None  # starting point (first point)
        self.end_point = None  # ending point (last point)

        self.actual_covariables = {}
        self.historical_covariables = []
        self.historical_matrices = []

        self.assign_cluster_cardinality = None
        self.assign_cluster_distance_to_centroid = None
        self.n_clusters = n_clusters

        self.encoding = {
            "time_of_day": {"peak": 3, "valley": 1, "night": 0},
            "day_of_week": {"workday": 1, "weekend": 0},
            "weather": {"dry": 0, "rain": 2, "snow_fog": 3},
            "accidents": {"no": 0, "yes": 5},
            "special_events": {"no": 0, "yes": 3},
            "season": {"normal": 0, "holiday": 2},
            "roadworks": {"no": 0, "yes": 4},
        }

        self.simulation_time_matrices = None
        self.simulation_time_matrices_simheuristic = None
        self.simulation_time_matrices_simlearnheuristic = None

        self.validation_time_matrices = None

        self._read_file()
        print(f"Reading historical matrices for {self.path}")
        self._read_historical_matrices()
        if probability_of_occurrence:
            self.probability_of_occurrence = self.compute_probability_of_occurrence(
                self.actual_covariables
            )
        if compute_mahalanobis_distance:
            self.mahalanobis_distance = self.compute_Mahalanobis_distance()
        else:
            self.time_min_matrix = self.calculate_time_min_matrix()

        self.simulation_time_matrices_simheuristic = (
            self.compute_lognormal_simulation_time(n_scenarios=100)
        )

    def compute_probability_of_occurrence(self, actual_covariables):
        probability_of_occurrence = {
            "time_of_day": {"peak": 0.3, "valley": 0.4, "night": 0.3},
            "day_of_week": {"workday": 0.7, "weekend": 0.3},
            "weather": {"dry": 0.82, "rain": 0.16, "snow_fog": 0.02},
            "accidents": {"yes": 0.003, "no": 0.997},
            "special_events": {"yes": 0.01, "no": 0.99},
            "season": {"normal": 0.962, "holiday": 0.038},
            "roadworks": {"yes": 0.1, "no": 0.9},
        }
        probability = 1
        for key, value in actual_covariables.items():
            probability *= probability_of_occurrence[key][value]
        self.probability_of_occurrence = probability
        return probability

    def _read_file(self):
        """
        Read and parse the text file according to the specified format.
        """
        try:
            name = "/".join(self.path.split("/"))
            print(name)
            with open(name + ".txt", "r") as file:
                lines = file.readlines()

            # Remove empty lines and strip whitespace
            lines = [line.strip() for line in lines if line.strip()]

            # Read the first three lines with header data
            if len(lines) >= 3:
                # Parse n N (number of vertices)
                n_line = lines[0].split()
                if len(n_line) >= 2:
                    self.n = int(n_line[1])

                # Parse m P (number of paths)
                m_line = lines[1].split()
                if len(m_line) >= 2:
                    self.m = int(m_line[1])

                # Parse tmax Tmax (available time budget per path)
                tmax_line = lines[2].split()
                if len(tmax_line) >= 2:
                    self.tmax_original = 1.8 * float(
                        tmax_line[1]
                    )  # Changed to float since tmax can be decimal ==> # TODO

            # Read the remaining lines with point data
            for i in range(3, len(lines)):
                line = lines[i]
                # Skip comment lines
                if line.startswith("*"):
                    continue

                # Parse point data: x y S
                point_data = line.split()
                if len(point_data) >= 3:
                    x = float(point_data[0])
                    y = float(point_data[1])
                    score = float(point_data[2])

                    point = {"x": x, "y": y, "score": score, "index": len(self.points)}
                    self.points.append(point)

            # Set start and end points
            if self.points:
                self.start_point = self.points[0]
                self.end_point = self.points[-1]

        except FileNotFoundError:
            print(f"Error: File '{self.path}' not found.")
        except Exception as e:
            print(f"Error reading file: {e}")

    def compute_Mahalanobis_distance(self):
        """
        Calcula la distancia de Mahalanobis entre actual_covariables y la nube de puntos historical_covariables.

        La distancia de Mahalanobis mide cuántas desviaciones estándar está el punto actual
        respecto al centroide de la distribución histórica, teniendo en cuenta la correlación
        entre las variables.

        Returns:
            float: Distancia de Mahalanobis del punto actual respecto a la distribución histórica
        """
        print(f"Computing Mahalanobis distance for {self.path}")
        # Codificar las covariables actuales
        actual_encoded = self.encode(self.actual_covariables, self.encoding)
        print(f"Actual encoded: {actual_encoded}")

        # Codificar todas las covariables históricas
        historical_encoded = np.array(
            [self.encode(cov, self.encoding) for cov in self.historical_covariables]
        )
        # Calcular la media (centroide) de la distribución histórica
        mean_historical = np.mean(historical_encoded, axis=0)
        print(f"Mean historical: {mean_historical}")
        # Calcular la matriz de covarianza de los datos históricos
        cov_matrix = np.cov(historical_encoded, rowvar=False)
        # Manejar el caso de matriz singular añadiendo regularización
        try:
            # Intentar calcular la inversa de la matriz de covarianza
            cov_inv = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            # Si la matriz es singular, usar pseudoinversa
            print("Matrix is singular")
            cov_inv = np.linalg.pinv(cov_matrix)
        # Calcular la diferencia entre el punto actual y la media
        diff = actual_encoded - mean_historical
        # Calcular la distancia de Mahalanobis: sqrt((x - μ)^T * Σ^(-1) * (x - μ))
        mahalanobis_distance = np.sqrt(np.dot(np.dot(diff, cov_inv), diff))
        print(f"Mahalanobis distance: {mahalanobis_distance}")
        print("End of Mahalanobis distance")
        return mahalanobis_distance

    def _read_historical_matrices(self):
        name_instance = self.path.split("/")[-1].split(".txt")[0]
        folder = self.path.split("/")[:-1]
        folder = "/".join(folder)
        folder = os.path.join(folder, name_instance)
        files = os.listdir(folder)

        for file in files:
            matrix = [[0] * self.n for _ in range(self.n)]
            covariables = {}
            with open(os.path.join(folder, file), "r") as f:
                lines = f.readlines()
                i = 1
                for line in lines:
                    if ":" in line:
                        covariable_name, covariable_value = line.split(":")
                        covariables[covariable_name] = covariable_value.strip()
                    elif line == "\n":
                        continue
                    else:
                        matrix[i - 1] = [float(x) for x in line.split()]
                        i += 1

            self.tmax = self.tmax_original  # * (1+extra_T_max) #TODO
            self.historical_covariables.append(covariables)
            self.historical_matrices.append(matrix)

    def config_current_instance(self, actual_covariables):
        self.actual_covariables = actual_covariables
        encode_actual = self.encode(self.actual_covariables, self.encoding)
        print(f"Actual covariates: {self.actual_covariables}")
        print(f"Encode actual: {encode_actual}")
        self.tmax = self.tmax_original  # * (1+extra_T_max)

        # 1. Clusterizar
        self.clusters, self.centroids, self.labels = self.clusterize_matrices(
            self.historical_covariables, self.historical_matrices, self.encoding
        )
        self.clusters, self.labels, labels_actual = self.clusterize_matrices_hdbscan(
            self.historical_covariables, self.historical_matrices, self.encoding
        )

        # 2. Ajustar distribuciones
        self.distributions = self.fit_cluster_distributions(self.clusters)

        # 4. Generar 100 muestras de matrices para el primer label

        self.simulation_time_matrices_simlearnheuristic = self.sample_from_cluster(
            labels_actual[0], self.distributions, n_samples=100
        )

        self.validation_time_matrices = self.sample_from_cluster(
            labels_actual[0], self.distributions, n_samples=1000
        )

    def clusterize_matrices_hdbscan(
        self, historical_covariables, historical_matrices, mapping, min_cluster_size=10
    ):
        # Codificar todas las covariables con tu función
        encoded = np.array(
            [self.encode(cov, mapping) for cov in historical_covariables]
        )

        # Crear el clusterizador HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size, metric="manhattan", prediction_data=True
        )

        # Ajustar HDBSCAN y obtener labels
        labels = clusterer.fit_predict(encoded)

        # Inicializar clusters vacíos
        clusters = {}
        for label in np.unique(labels):
            clusters[label] = []

        # Agrupar matrices según labels
        for idx, label in enumerate(labels):
            clusters[label].append(historical_matrices[idx])
        self.n_clusters = (
            len(np.unique(labels)) - 1
        )  # Quedan -1 porque -1 es el cluster de ruido

        # 3. Asignar nuevo cluster
        # labels_new = self.assign_cluster(self.actual_covariables, self.encoding, self.centroids, self.clusters)
        actual_encoded = np.array([self.encode(self.actual_covariables, self.encoding)])

        labels_actual, strengths = hdbscan.approximate_predict(
            clusterer, actual_encoded
        )

        if labels_actual[0] == -1:
            self.assign_cluster_cardinality = 1
        else:
            self.assign_cluster_cardinality = len(clusters[labels_actual[0]])
        self.assign_cluster_distance_to_centroid = strengths[0]

        return clusters, labels, labels_actual

    def clusterize_matrices(self, historical_covariables, historical_matrices, mapping):
        # Codificar todas las covariables con tu función
        encoded = np.array(
            [self.encode(cov, mapping) for cov in historical_covariables]
        )

        # Inicializar clusters vacíos
        clusters = {i: [] for i in range(self.n_clusters)}

        # Inicializar centroides aleatorios
        np.random.seed(0)
        centroids_idx = np.random.choice(len(encoded), self.n_clusters, replace=False)
        centroids = encoded[centroids_idx].astype(float)

        for _ in range(10):  # iteraciones simples de actualización tipo KMeans
            # asignar cada vector al centroide más cercano (Manhattan)
            labels = []
            for vec in encoded:
                distances = np.sum(np.abs(centroids - vec), axis=1)
                labels.append(np.argmin(distances))
            labels = np.array(labels)

            # actualizar centroides
            for i in range(self.n_clusters):
                if np.any(labels == i):
                    centroids[i] = np.mean(encoded[labels == i], axis=0)

        # agrupar matrices
        for idx, label in enumerate(labels):
            clusters[label].append(historical_matrices[idx])

        return clusters, centroids, labels

    def assign_cluster(self, new_covariables, mapping, centroids, clusters):
        """
        Asigna cada vector de nuevas covariables a su cluster más cercano.

        Args:
            new_covariables: lista con las nuevas covariables a evaluar.
            mapping: diccionario para codificación (igual al usado en clusterize_matrices).
            centroids: centroides calculados por clusterize_matrices.

        Returns:
            labels: lista con el índice de cluster asignado a cada nuevo vector.
        """
        # Codificar nuevas covariables
        encoded_new = self.encode(new_covariables, mapping)

        labels = []
        for vec in encoded_new:
            distances = np.sum(np.abs(centroids - vec), axis=1)  # Manhattan
            labels.append(np.argmin(distances))

        self.assign_cluster_cardinality = len(clusters[labels[0]])
        self.assign_cluster_distance_to_centroid = min(distances)
        return labels

    def fit_cluster_distributions(self, clusters):
        """
        Ajusta distribuciones lognormales (o bootstrap si <20 datos) para cada posición
        de las matrices de un cluster.

        Args:
            clusters: diccionario {cluster_id: [matrices]} devuelto por clusterize_matrices

        Returns:
            distributions: dict {cluster_id: matriz de distribuciones o datos bootstrap}
        """
        distributions = {}

        for cluster_id, matrices in clusters.items():
            print(f"Cluster {cluster_id}")
            if cluster_id == -1:
                # En caso de ser ruido cogemos todas las matrices
                matrices = np.array(self.historical_matrices)
            else:
                matrices = np.array(matrices)  # shape: (n_samples, m, n)
            if len(matrices) == 0:
                continue

            m, n = matrices[0].shape
            cluster_dists = [[None for _ in range(n)] for _ in range(m)]

            for i in range(m):
                for j in range(n):
                    if i == j:
                        cluster_dists[i][j] = ("0", 0, 0, 0)
                        continue
                    values = matrices[:, i, j]

                    if len(values) >= 30:
                        # Ajuste lognormal
                        shape, loc, scale = st.lognorm.fit(
                            values, floc=0
                        )  # fijamos loc=0
                        cluster_dists[i][j] = ("lognorm", shape, loc, scale)
                    else:
                        # Bootstrap simple
                        cluster_dists[i][j] = ("bootstrap", values)

            distributions[cluster_id] = cluster_dists

        return distributions

    def sample_from_cluster(self, cluster_id, distributions, n_samples=100):
        """
        Genera matrices simuladas a partir de las distribuciones ajustadas para un cluster.

        Args:
            cluster_id: id del cluster al que pertenece el nuevo label
            distributions: salida de fit_cluster_distributions
            n_samples: número de muestras a generar

        Returns:
            samples: lista de matrices simuladas
        """
        cluster_dists = distributions[cluster_id]
        m, n = len(cluster_dists), len(cluster_dists[0])
        samples = []

        for k in range(n_samples):
            mat = np.zeros((m, n))
            for i in range(m):
                for j in range(n):
                    dist = cluster_dists[i][j]
                    if dist[0] == "lognorm":
                        _, shape, loc, scale = dist
                        mat[i, j] = st.lognorm.rvs(shape, loc=loc, scale=scale)
                    elif dist[0] == "bootstrap":
                        _, values = dist
                        mat[i, j] = np.random.choice(values, replace=True)
                    else:
                        mat[i, j] = 0
            samples.append(mat)

        return samples

    def calculate_euclidean_distance(self, point1, point2):
        """
        Calculate Euclidean distance between two points.

        Args:
            point1 (dict): First point with 'x' and 'y' coordinates
            point2 (dict): Second point with 'x' and 'y' coordinates

        Returns:
            float: Euclidean distance between the points
        """
        dx = point1["x"] - point2["x"]
        dy = point1["y"] - point2["y"]
        return (dx**2 + dy**2) ** 0.5

    def print_summary(self):
        """
        Print a summary of the loaded instance.
        """
        print("Instance Summary:")
        print(f"Number of vertices (N): {self.n}")
        print(f"Number of paths (P): {self.m}")
        print(f"Time budget per path (Tmax): {self.tmax}")
        print(f"Number of points: {len(self.points)}")

        if self.start_point:
            print(
                f"Start point: ({self.start_point['x']}, {self.start_point['y']}) - Score: {self.start_point['score']}"
            )
        if self.end_point:
            print(
                f"End point: ({self.end_point['x']}, {self.end_point['y']}) - Score: {self.end_point['score']}"
            )

    def calculate_time_min_matrix(self):
        """
        Calculate the time matrix for all points.
        """
        time_min_matrix = [[0] * self.n for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                time_min_matrix[i][j] = self.calculate_euclidean_distance(
                    self.points[i], self.points[j]
                )
        return time_min_matrix

    def get_deterministic_time(self, point1, point2):
        """
        Get the time between two points.
        """
        return self.deterministic_time_matrix[point1["index"]][point2["index"]]

    def compute_mean_deterministic_time(self):
        mean_deterministic_time_matrix = [[0] * self.n for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                mean_value = np.mean([m[i][j] for m in self.historical_matrices])
                mean_deterministic_time_matrix[i][j] = mean_value
        return mean_deterministic_time_matrix

    def compute_lognormal_simulation_time(self, n_scenarios=100):
        lognormal_historical_time_matrix = [
            [[0] * self.n for _ in range(self.n)] for _ in range(n_scenarios)
        ]
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    values = [0] * n_scenarios
                else:
                    historical_values = [m[i][j] for m in self.historical_matrices]
                    # Convert to lognormal parameters (mean and std of lognormal distribution)
                    mean_lognormal = np.mean(historical_values)
                    std_lognormal = np.std(historical_values)

                    # Convert to parameters of underlying normal distribution
                    # For lognormal distribution: mean = exp(mu + sigma^2/2), var = exp(2*mu + sigma^2)*(exp(sigma^2) - 1)
                    # Solving for mu and sigma:
                    sigma_squared = np.log(1 + (std_lognormal**2) / (mean_lognormal**2))
                    sigma = np.sqrt(sigma_squared)
                    mu = np.log(mean_lognormal) - sigma_squared / 2

                    values = np.random.lognormal(mean=mu, sigma=sigma, size=n_scenarios)
                for k in range(n_scenarios):
                    lognormal_historical_time_matrix[k][i][j] = values[k]
        return lognormal_historical_time_matrix

    def compute_predicted_deterministic_time(
        self,
        max_depth=5,
        learning_rate=0.05,
        max_iter=300,
        l2_regularization=1.0,
        seed=42,
        n_jobs=4,
        verbose=10,
    ):

        # 1) Precompute para evitar repetir trabajo 10.000 veces
        X_train_all = np.asarray(
            [self.encode(row, self.encoding) for row in self.historical_covariables],
            dtype=float,
        )
        X_test = np.asarray(
            self.encode(self.actual_covariables, self.encoding), dtype=float
        ).reshape(1, -1)

        def _one_row(i):
            row_vals = np.zeros(self.n, dtype=float)
            for j in range(self.n):
                if i == j:
                    row_vals[j] = 0.0
                    continue
                val = self.predict_value(
                    i,
                    j,
                    X_train_all=X_train_all,
                    X_test=X_test,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    max_iter=max_iter,
                    l2_regularization=l2_regularization,
                    seed=seed,
                )
                row_vals[j] = val
            return i, row_vals

        results = Parallel(n_jobs=n_jobs, backend="threading", verbose=verbose)(
            delayed(_one_row)(i) for i in range(self.n)
        )
        results.sort(key=lambda x: x[0])  # sort by i
        mat = [row_vals for i, row_vals in results]
        return mat

    def set_deterministic_time(self, deterministic_time_matrix):
        self.deterministic_time_matrix = deterministic_time_matrix

    def predict_value(
        self,
        i,
        j,
        X_train_all=None,
        X_test=None,
        max_depth=5,
        learning_rate=0.05,
        max_iter=300,
        l2_regularization=1.0,
        seed=42,
    ):
        # Diagonal
        if i == j:
            return 0.0

        # X (contexto) y y (tiempo OD para ese par)
        # X_train_all ya precomputado
        if X_train_all is None:
            X_train_all = np.asarray(
                [
                    self.encode(row, self.encoding)
                    for row in self.historical_covariables
                ],
                dtype=float,
            )

        y_raw = np.asarray([m[i][j] for m in self.historical_matrices], dtype=float)

        # Limpieza básica
        mask = np.isfinite(y_raw)
        X_train = X_train_all[mask]
        y_raw = y_raw[mask]

        # Si no hay datos suficientes, devuelve baseline razonable
        if len(y_raw) < 5:
            return float(np.median(y_raw)) if len(y_raw) else 0.0

        # Transformación del target (muy importante para tiempos)
        y_train = np.log1p(y_raw)

        # Test
        if X_test is None:
            X_test = np.asarray(
                self.encode(self.actual_covariables, self.encoding), dtype=float
            ).reshape(1, -1)

        # Modelo (mejor que RandomForest para este caso normalmente)
        model = HistGradientBoostingRegressor(
            max_depth=max_depth,
            learning_rate=learning_rate,
            max_iter=max_iter,
            l2_regularization=l2_regularization,
            random_state=seed,
        )
        model.fit(X_train, y_train)

        # Predicción y vuelta a escala original
        pred = max(0.0, float(np.expm1(model.predict(X_test)[0])))

        return pred

    def set_simulation_time_matrices(self, simulation_time_matrices):
        self.simulation_time_matrices = simulation_time_matrices

    # Función para convertir a vector numérico
    def encode(self, instance, mapping):
        return np.array([mapping[key][instance[key]] for key in mapping])

    def compute_validation_score(self, solution):
        scores = []
        for matrix in self.validation_time_matrices:
            score = 0
            for path in solution.paths:
                last_point = self.start_point
                route_time = 0
                for point in path.points[1:]:
                    if (
                        route_time
                        + matrix[last_point["index"]][point["index"]]
                        + matrix[point["index"]][self.end_point["index"]]
                        <= self.tmax
                    ):
                        route_time += matrix[last_point["index"]][point["index"]]
                        score += point["score"]
                    else:
                        break
                    last_point = point
                route_time += matrix[last_point["index"]][self.end_point["index"]]
            scores.append(score)

        solution.validation_score = np.mean(scores)
        return solution.validation_score
