import numpy as np


class GeneratingInstance:
    def __init__(self, path):
        """
        Initialize the GeneratingInstance class with a file path.

        Args:
            path (str): Path to the text file to be read
        """
        self.path = path
        self.n = None  # number of vertices
        self.m = None  # number of paths
        self.tmax = None  # available time budget per path
        self.points = []  # list of points with coordinates and scores
        self.start_point = None  # starting point (first point)
        self.end_point = None  # ending point (last point)

        self._read_file()

        self.time_min_matrix = self.calculate_time_min_matrix()

    def _read_file(self):
        """
        Read and parse the text file according to the specified format.
        """
        try:
            with open(self.path, "r") as file:
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
                    self.tmax = float(
                        tmax_line[1]
                    )  # Changed to float since tmax can be decimal

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
