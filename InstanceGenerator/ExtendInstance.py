import numpy as np
import os


class ExtendInstance:
    def __init__(self, log_variance=0.01):
        self.covariables = {
            "time_of_day": {
                "peak": {"mean_increase": 30, "probability": 0.3},
                "valley": {"mean_increase": 10, "probability": 0.4},
                "night": {"mean_increase": 0, "probability": 0.3},
            },
            "day_of_week": {
                "workday": {"mean_increase": 10, "probability": 0.7},
                "weekend": {"mean_increase": 0, "probability": 0.3},
            },
            "weather": {
                "dry": {"mean_increase": 0, "probability": 0.82},
                "rain": {"mean_increase": 20, "probability": 0.16},
                "snow_fog": {"mean_increase": 30, "probability": 0.02},
            },
            #######################################
            "accidents": {
                "no": {"mean_increase": 0, "probability": 0.997},
                "yes": {"mean_increase": 50, "probability": 0.003},
            },
            "special_events": {
                "no": {"mean_increase": 0, "probability": 0.99},
                "yes": {"mean_increase": 30, "probability": 0.01},
            },
            "season": {
                "normal": {"mean_increase": 0, "probability": 0.962},
                "holiday": {"mean_increase": 20, "probability": 0.038},
            },
            "roadworks": {
                "no": {"mean_increase": 0, "probability": 0.9},
                "yes": {"mean_increase": 40, "probability": 0.1},
            },
        }

        self.log_variance = log_variance

    def generate_historical_matrix(
        self,
        instance,
        covariable_values: dict = {
            "time_of_day": "night",
            "day_of_week": "weekend",
            "weather": "dry",
            "accidents": "no",
            "special_events": "no",
            "season": "normal",
            "roadworks": "no",
        },
        seed: int = 43,
    ):
        np.random.seed(seed)

        Tmin = instance.time_min_matrix
        Tstoch = [
            [Tmin[i][j] for j in range(len(instance.points))]
            for i in range(len(instance.points))
        ]

        for index_i in range(len(instance.points)):
            for index_j in range(len(instance.points)):
                if index_i == index_j:
                    continue
                original_time = Tmin[index_i][index_j]
                add_time = 1
                for covariable in self.covariables:
                    add_time *= (
                        1
                        + self.covariables[covariable][covariable_values[covariable]][
                            "mean_increase"
                        ]
                        / 100
                    )
                mu = original_time * add_time
                sigma = float(self.log_variance)

                # Handle the case when mu = 0
                if mu <= 0:
                    mu = 0.1  # Small positive value to avoid division by zero

                sigma2_ln = np.log(1 + sigma / mu**2)
                sigma_ln = np.sqrt(sigma2_ln)
                mu_ln = np.log(mu) - sigma2_ln / 2
                lognormales = np.random.lognormal(mean=mu_ln, sigma=sigma_ln, size=1)
                final_time = float(lognormales[0])
                Tstoch[index_i][index_j] = final_time

        return Tstoch

    def generate_historical_matrices(self, instance, n_historical_matrices=10, seed=43):
        indexes_peak = 1 / 3 * 1 / 3 * n_historical_matrices
        indexes_valley = indexes_peak + 1 / 3 * 1 / 3 * n_historical_matrices
        indexes_night = indexes_valley + 1 / 3 * 1 / 3 * n_historical_matrices

        indexes_workday = indexes_night + 1 / 3 * 1 / 2 * n_historical_matrices
        indexes_weekend = indexes_workday + 1 / 3 * 1 / 2 * n_historical_matrices

        indexes_dry = indexes_weekend + 1 / 3 * 1 / 3 * n_historical_matrices
        indexes_rain = indexes_dry + 1 / 3 * 1 / 3 * n_historical_matrices
        indexes_snow_fog = indexes_rain + 1 / 3 * 1 / 3 * n_historical_matrices

        for i in range(n_historical_matrices):
            seed_i = seed + i
            random_covariable_values = {
                covariable: np.random.choice(
                    list(self.covariables[covariable].keys()),
                    p=[
                        self.covariables[covariable][key]["probability"]
                        for key in self.covariables[covariable].keys()
                    ],
                )
                for covariable in self.covariables
            }
            if i < indexes_peak:
                random_covariable_values["time_of_day"] = "peak"
            elif i < indexes_valley:
                random_covariable_values["time_of_day"] = "valley"
            elif i < indexes_night:
                random_covariable_values["time_of_day"] = "night"
            elif i < indexes_workday:
                random_covariable_values["day_of_week"] = "workday"
            elif i < indexes_weekend:
                random_covariable_values["day_of_week"] = "weekend"
            elif i < indexes_dry:
                random_covariable_values["weather"] = "dry"
            elif i < indexes_rain:
                random_covariable_values["weather"] = "rain"
            elif i < indexes_snow_fog:
                random_covariable_values["weather"] = "snow_fog"

            random_covariable_values["accidents"] = np.random.choice(
                list(self.covariables["accidents"].keys()),
                p=[
                    self.covariables["accidents"][key]["probability"]
                    for key in self.covariables["accidents"].keys()
                ],
            )
            random_covariable_values["special_events"] = np.random.choice(
                list(self.covariables["special_events"].keys()),
                p=[
                    self.covariables["special_events"][key]["probability"]
                    for key in self.covariables["special_events"].keys()
                ],
            )
            random_covariable_values["season"] = np.random.choice(
                list(self.covariables["season"].keys()),
                p=[
                    self.covariables["season"][key]["probability"]
                    for key in self.covariables["season"].keys()
                ],
            )
            random_covariable_values["roadworks"] = np.random.choice(
                list(self.covariables["roadworks"].keys()),
                p=[
                    self.covariables["roadworks"][key]["probability"]
                    for key in self.covariables["roadworks"].keys()
                ],
            )

            Tstoch = self.generate_historical_matrix(
                instance, random_covariable_values, seed=seed_i
            )
            name = f"T_historical_{'_'.join(str(v) for v in random_covariable_values.values())}_{seed_i}.txt"
            self.save_matrix_to_txt(Tstoch, instance, name, random_covariable_values)

    def save_matrix_to_txt(self, matrix, instance, file_name, covariable_values):
        """
        Save a matrix to a text file with space-separated values.

        Args:
            matrix: 2D list or numpy array representing the matrix
            filename: Name of the output file (including .txt extension)
        """
        path = instance.path.split("/")[:-1]
        path = "/".join(path)
        name_instance = instance.path.split(".txt")[0].split("/")[-1]
        make_dir = os.path.join(path, name_instance)
        if not os.path.exists(make_dir):
            os.makedirs(make_dir)
        final_path = os.path.join(make_dir, file_name)

        with open(final_path, "w") as f:
            for covariable_name, covariable_value in covariable_values.items():
                f.write(f"{covariable_name}: {covariable_value}\n")
            f.write("\n")
            for row in matrix:
                # Convert each element to string and join with spaces
                row_str = " ".join(str(element) for element in row)
                f.write(row_str + "\n")
