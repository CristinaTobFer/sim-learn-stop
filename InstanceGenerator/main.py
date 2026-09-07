import os
from GeneratingInstance import GeneratingInstance
from ExtendInstance import ExtendInstance
import numpy as np
import logging

np.random.seed(43)
logging.basicConfig(level=logging.INFO)


def extend_instance(file_path):
    if not os.path.exists(file_path):
        print(f"Test file {file_path} not found. Please check the path.")
        return

    print(f"Testing with file: {file_path}")
    print("=" * 50)

    # Create Instance object
    instance = GeneratingInstance(file_path)

    # Print summary
    instance.print_summary()

    log_variance = 0.1
    n_historical_matrices = 10000
    print(
        f"Generating {n_historical_matrices} historical matrices with log variance {log_variance}"
    )

    extend_instance = ExtendInstance(log_variance=log_variance)

    extend_instance.generate_historical_matrices(
        instance, n_historical_matrices=n_historical_matrices
    )


if __name__ == "__main__":
    instance_path = "original_instances"
    if not os.path.exists(instance_path):
        raise FileNotFoundError(f"El directorio '{instance_path}' no existe")

    # Obtener todas las carpetas (no archivos)
    folders = [
        item
        for item in os.listdir(instance_path)
        if os.path.isdir(os.path.join(instance_path, item))
    ]
    logging.info(f"Folders: {folders}")
    files_to_folder = {
        "Set_21_234": ["p2.3.g", "p2.3.k"],
        "Set_32_234": ["p1.3.k", "p1.4.b"],
        "Set_33_234": ["p3.2.i", "p3.2.j"],
        "Set_64_234": ["p6.4.a", "p6.4.n"],
        "Set_66_234": ["p5.2.t", "p5.3.f"],
        "Set_100_234": ["p4.3.r", "p4.3.i"],
        "Set_102_234": ["p7.2.j", "p7.3.j"],
    }
    for folder in folders:
        files = [
            instance_path + "/" + folder + "/" + item + ".txt"
            for item in files_to_folder[folder]
        ]
        logging.info(f"Files: {files}")
        for file in files:
            file = str(file)
            logging.info(f"Folder: {folder}, File: {file}")

            extend_instance(file)
