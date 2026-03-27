import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler


def generar_caso_de_uso_clustering_dbscan():
    """
    Genera un caso de uso aleatorio para:
    clustering_dbscan(df, eps, min_samples)

    Retorna:
    - input_data: dict con argumentos de entrada
    - output_data: tuple con salida esperada
    """
    rng = np.random.default_rng()

    n_samples = int(rng.integers(120, 240))
    n_features = int(rng.integers(2, 6))
    n_clusters_true = int(rng.integers(2, 5))
    min_samples = int(rng.integers(4, 10))
    eps = float(rng.uniform(0.35, 1.1))

    per_cluster = max(20, n_samples // n_clusters_true)
    clusters = []
    for _ in range(n_clusters_true):
        center = rng.uniform(-5, 5, size=n_features)
        spread = float(rng.uniform(0.25, 0.75))
        chunk = center + rng.normal(0, spread, size=(per_cluster, n_features))
        clusters.append(chunk)

    x = np.vstack(clusters)
    n_noise = int(rng.integers(0, max(3, n_samples // 8)))
    noise = rng.uniform(-8, 8, size=(n_noise, n_features))
    x = np.vstack([x, noise])
    rng.shuffle(x)

    feature_cols = [f"x{i + 1}" for i in range(n_features)]
    df = pd.DataFrame(x, columns=feature_cols)

    input_data = {"df": df.copy(), "eps": eps, "min_samples": min_samples}

    scaler = RobustScaler()
    x_scaled = scaler.fit_transform(df.to_numpy())

    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(x_scaled)

    unique_labels = set(labels)
    n_clusters = int(len(unique_labels - {-1}))
    noise_pct = float(np.mean(labels == -1) * 100.0)

    valid_mask = labels != -1
    valid_labels = labels[valid_mask]
    if n_clusters >= 2 and len(valid_labels) > 1 and len(np.unique(valid_labels)) >= 2:
        silhouette = float(silhouette_score(x_scaled[valid_mask], valid_labels))
    else:
        silhouette = None

    # Se calcula noise_pct por consistencia con el enunciado, pero no se retorna.
    _ = noise_pct
    output_data = (labels.tolist(), n_clusters, silhouette)
    return input_data, output_data


if __name__ == "__main__":
    entrada, salida = generar_caso_de_uso_clustering_dbscan()
    print("=== INPUT ===")
    print("eps:", entrada["eps"])
    print("min_samples:", entrada["min_samples"])
    print("df shape:", entrada["df"].shape)
    print(entrada["df"].head())

    print("\n=== OUTPUT ESPERADO ===")
    print(salida)