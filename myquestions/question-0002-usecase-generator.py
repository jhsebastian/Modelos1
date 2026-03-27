import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def generar_caso_de_uso_modelo_polinomico_optimo():
    """
    Genera un caso de uso aleatorio para:
    modelo_polinomico_optimo(df, target_col, grado_max)

    Retorna:
    - input_data: dict con argumentos de entrada
    - output_data: tuple con salida esperada
    """
    rng = np.random.default_rng()

    n_samples = int(rng.integers(90, 180))
    n_features = int(rng.integers(2, 6))
    grado_max = int(rng.integers(2, 6))
    target_col = "target"

    x = rng.uniform(-3, 3, size=(n_samples, n_features))
    true_degree = int(rng.integers(2, grado_max + 1))
    base_signal = np.zeros(n_samples)
    for d in range(1, true_degree + 1):
        w = rng.normal(0, 1, size=n_features)
        base_signal += (x**d) @ w
    y = base_signal + rng.normal(0, 0.7, size=n_samples)

    feature_cols = [f"x_{i}" for i in range(n_features)]
    df = pd.DataFrame(x, columns=feature_cols)
    df[target_col] = y

    input_data = {
        "df": df.copy(),
        "target_col": target_col,
        "grado_max": grado_max,
    }

    x_data = df.drop(columns=[target_col]).to_numpy()
    y_data = df[target_col].to_numpy()

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scorer = make_scorer(root_mean_squared_error, greater_is_better=False)

    mejor_grado = None
    mejor_rmse = float("inf")

    for grado in range(1, grado_max + 1):
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("poly", PolynomialFeatures(degree=grado, include_bias=False)),
                ("reg", LinearRegression()),
            ]
        )
        scores = cross_val_score(model, x_data, y_data, cv=cv, scoring=rmse_scorer)
        rmse_prom = float(-scores.mean())
        if rmse_prom < mejor_rmse:
            mejor_rmse = rmse_prom
            mejor_grado = grado

    output_data = (int(mejor_grado), float(mejor_rmse))
    return input_data, output_data


if __name__ == "__main__":
    entrada, salida = generar_caso_de_uso_modelo_polinomico_optimo()
    print("=== INPUT ===")
    print("target_col:", entrada["target_col"])
    print("grado_max:", entrada["grado_max"])
    print("df shape:", entrada["df"].shape)
    print(entrada["df"].head())

    print("\n=== OUTPUT ESPERADO ===")
    print(salida)