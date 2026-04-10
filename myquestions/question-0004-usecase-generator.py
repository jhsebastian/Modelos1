import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler


def generar_caso_de_uso_seleccion_y_validacion_q4():
    """
    Genera un caso de uso aleatorio para:
    seleccion_y_validacion(df, target_col, k_features)

    Retorna:
    - input_data: dict con argumentos de entrada
    - output_data: tuple con salida esperada
    """
    rng = np.random.default_rng()

    n_samples = int(rng.integers(100, 180))
    n_features = int(rng.integers(105, 150))
    target_col = "target"
    k_features = int(rng.integers(5, 20))

    x = rng.normal(0, 1, size=(n_samples, n_features))
    coef = rng.normal(0, 1, size=n_features)

    for _ in range(120):
        logits = x @ coef + rng.normal(0, 0.6, size=n_samples)
        probs = 1.0 / (1.0 + np.exp(-logits))
        y = (probs > 0.5).astype(int)
        class_counts = np.bincount(y, minlength=2)
        if class_counts.min() >= 5:
            break
        coef = rng.normal(0, 1, size=n_features)
    else:
        y = np.array([0] * (n_samples // 2) + [1] * (n_samples - n_samples // 2))
        rng.shuffle(y)

    feature_cols = [f"var_{i}" for i in range(n_features)]
    df = pd.DataFrame(x, columns=feature_cols)
    df[target_col] = y

    input_data = {
        "df": df.copy(),
        "target_col": target_col,
        "k_features": k_features,
    }

    x_data = df.drop(columns=[target_col]).to_numpy()
    y_data = df[target_col].to_numpy()

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_data)

    selector = SelectKBest(score_func=mutual_info_classif, k=k_features)
    selector.fit(x_scaled, y_data)
    selected_indices = np.where(selector.get_support())[0].tolist()
    x_selected = selector.transform(x_scaled)

    model = GradientBoostingClassifier(random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, x_selected, y_data, cv=cv, scoring="accuracy")
    avg_accuracy = float(np.mean(scores))

    output_data = (
        selected_indices,
        avg_accuracy,
        (int(x_data.shape[1]), int(k_features)),
    )
    return input_data, output_data


if __name__ == "__main__":
    entrada, salida = generar_caso_de_uso_seleccion_y_validacion_q4()
    print("=== INPUT ===")
    print("target_col:", entrada["target_col"])
    print("k_features:", entrada["k_features"])
    print("df shape:", entrada["df"].shape)
    print(entrada["df"].head())

    print("\n=== OUTPUT ESPERADO ===")
    print(salida)