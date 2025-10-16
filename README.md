# Modelo predictivo de transacciones (usuarios nuevos)

    **Fecha:** 2025-10-16

    Este repositorio contiene el código y artefactos para un modelo que estima la probabilidad de que un **usuario nuevo** realice su primera transacción.

    ## Estructura (resumen)
    Consulta `FILE_TREE.txt` para el detalle de archivos tras importar la entrega original.

    - `src/` o `03_src/`: código fuente (módulos de datos, features, entrenamiento).
    - `notebooks/` o `02_notebooks/`: notebooks de exploración, feature engineering, modelado y evaluación.
    - `model/` o `04_model/`: artefactos del modelo (ej.: `model_artifact.pkl`, `encoder.pkl`).
    - `reports/` o `05_reports/`: métricas, curvas (ROC/PR/calibración), interpretabilidad (SHAP).
    - `deploy/` o `06_deploy/`: ejemplo de scoring (API/Notebook).
    - `monitoring/` o `07_monitoring/`: plan y checks de deriva/performance.

    > Nota: Este repo replica la estructura recibida. Renombra carpetas si deseas homogeneizar (p. ej., `src/`, `notebooks/`, `model/`, `reports/`).

    ## Requisitos
    - Python 3.10+ recomendado
    - (opcional) Conda o pyenv

    Si existe `environment.yml`:
    ```bash
    conda env create -f environment.yml
    conda activate transacciones
    ```

    Si existe `requirements.txt`:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    ```

    Requisitos sugeridos detectados automáticamente (revísalos y muévelos a `requirements.txt` si aplica):
    ```
    joblib
    matplotlib
    numpy
    pandas
    sklearn
    ```

    ## Reproducibilidad (ejemplo)
    1. Definir variables de entorno si aplica (credenciales, rutas).
    2. Ejecutar notebooks en `02_notebooks/` en orden (01→04) **o** el script de entrenamiento en `src/models/`.
    3. Exportar métricas y curvas a `05_reports/` y artefactos a `04_model/`.
    4. Calibrar el modelo si corresponde (Platt/Isotónica) y guardar el calibrador.

    ## Scoring (ejemplo)
    ```python
    # ejemplo minimal
    import joblib, json
    import numpy as np
    model = joblib.load("04_model/model_artifact.pkl")
    encoder = joblib.load("04_model/encoder.pkl")  # si corresponde
    x = json.load(open("sample_input.json"))
    yhat = model.predict_proba([x["features"]])[:,1][0]
    print("score:", float(yhat))
    ```

    ## Model Card
    Ver `Model_Card.md` para propósito, métricas clave, ventanas, riesgos y mantenimiento.

    ## Licencia y privacidad
    - Asegúrate de remover PII o usar datos anonimizados antes de abrir este repo.
    - Define una licencia apropiada (por defecto, se deja sin licencia).
