
### ✅ Q1. **Install MLflow**

* Create a Python environment (e.g., with `conda` or `venv`)
* Install MLflow:

  ```bash
  pip install mlflow
  ```
* Run and check:

  ```bash
  mlflow --version
  ```


---

### ✅ Q2. **Download and preprocess the data**

* Download the Yellow Taxi trip parquet files (Jan–Mar 2023).
* Run:

  ```bash
  python preprocess_data.py --raw_data_path <your_data_path> --dest_path ./output
  ```
* Go to the `output/` folder and count the files saved.

---

### ✅ Q3. **Train a model with autolog**

* In `train.py`, add:

  ```python
  import mlflow
  import mlflow.sklearn

  mlflow.sklearn.autolog()

  with mlflow.start_run():
      # training code
  ```
* Run the script:

  ```bash
  python train.py
  ```
* Launch MLflow UI:

  ```bash
  mlflow ui
  ```


---

### ✅ Q4. **Launch the tracking server locally**

* Run:

  ```bash
  mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./artifacts \
    --host 0.0.0.0 \
    --port 5000
  ```

---

### ✅ Q5. **Tune model hyperparameters**

* In `hpo.py`, update the objective function to include:

  ```python
  with mlflow.start_run():
      mlflow.log_params(params)
      mlflow.log_metric("rmse", rmse)
  ```
* Run it:

  ```bash
  python hpo.py
  ```
* Go to MLflow UI, check the **random-forest-hyperopt** experiment.
* Sort by `rmse` ascending.

---

### ✅ Q6. **Promote the best model**

* In `register_model.py`, use:

  ```python
  from mlflow.tracking import MlflowClient
  client = MlflowClient()
  client.register_model(model_uri, "your-model-name")
  ```


