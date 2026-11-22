# AI-Driven Predictive Auto-Scaling Framework for Efficient Cloud Resource Optimization

A research prototype for evaluating machine-learning-based CPU workload prediction as a foundation for proactive Kubernetes auto-scaling.

**Project status:** Experimental research and model evaluation. Predictive scaling is not yet implemented as a production-ready controller.

## 1. Research Objective

This project investigates whether workload and infrastructure telemetry can be used to predict future CPU utilization and support proactive resource allocation in Kubernetes.

The primary research comparison is between:

- **Kubernetes Horizontal Pod Autoscaler (HPA):** Reactive scaling based on observed CPU utilization.
- **Predictive scaling:** A proposed approach that uses machine-learning forecasts to anticipate resource demand.

Both approaches must be evaluated independently under equivalent workload conditions. The HPA and a future predictive controller must never compete to manage the same Deployment simultaneously.

## 2. Architecture

The experimental architecture consists of:

1. **Locust:** Generates constant, ramp-up, spike, burst, and ramp-down workloads.
2. **Flask application:** Provides CPU-intensive workload endpoints and exposes Prometheus metrics.
3. **Kubernetes:** Runs the application and manages replicas through HPA.
4. **Prometheus:** Collects application telemetry.
5. **Metrics collector:** Records CPU, memory, replica counts, request rate, and latency.
6. **ML engine:** Builds time-series features, trains regression models, and evaluates prediction performance.

### Technology Stack

- Python and Flask
- Docker
- Kubernetes and Minikube
- Prometheus
- Locust
- scikit-learn
- Pandas and NumPy
- Matplotlib and Joblib

## 3. Repository Structure

```text
application/       Flask application and Dockerfile
kubernetes/        Application deployment, service and HPA
monitoring/        Prometheus configuration and deployment
workloads/         Locust workload definitions
experiments/       Experiment runner, metrics and collected results
ml-engine/         Dataset generation and model evaluation
datasets/
  processed/       Versioned datasets and train/test splits
models/            Locally generated model artifacts
reports/
  model_evaluation/ Model evaluation metrics and plots
```

## 4. Local Environment Setup

The experimental environment uses Ubuntu, Python, Docker, Minikube and kubectl.

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the application and machine-learning dependencies:

```bash
pip install Flask==3.1.2 gunicorn==23.0.0 prometheus-client==0.23.1
pip install scikit-learn==1.9.1 pandas==3.1.1 numpy==2.5.5
pip install matplotlib==3.11.2 joblib==1.6.0 locust
```

Start Minikube:

```bash
minikube start \
  --driver=docker \
  --cpus=2 \
  --memory=2048 \
  --disk-size=20g
```

Verify the cluster:

```bash
kubectl get nodes
kubectl top nodes
```

## 5. Build and Deploy the Application

Build the application image:

```bash
cd application
docker build -t predictive-autoscaler-app:v3 .
cd ..
```

Load the image into Minikube:

```bash
minikube image load predictive-autoscaler-app:v3
```

Deploy the application and HPA:

```bash
kubectl apply -f kubernetes/app-deployment.yaml
kubectl apply -f kubernetes/app-service.yaml
kubectl apply -f kubernetes/app-hpa.yaml
```

Deploy Prometheus:

```bash
kubectl apply -f monitoring/prometheus-pvc.yaml
kubectl apply -f monitoring/prometheus-deployment.yaml
kubectl apply -f monitoring/prometheus-service.yaml
```

Verify deployment:

```bash
kubectl get pods
kubectl get services
kubectl get hpa
kubectl top pods
```

Experimental endpoints:

- Application: `http://192.168.49.2:30080`
- Prometheus: `http://192.168.49.2:30090`

These addresses assume the documented Minikube Docker-driver environment. The Minikube IP may differ on another machine.

## 6. HPA Baseline and Workload Experiments

The baseline experiments use Kubernetes HPA with a 60% CPU utilization target, a minimum of one replica and a maximum of five replicas.

Workload profiles include:

- Constant load
- Ramp-up load
- Spike load
- Burst load
- Ramp-down load

The training experiment runner supports new batches, resuming existing batches, and retrying failed trials.

Example: create a new batch of 15 trials, each with a 240-second workload:

```bash
python experiments/run_training_experiments.py \
  --limit 15 \
  --duration 240
```

Resume an existing batch:

```bash
python experiments/run_training_experiments.py \
  --resume experiments/results/hpa_training/<batch_directory>
```

Retry failed trials in a resumed batch:

```bash
python experiments/run_training_experiments.py \
  --resume experiments/results/hpa_training/<batch_directory> \
  --retry-failed
```

The experiment outputs include workload statistics, experiment metadata and infrastructure metrics.

## 7. Metrics Collection

The metrics collector records:

- Application CPU utilization in millicores
- Application memory consumption in MiB
- Desired, ready and available replicas
- Request rate
- Average request latency
- P95 request latency
- Collection status and errors

Metrics are sampled at five-second intervals.

CPU and memory measurements are aggregated across application pods. HPA CPU utilization is calculated relative to requested CPU, so it must not be treated as equivalent to aggregate CPU millicores.

## 8. Dataset Generation

The dataset builder audits experiment metadata before accepting a trial.

Eligibility requires a completed v3 application run, the expected Gunicorn configuration, zero Locust failures, and 50 successful metrics samples.

Dataset generation:

```bash
python ml-engine/build_dataset.py
```

The current dataset contains:

| Property | Value |
|---|---:|
| Eligible trials | 13 |
| Total rows | 533 |
| Features | 16 |
| Training rows | 369 |
| Test rows | 164 |
| Training trials | 9 |
| Test trials | 4 |
| Missing values | 0 |

Features include CPU, memory, request rate, latency, replica count, workload activity and lagged telemetry.

The prediction target is CPU utilization 30 seconds ahead.

The train/test split is performed by trial rather than by randomly splitting individual rows. This prevents samples from the same experiment appearing in both sets.

Earlier v2 experiments and failed trials are excluded from the v3 training dataset.

## 9. Model Training and Evaluation

Train the Random Forest model:

```bash
python ml-engine/train_model.py
```

Compare candidate models:

```bash
python ml-engine/compare_models.py
```

Evaluation uses:

- Grouped five-fold cross-validation by trial
- A held-out test set
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² score
- Persistence baseline comparison

### Current Evaluation Findings

The current model comparison produced these held-out results:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Persistence baseline | 62.19 | 110.83 | 0.579 |
| ExtraTrees | 75.63 | 104.52 | 0.625 |

ExtraTrees improves RMSE and R² but has 21.61% higher MAE than persistence.

The Random Forest model also failed to outperform persistence on held-out MAE.

**Research conclusion:** Current evidence does not establish that the evaluated ML models provide sufficiently accurate CPU forecasts to justify their use in an autonomous predictive scaling controller.

Model artifacts and evaluation reports are generated locally. The trained Joblib model is intentionally excluded from version control.

## 10. Reproducibility and Research Integrity

The project retains experiment metadata, raw collected metrics, processed datasets, train/test splits and evaluation reports.

Important controls:

- Separate v2 and v3 application configurations.
- Exclude failed or ineligible experiments from training.
- Preserve failed trial metadata for auditability.
- Split datasets by experiment trial.
- Compare ML models against a persistence baseline.
- Do not fabricate missing measurements.
- Do not claim predictive scaling improvements before controlled comparative experiments.

## 11. Current Limitations and Future Work

Current limitations:

- ML prediction accuracy has not demonstrated an MAE advantage over persistence.
- Only a limited set of workload profiles and experiment trials has been evaluated.
- A production-ready predictive scaling controller has not been implemented.
- Comparative HPA-versus-predictive scaling benchmarks remain future work.

Planned research:

1. Evaluate multiple forecast horizons.
2. Improve model generalization across workload profiles.
3. Establish a reliable predictive model against persistence.
4. Implement a controlled predictive scaling controller.
5. Compare HPA and predictive scaling under equivalent workloads.
6. Evaluate latency, SLA violations, scaling reaction time, replica utilization and estimated resource cost.

---

**Academic Project:** M.Tech in Cloud Computing
**Institution:** IIT Patna
**Project:** AI-Driven Predictive Auto-Scaling Framework for Efficient Cloud Resource Optimization