# Testing-Linux-and-Servers
# 5g-airflow-mlops
Quick reference: all scripts

File	Responsibility


src/ingest_data.py	Load telemetry
src/validate_data.py	Validate input data
src/feature_engineering.py	Build engineered features
feature_store/.../feature_definitions.py	Define Feast entity and FeatureView
scripts/feast_apply.py	Register Feast definitions
scripts/feast_materialize.py	Populate Feast online features
src/train_model.py	Retrieve historical Feast features, train, log to MLflow
src/evaluate_model.py	Evaluate model against the quality gate
src/register_model.py	Register and promote an approved model
src/predict.py	Run a sample prediction
src/app.py	Serve predictions over HTTP
dags/5g_ml_pipeline.py	Orchestrate the pipeline
