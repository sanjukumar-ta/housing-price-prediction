"""Processors for the model scoring/evaluation step of the workflow."""

import os.path as op
import numpy as np
from sklearn.metrics import mean_squared_error

from ta_lib.core.api import (
    get_dataframe,
    get_feature_names_from_column_transformer,
    load_dataset,
    load_pipeline,
    register_processor,
    save_dataset,
    DEFAULT_ARTIFACTS_PATH,
)
from ta_lib.core.tracking import start_experiment, is_tracker_supported


@register_processor("model-eval", "score-model")
def score_model(context, params):
    """Score a pre-trained model."""

    input_features_ds = "test/housing/features"
    input_target_ds = "test/housing/target"
    output_ds = "score/housing/output"

    artifacts_folder = DEFAULT_ARTIFACTS_PATH

    # load test datasets
    test_X = load_dataset(context, input_features_ds)
    test_y = load_dataset(context, input_target_ds)

    # load the feature pipeline and training pipelines
    features_transformer = load_pipeline(op.join(artifacts_folder, "features.joblib"))
    model_pipeline = load_pipeline(op.join(artifacts_folder, "train_pipeline.joblib"))
    feature_names = load_pipeline(op.join(artifacts_folder, "feature_names.joblib"))

    # transform the test dataset
    test_X_prepared = features_transformer.transform(test_X)

    # make predictions
    predictions = model_pipeline.predict(test_X_prepared)

    # Calculate metrics
    mse = mean_squared_error(test_y, predictions)
    rmse = np.sqrt(mse)

    # Create a dataframe with predictions for output
    test_results = test_X.copy()
    test_results["predicted_value"] = predictions
    test_results["actual_value"] = test_y.values
    test_results["error"] = (
        test_results["actual_value"] - test_results["predicted_value"]
    )

    # store the predictions for any further processing
    save_dataset(context, test_results, output_ds)

    # Start an MLflow experiment and log parameters and metrics
    if is_tracker_supported(context):
        with start_experiment(context, expt_name="Mlflow-tracker") as tracker:
            # Log parameters
            tracker.log_param("input_features_ds", input_features_ds)
            tracker.log_param("input_target_ds", input_target_ds)
            tracker.log_param("output_ds", output_ds)

            # Log metrics
            tracker.log_metric("mse", mse)
            tracker.log_metric("rmse", rmse)

            # Log the model
            tracker.sklearn.log_model(model_pipeline, "model")

    # Print results for user feedback
    print(f"Model Evaluation Results:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
