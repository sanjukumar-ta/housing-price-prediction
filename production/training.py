"""Processors for the model training step of the workflow."""

import logging
import os.path as op

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from ta_lib.core.api import (
    get_dataframe,
    get_feature_names_from_column_transformer,
    load_dataset,
    load_pipeline,
    register_processor,
    save_pipeline,
    DEFAULT_ARTIFACTS_PATH,
)

logger = logging.getLogger(__name__)


@register_processor("model-gen", "train-model")
def train_model(context, params):
    """Train a regression model for housing price prediction."""
    artifacts_folder = DEFAULT_ARTIFACTS_PATH

    input_features_ds = "train/housing/features"
    input_target_ds = "train/housing/target"

    # load training datasets
    train_X = load_dataset(context, input_features_ds)
    train_y = load_dataset(context, input_target_ds)

    # load pre-trained feature pipelines and other artifacts
    features_transformer = load_pipeline(op.join(artifacts_folder, "features.joblib"))

    # Sample data if needed
    sample_frac = params.get("sampling_fraction", None)
    if sample_frac is not None and sample_frac < 1.0:
        logger.warn(f"The data has been sampled by fraction: {sample_frac}")
        sample_X = train_X.sample(frac=sample_frac, random_state=context.random_seed)
        sample_y = train_y.loc[sample_X.index]
    else:
        sample_X = train_X
        sample_y = train_y

    # Transform the training data
    train_X_prepared = features_transformer.transform(sample_X)
    column_names = get_feature_names_from_column_transformer(features_transformer)

    # Train models - Linear Regression and Random Forest
    lin_reg = LinearRegression()
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=context.random_seed)

    # We'll use Random Forest as our final model
    model = rf_reg
    model.fit(train_X_prepared, sample_y.values.ravel())

    # Create training pipeline (for reproducibility)
    train_pipeline = Pipeline([("model", model)])

    # Save fitted training pipeline
    save_pipeline(
        train_pipeline, op.abspath(op.join(artifacts_folder, "train_pipeline.joblib"))
    )

    # Save feature names for later reference
    save_pipeline(
        column_names, op.abspath(op.join(artifacts_folder, "feature_names.joblib"))
    )
