import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.model_selection import train_test_split
from category_encoders import BinaryEncoder
import numpy as np

def preprocessor(
        mode_cols,
        median_cols,
        drop_cols,
        missing_cols,
        log_cols,
        onehot_cols,
        binary_cols
        
    ):
    

        mode_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent"))
        ])

        median_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ])

        missing_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        log_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one"))
        ])

        onehot_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        binary_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("binary", BinaryEncoder())
        ])


        preprocessor = ColumnTransformer(
            transformers=[
                ("mode", mode_pipeline, mode_cols),
                ("median", median_pipeline, median_cols),
                ("missing", missing_pipeline, missing_cols),
                ("log", log_pipeline, log_cols),
                ("onehot", onehot_pipeline, onehot_cols),
                ("binary", binary_pipeline, binary_cols),
                ("drop", "drop", drop_cols)
            ],
            remainder="passthrough",
        )

        return preprocessor

      