# numeric_type_null_value.py
# Streamlit-friendly numeric imputation helpers
import pandas as pd

def handel_numeric_nulls(data: pd.DataFrame, method: str):
    """
    data : original dataframe (will not be modified in-place)
    method : one of "mean", "median", "mode", "mice"
    returns: new dataframe with numeric columns imputed
    """
    new_data = data.copy()
    numeric_cols = new_data.select_dtypes(include=['float64', 'int64', 'int32', 'float32']).columns.tolist()

    if not numeric_cols:
        return new_data

    method = method.lower()
    if method == "mean":
        for col in numeric_cols:
            new_data[col] = new_data[col].fillna(new_data[col].mean())
    elif method == "median":
        for col in numeric_cols:
            new_data[col] = new_data[col].fillna(new_data[col].median())
    elif method == "mode":
        for col in numeric_cols:
            try:
                mode_val = new_data[col].mode(dropna=True)[0]
            except Exception:
                mode_val = new_data[col].median()  # fallback
            new_data[col] = new_data[col].fillna(mode_val)
    elif method == "mice":
        # IterativeImputer (MICE) — sklearn required
        try:
            from sklearn.experimental import enable_iterative_imputer  # noqa
            from sklearn.impute import IterativeImputer
            import numpy as np

            float_like_cols = new_data.select_dtypes(include=['float64', 'float32']).columns.tolist()
            int_like_cols = new_data.select_dtypes(include=['int64', 'int32']).columns.tolist()

            # Apply IterativeImputer on float-like then on int-like (keep dtypes)
            if float_like_cols:
                imputer = IterativeImputer(random_state=0)
                transformed = imputer.fit_transform(new_data[float_like_cols])
                new_data[float_like_cols] = pd.DataFrame(transformed, columns=float_like_cols, index=new_data.index)

            if int_like_cols:
                # convert to float for imputation then round back to ints
                imputer = IterativeImputer(random_state=0)
                transformed = imputer.fit_transform(new_data[int_like_cols].astype(float))
                transformed = pd.DataFrame(transformed, columns=int_like_cols, index=new_data.index)
                # Round and cast back to original int dtype
                for col in int_like_cols:
                    # preserve NaNs (if any)
                    if new_data[col].isna().any():
                        new_data[col] = transformed[col].round().astype('Int64')
                    else:
                        new_data[col] = transformed[col].round().astype(new_data[col].dtype)
        except Exception as e:
            raise RuntimeError(f"Error using MICE (IterativeImputer). Ensure scikit-learn is installed. Details: {e}")
    else:
        raise ValueError("Unknown numeric imputation method. Use: mean, median, mode, mice")

    return new_data
