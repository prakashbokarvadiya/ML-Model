# object_type_null_value_handel.py
# Streamlit-friendly categorical imputation helpers
import pandas as pd

def handel_object_nulls(data: pd.DataFrame, method: str, const_value: str = None):
    """
    method: one of "mode", "constant", "ffill", "bfill", "missing", "simpleimputer"
    const_value: required if method == "constant"
    """
    df = data.copy()
    object_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if not object_cols:
        return df

    method = method.lower()

    if method == "mode":
        for col in object_cols:
            if df[col].isnull().any():
                try:
                    df[col] = df[col].fillna(df[col].mode(dropna=True)[0])
                except Exception:
                    df[col] = df[col].fillna("Missing")
    elif method == "constant":
        if const_value is None:
            const_value = "Unknown"
        for col in object_cols:
            df[col] = df[col].fillna(const_value)
    elif method == "ffill":
        df[object_cols] = df[object_cols].fillna(method='ffill')
    elif method == "bfill":
        df[object_cols] = df[object_cols].fillna(method='bfill')
    elif method == "missing":
        for col in object_cols:
            df[col] = df[col].fillna("Missing")
    elif method == "simpleimputer":
        try:
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='most_frequent')
            for col in object_cols:
                # SimpleImputer expects 2D input
                filled = imputer.fit_transform(df[[col]])
                df[col] = pd.Series(filled.ravel(), index=df.index).astype(df[col].dtype)
        except Exception as e:
            raise RuntimeError(f"Error using SimpleImputer. Ensure scikit-learn is installed. Details: {e}")
    else:
        raise ValueError("Unknown object imputation method. Use: mode, constant, ffill, bfill, missing, simpleimputer")

    return df
