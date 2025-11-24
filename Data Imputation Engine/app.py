# app.py
import streamlit as st
import pandas as pd
from pathlib import Path

import numeric_type_null_value as ntnv
import object_type_null_value_handel as otnvh

st.set_page_config(page_title="Data Imputation Engine", layout="centered")
st.title("📌 Data Imputation Engine")
st.write("Upload a CSV and choose how you want to handle missing values.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if not uploaded_file:
    st.info("Please upload a CSV file to continue.")
    st.stop()

# read csv into dataframe
try:
    data = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Error reading CSV: {e}")
    st.stop()

original_name = Path(uploaded_file.name).name
st.success("File loaded")
st.write(f"**File:** `{original_name}`")
st.write(f"**Rows:** {data.shape[0]} — **Columns:** {data.shape[1]}")

total_nulls = int(data.isnull().sum().sum())
total_null_percent = (total_nulls * 100) / max(1, data.shape[0])
st.write(f"**Total nulls:** {total_nulls}  |  **Null % (rows):** {total_null_percent:.2f}%")
st.write("---")

if total_null_percent > 40:
    st.error("❌ Data has more than 40% nulls (by rows). Consider revising dataset or sampling.")
    # still allow user to proceed if they insist
    proceed = st.checkbox("Proceed anyway")
    if not proceed:
        st.stop()
else:
    st.info("Dataset OK for imputation.")

st.header("Choose processing")

# If null% <= 5 allow drop vs fill
if total_null_percent <= 5:
    action = st.selectbox("Action for low-null datasets (≤5%):",
                          ["Fill null values", "Drop rows with any nulls"])
    if action == "Drop rows with any nulls":
        cleaned = data.dropna()
        new_name = f"new_{original_name}"
        csv = cleaned.to_csv(index=False).encode('utf-8')
        st.success("Rows with nulls dropped.")
        st.download_button("⬇ Download cleaned CSV", csv, file_name=new_name, mime="text/csv")
        st.stop()
else:
    action = "Fill null values"

# Choose fill options
st.subheader("Numeric columns imputation")
numeric_cols = data.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns.tolist()
if numeric_cols:
    st.write("Numeric columns detected:", numeric_cols)
    num_method = st.selectbox("Select numeric imputation method",
                              ["mean", "median", "mode", "mice"])
else:
    st.info("No numeric columns found.")
    num_method = None

st.subheader("Categorical columns imputation")
object_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
if object_cols:
    st.write("Categorical columns detected:", object_cols)
    obj_method = st.selectbox("Select categorical imputation method",
                              ["mode", "constant", "ffill", "bfill", "missing", "simpleimputer"])
    const_val = None
    if obj_method == "constant":
        const_val = st.text_input("Enter constant value to fill categorical columns with", value="Unknown")
else:
    st.info("No categorical columns found.")
    obj_method = None
    const_val = None

# Preview
if st.checkbox("Preview first 5 rows of original data"):
    st.dataframe(data.head())

if st.button("Run Imputation"):
    df_work = data.copy()
    # numeric
    try:
        if num_method and numeric_cols:
            df_work = ntnv.handel_numeric_nulls(df_work, num_method)
            st.success(f"Numeric columns imputed using: {num_method}")
    except Exception as e:
        st.error(f"Numeric imputation error: {e}")
        st.stop()

    # object
    try:
        if obj_method and object_cols:
            df_work = otnvh.handel_object_nulls(df_work, obj_method, const_value=const_val)
            st.success(f"Categorical columns imputed using: {obj_method}")
    except Exception as e:
        st.error(f"Categorical imputation error: {e}")
        st.stop()

    # Save & provide download
    new_name = f"new_{original_name}"
    csv_bytes = df_work.to_csv(index=False).encode('utf-8')
    st.success("Imputation finished ✅")
    st.download_button("⬇ Download cleaned CSV", csv_bytes, file_name=new_name, mime="text/csv")
    st.write("Preview of cleaned data:")
    st.dataframe(df_work.head())

    # Optionally save to disk (same folder where you run app)
    try:
        Path(new_name).write_bytes(csv_bytes)
        st.info(f"Saved cleaned file to app working directory as `{new_name}`.")
    except Exception:
        # ignore if cannot write
        pass

st.caption("Methods: numeric: mean/median/mode/MICE (IterativeImputer). categorical: mode/constant/ffill/bfill/missing/simpleimputer")
