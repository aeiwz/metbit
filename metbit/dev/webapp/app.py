import streamlit as st
import zipfile
import os
import tempfile
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from nmr_preprocessing import nmr_preprocessing

st.set_page_config(page_title="NMR Preprocessing Tool", layout="wide")

st.title("🧪 NMR Data Preprocessing App")

# --- Upload Section ---
uploaded_file = st.file_uploader("Upload zipped Bruker FID folder", type=['zip'])

with st.expander("⚙️ Preprocessing Options"):
    bin_size = st.number_input("Bin size", min_value=0.0001, max_value=0.01, value=0.0003, step=0.0001)
    auto_phasing = st.checkbox("Automatic Phase Correction", value=True)
    baseline_correction = st.checkbox("Apply Baseline Correction", value=True)
    baseline_type = st.selectbox("Baseline Correction Type", options=['linear', 'constant', 'median', 'solvent filter'])
    calibrate = st.checkbox("Calibrate Spectrum", value=True)
    calib_type = st.selectbox("Calibration Type", options=['tsp', 'glucose', 'alanine'])

if uploaded_file:
    with st.spinner("Unpacking and processing..."):
        # Extract the zip
        temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        try:
            nmr = nmr_preprocessing(
                data_path=temp_dir,
                bin_size=bin_size,
                auto_phasing=auto_phasing,
                baseline_correction=baseline_correction,
                baseline_type=baseline_type,
                calibration=calibrate,
                calib_type=calib_type
            )

            st.success("✅ Preprocessing completed!")

            # Show data info
            st.write(nmr)

            # Plot spectra
            df = nmr.get_data()
            ppm = nmr.get_ppm()
            st.subheader("📈 Spectra Overview")
            fig, ax = plt.subplots(figsize=(12, 5))
            for idx in df.index:
                ax.plot(ppm, df.loc[idx], label=str(idx))
            ax.set_xlabel("PPM")
            ax.set_ylabel("Intensity")
            ax.invert_xaxis()
            st.pyplot(fig)

            # Download processed data
            csv = df.to_csv().encode('utf-8')
            st.download_button("Download Preprocessed Data", csv, "preprocessed_nmr.csv", "text/csv")

        except Exception as e:
            st.error(f"❌ Error during processing: {e}")
        finally:
            shutil.rmtree(temp_dir)