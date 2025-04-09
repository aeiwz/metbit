import streamlit as st
import zipfile
import os
import tempfile
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from lingress import plot_NMR_spec
from test import nmr_preprocessing  # Adjust this import if needed

st.set_page_config(page_title="NMR Preprocessing Tool", layout="wide")
st.title("🧪 NMR Data Preprocessing App")

uploaded_file = st.file_uploader("📂 Upload zipped Bruker FID folder", type=['zip'])

with st.expander("⚙️ Preprocessing Options"):
    bin_size = st.number_input("Bin size", min_value=0.0001, max_value=0.0100,
                               value=0.0003, step=0.0001, format="%.4f")
    auto_phasing = st.checkbox("Automatic Phase Correction", value=True)
    auto_phasing_method = st.selectbox("Automatic Phase Correction Method",
                                       options=['acme', 'peak_minima'])
    baseline_correction = st.checkbox("Apply Baseline Correction", value=True)
    baseline_type = st.selectbox("Baseline Correction Type",
                                 options=['corrector', 'constant', 'explicit',  'median', 'solvent filter'])
    calibrate_data = st.checkbox("Calibrate Spectrum", value=True)
    calib_type = st.selectbox("Calibration Type", options=['tsp', 'glucose', 'alanine'])

# Add a process button
process_button = st.button("▶️ Process Data")

if uploaded_file and process_button:
    with st.spinner("🔧 Unpacking and processing..."):
        temp_dir = tempfile.mkdtemp()
        try:
            with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            # Detect whether we have one or more FID directories
            fid_paths = []
            for root, dirs, files in os.walk(temp_dir):
                if 'fid' in dirs:
                    fid_paths.append(os.path.join(root, 'fid'))

            # If the uploaded zip is a single FID folder directly, wrap it in a parent folder
            if len(fid_paths) == 1 and os.path.basename(temp_dir) == 'fid':
                parent_dir = os.path.join(temp_dir, "single_exp")
                os.makedirs(parent_dir, exist_ok=True)
                shutil.move(fid_paths[0], os.path.join(parent_dir, 'fid'))
                data_path_to_use = parent_dir
            else:
                data_path_to_use = temp_dir

            # Initialize and run preprocessing
            nmr = nmr_preprocessing(
                data_path=data_path_to_use,
                bin_size=bin_size,
                auto_phasing=auto_phasing,
                fn_=auto_phasing_method,
                baseline_correction=baseline_correction,
                baseline_type=baseline_type,
                calibration=calibrate_data,
                calib_type=calib_type
            )

            df = nmr.get_data()
            ppm = nmr.get_ppm()

            # Validate results
            if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                raise ValueError("❌ Invalid or empty DataFrame returned from get_data()")
            if ppm is None or not hasattr(ppm, '__iter__'):
                raise ValueError("❌ Invalid or non-iterable ppm returned from get_ppm()")

            st.success("✅ Preprocessing completed successfully!")
            #st.write("📋 Processed NMR Data (first few rows):")
            #st.dataframe(df.head())

            # Plotting
            try:
                fig = plot_NMR_spec(spectra=df, ppm=df.columns.astype(float).to_list(), label=None).single_spectra()
                if fig is None:
                    raise ValueError("plot_NMR_spec().single_spectra() returned None.")
                st.subheader("📈 NMR Spectra")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as plot_err:
                st.error("⚠️ Could not generate plot.")
                st.exception(plot_err)

            # Download
            csv = df.to_csv().encode('utf-8')
            st.download_button("⬇️ Download Processed Data", csv, "preprocessed_nmr.csv", "text/csv")

        except Exception as e:
            st.error("❌ An error occurred during preprocessing:")
            st.exception(e)
        finally:
            shutil.rmtree(temp_dir)