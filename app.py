import streamlit as st
from pathlib import Path
import tempfile
import subprocess
import sys
import zipfile
import shutil

st.set_page_config(
    page_title="Photometry Analysis Website",
    layout="wide"
)

st.title("Fiber Photometry Analysis Website")
st.write(
    "Upload a pyPhotometry `.ppd` file, run Jones-style analysis, "
    "and download the output graphs/CSVs."
)

uploaded_file = st.file_uploader(
    "Upload your .ppd file",
    type=["ppd"]
)

roi_name = st.text_input("ROI name", value="BLA")

scale_mode = st.selectbox(
    "Y-axis scaling mode",
    ["full", "robust", "none"],
    index=0,
    help=(
        "full = match y-axis using full min/max; "
        "robust = ignore extreme outlier spikes; "
        "none = allow each plot to autoscale."
    )
)

save_full = st.checkbox(
    "Save full 20 Hz processed CSV",
    value=False,
    help="This can make large output files. Leave unchecked for faster website use."
)

if uploaded_file is not None:
    st.success(f"Uploaded: {uploaded_file.name}")

    if st.button("Run analysis"):
        with st.spinner("Running photometry analysis..."):
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)

                input_path = tmpdir / uploaded_file.name
                input_path.write_bytes(uploaded_file.getbuffer())

                output_dir = tmpdir / "analysis_output"

                command = [
                    sys.executable,
                    "ana.py",
                    str(input_path),
                    "--roi",
                    roi_name,
                    "--outdir",
                    str(output_dir),
                    "--scale-mode",
                    scale_mode,
                ]

                if save_full:
                    command.append("--save-full")

                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True
                )

                if result.returncode != 0:
                    st.error("Analysis failed.")
                    st.code(result.stderr)
                else:
                    st.success("Analysis complete.")

                    st.subheader("Run log")
                    st.code(result.stdout)

                    png_files = sorted(output_dir.glob("*.png"))
                    html_files = sorted(output_dir.glob("*.html"))
                    csv_files = sorted(output_dir.glob("*.csv"))
                    json_files = sorted(output_dir.glob("*.json"))
                    md_files = sorted(output_dir.glob("*.md"))

                    st.subheader("Graphs")

                    for png in png_files:
                        st.markdown(f"### {png.name}")
                        st.image(str(png), use_container_width=True)

                    st.subheader("Interactive graph")

                    for html in html_files:
                        html_content = html.read_text(encoding="utf-8")
                        st.components.v1.html(
                            html_content,
                            height=900,
                            scrolling=True
                        )

                    zip_path = tmpdir / "photometry_analysis_outputs.zip"
                    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
                        for file_path in png_files + html_files + csv_files + json_files + md_files:
                            z.write(file_path, arcname=file_path.name)

                    st.subheader("Download outputs")
                    st.download_button(
                        label="Download all outputs as ZIP",
                        data=zip_path.read_bytes(),
                        file_name="photometry_analysis_outputs.zip",
                        mime="application/zip"
                    )
else:
    st.info("Upload a `.ppd` file to begin.")
