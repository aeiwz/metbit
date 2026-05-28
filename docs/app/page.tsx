import Link from 'next/link'
import { FiBookOpen, FiArrowRight, FiSettings, FiActivity, FiBarChart, FiCode, FiPackage } from 'react-icons/fi'

export default function HomePage() {
  const year = new Date().getFullYear()
  return (
    <>
      <header>
        <h1><FiBookOpen className="icon" aria-hidden /> metbit documentation</h1>
        <p>End-to-end NMR metabolomics preprocessing, modeling, and visualization in Python.</p>
      </header>
      <div className="container">
        <h2>Build reproducible NMR metabolomics workflows</h2>
        <p>Install from PyPI, process Bruker FID folders, normalize and align spectra, fit PCA or OPLS-DA models, then inspect interactive Plotly outputs.</p>
        <p style={{marginTop:12}}>
          <Link className="btn" href="/docs/getting-started"><FiPackage aria-hidden /> Install &amp; Quick Start</Link>
          <span style={{ marginLeft: 8 }} />
          <Link className="btn secondary" href="/docs/api">Browse API</Link>
        </p>
        <div className="grid">
          <div className="card">
            <h3><FiSettings aria-hidden /> Data Processing</h3>
            <p>Read Bruker data, preprocess spectra, normalize intensities, calibrate shifts, and align peaks.</p>
            <div className="links">
              <Link className="pill" href="/docs/api/nmr_preprocess">nmr_preprocessing</Link>
              <Link className="pill" href="/docs/api/baseline">baseline_correct</Link>
              <Link className="pill" href="/docs/api/spec_norm">Normalization</Link>
              <Link className="pill" href="/docs/api/calibrate">calibrate</Link>
              <Link className="pill" href="/docs/api/peak_processe">peak_chops</Link>
            </div>
          </div>
          <div className="card">
            <h3><FiActivity aria-hidden /> Statistical Models</h3>
            <p>Fit PCA/OPLS-DA models, cross-validate and compute VIP scores.</p>
            <div className="links">
              <Link className="pill" href="/docs/api/metbit">opls_da</Link>
              <Link className="pill" href="/docs/api/metbit">pca</Link>
              <Link className="pill" href="/docs/api/lazy_opls_da">lazy_opls_da</Link>
              <Link className="pill" href="/docs/api/cross_validation">CrossValidation</Link>
              <Link className="pill" href="/docs/api/vip">VIP helpers</Link>
            </div>
          </div>
          <div className="card">
            <h3><FiBarChart aria-hidden /> Visualization</h3>
            <p>Generate model plots, STOCSY exploration tools, peak-picking interfaces, and annotation helpers.</p>
            <div className="links">
              <Link className="pill" href="/docs/api/STOCSY">STOCSY</Link>
              <Link className="pill" href="/docs/api/ui_stocsy">STOCSY_app</Link>
              <Link className="pill" href="/docs/api/ui_picky_peak">pickie_peak</Link>
              <Link className="pill" href="/docs/api/take_intensity">get_intensity</Link>
              <Link className="pill" href="/docs/api/annotate_peak">annotate_peak</Link>
            </div>
          </div>
        </div>

        <div className="prose" style={{ textAlign: 'left', marginTop: 24 }}>
          <h3><FiCode className="icon" aria-hidden /> Quick Start</h3>
          <pre><code>{`pip install metbit

from metbit import nmr_preprocessing, Normalization, pca

# Path to a Bruker project folder containing sample subfolders with "fid"
fid_dir = 'path/to/bruker_project'

# Preprocess NMR data: FFT, phasing, baseline correction, calibration
nmr = nmr_preprocessing(
    fid_dir,
    bin_size=0.0005,
    auto_phasing=True,
    baseline_correction=True,
    baseline_type='corrector',
    calibration=True,
    calib_type='tsp',
)

# Get processed matrix and ppm axis
X = nmr.get_data()   # pandas.DataFrame (samples x ppm)
ppm = nmr.get_ppm()  # numpy.ndarray

# Normalize and fit PCA
X_norm = Normalization.pqn_normalization(X)
model = pca(X=X_norm, features_name=ppm, n_components=2)
model.fit()
model.plot_pca_scores().show()
`}</code></pre>
        </div>

        <Link className="ctaBtn" href="/docs/api">
          Browse Full API <FiArrowRight aria-hidden />
        </Link>
      </div>
      <footer>
        &copy; {year} Metbit. All rights reserved.
      </footer>
    </>
  )
}
