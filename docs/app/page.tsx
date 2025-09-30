import Link from 'next/link'
import { FiBookOpen, FiArrowRight, FiSettings, FiActivity, FiBarChart, FiCode, FiPackage } from 'react-icons/fi'

export default function HomePage() {
  const year = new Date().getFullYear()
  return (
    <>
      <header>
        <h1><FiBookOpen className="icon" aria-hidden /> Metbit Python API</h1>
        <p>Preprocess spectra, build PCA/OPLS-DA models, and visualize results.</p>
      </header>
      <div className="container">
        <h2>Welcome</h2>
        <p>Install from PyPI and jump into processing, modeling, and visualization.</p>
        <p style={{marginTop:12}}>
          <a className="btn" href="/docs/getting-started"><FiPackage aria-hidden /> Install & Quick Start</a>
          <span style={{ marginLeft: 8 }} />
          <a className="btn secondary" href="/docs/api">Browse API</a>
        </p>
        <div className="grid">
          <div className="card">
            <h3><FiSettings aria-hidden /> Data Processing</h3>
            <p>Preprocess, normalize and prepare spectra.</p>
            <div className="links">
              <Link className="pill" href="/docs/api/nmr_preprocess">nmr_preprocessing</Link>
              <Link className="pill" href="/docs/api/utility">Normalise</Link>
              <Link className="pill" href="/docs/api/spec_norm">Normalization</Link>
              <Link className="pill" href="/docs/api/calibrate">calibrate</Link>
            </div>
          </div>
          <div className="card">
            <h3><FiActivity aria-hidden /> Statistical Models</h3>
            <p>Fit PCA/OPLS-DA models, cross-validate and compute VIP scores.</p>
            <div className="links">
              <Link className="pill" href="/docs/api/metbit">opls_da</Link>
              <Link className="pill" href="/docs/api/metbit">pca</Link>
              <Link className="pill" href="/docs/api/lazy_opls_da">lazy_opls_da</Link>
              <Link className="pill" href="/docs/api/utility">UnivarStats</Link>
            </div>
          </div>
          <div className="card">
            <h3><FiBarChart aria-hidden /> Visualization</h3>
            <p>Generate plots, STOCSY and interactive UIs.</p>
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

from metbit.nmr_preprocess import nmr_preprocessing

# Path to a Bruker project folder containing sample subfolders with 'fid'
fid_dir = 'path/to/bruker_project'

# 1) Preprocess NMR data (bin, FFT, phasing, optional baseline + calibration)
nmr = nmr_preprocessing(
    fid_dir,
    bin_size=0.0005,
    auto_phasing=True,
    baseline_correction=True,
    baseline_type='corrector',
    calibration=True,
    calib_type='tsp',
)

# 2) Get processed matrix and ppm axis
X = nmr.get_data()   # pandas.DataFrame (samples x ppm)
ppm = nmr.get_ppm()  # numpy.ndarray

print(X.shape, ppm[:5])
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
