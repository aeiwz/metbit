import io
import os
import re
import sys
import tempfile
import subprocess
from os.path import abspath, dirname, join
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


HERE = dirname(abspath(__file__))
LOAD_TEXT = lambda name: io.open(join(HERE, name), encoding='UTF-8').read()


def _read_version():
    text = io.open(join(HERE, "metbit", "__init__.py"), encoding="UTF-8").read()
    m = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not m:
        raise RuntimeError("Cannot find __version__ in metbit/__init__.py")
    return m.group(1)
DESCRIPTION = '\n\n'.join(LOAD_TEXT(_) for _ in [
    'README.rst'
])


# ---------------------------------------------------------------------------
# OpenMP detection: try to compile a trivial OpenMP program. Returns the
# compiler flags to use, or empty lists if unavailable.
# ---------------------------------------------------------------------------

_OPENMP_TEST_SRC = r"""
#include <omp.h>
int main(void) { return omp_get_max_threads(); }
"""

def _detect_openmp():
    """Return (compile_args, link_args) for OpenMP, or ([], []) if unavailable."""
    cc = os.environ.get("CC", "cc")

    # Platform-specific flag candidates
    if sys.platform == "darwin":
        # Apple clang does not ship with OpenMP; Homebrew libomp is the standard fix.
        candidates = [
            (["-Xpreprocessor", "-fopenmp"], ["-lomp"]),
            (["-fopenmp"], ["-fopenmp"]),
        ]
    else:
        candidates = [
            (["-fopenmp"], ["-fopenmp"]),
        ]

    for cflags, lflags in candidates:
        with tempfile.TemporaryDirectory() as tmp:
            src = join(tmp, "omp_test.c")
            exe = join(tmp, "omp_test")
            with open(src, "w") as f:
                f.write(_OPENMP_TEST_SRC)
            cmd = [cc] + cflags + [src, "-o", exe] + lflags
            try:
                r = subprocess.run(
                    cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                if r.returncode == 0:
                    print(f"setup.py: OpenMP available ({' '.join(cflags)})")
                    return cflags, lflags
            except Exception:
                pass

    print("setup.py: OpenMP not found - building single-threaded native backend.")
    return [], []


_OMP_COMPILE, _OMP_LINK = _detect_openmp()


def _native_compile_args():
    """Return optimization flags suitable for local and portable wheel builds."""
    if sys.platform == "win32":
        return ["/O2"]

    args = ["-O3"]
    portable_build = os.environ.get("METBIT_PORTABLE_BUILD", "").lower() in {
        "1", "true", "yes",
    }
    if not portable_build:
        args.insert(1, "-march=native")
    return args


# ---------------------------------------------------------------------------
# Custom build_ext: tolerate compiler errors so a missing C toolchain never
# prevents a pure-Python install.
# ---------------------------------------------------------------------------

class OptionalBuildExt(build_ext):
    def run(self):
        try:
            super().run()
        except Exception as exc:
            self.warn(f"optional native extension was not built: {exc}")

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as exc:
            self.warn(
                f"optional extension {ext.name!r} was not built: {exc}\n"
                "metbit will fall back to its pure-Python / NumPy kernels."
            )


_native_ext = Extension(
    "metbit._native_backend",
    sources=["metbit/_native_backend.c"],
    extra_compile_args=_native_compile_args() + _OMP_COMPILE,
    extra_link_args=_OMP_LINK,
    optional=True,
)


setup(
    name="metbit",
    packages=find_packages(),
    ext_modules=[_native_ext],
    cmdclass={"build_ext": OptionalBuildExt},
    version=_read_version(),
    license="MIT",
    description="Metabolomics data analysis and visualization tools.",
    author="aeiwz",
    author_email="theerayut_aeiw_123@hotmail.com",
    url="https://github.com/aeiwz/metbit.git",
    download_url="https://github.com/aeiwz/metbit/archive/refs/tags/V8.7.7.tar.gz",
    keywords=[
        "Omics", "Multivariate analysis", "Visualization",
        "Data Analysis", "Metabolomics", "Chemometrics",
    ],
    install_requires=[
        "scikit-learn",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "scipy>=1.10",
        "statsmodels",
        "plotly",
        "pyChemometrics",
        "lingress",
        "tqdm",
        "dash",
        "pingouin",
        "nmrglue",
        "pybaselines",
    ],
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
    ],
)
