/*
 * _native_backend.c - Numerical kernels for metbit large-scale metabolomics.
 *
 * Functions exported to Python
 * ----------------------------
 *   pearson_columns        float64 input, all columns vs anchor (original, single-thread)
 *   pearson_columns_par    float64 input, OpenMP row-parallel version
 *   pearson_columns_f32    float32 input, accumulates in float64 (half bandwidth)
 *   column_variances       float64 per-column sample variance, two-pass numerically stable
 *   column_variances_f32   float32 input variant
 *   vip_scores             vectorised VIP: t(n,h), w(p,h), q(1,h) -> vip(p,)
 *
 * Auto-dispatch strategy (implemented in _native.py, not here)
 * -------------------------------------------------------------
 *   n*p <= 10_000_000  ->  pearson_columns   (single-thread, fits L3 cache on most CPUs)
 *   n*p >  10_000_000  ->  pearson_columns_par (OpenMP, bounded thread-local memory)
 *   float32 input      ->  pearson_columns_f32 / column_variances_f32
 *   dtype doesn't matter for vip_scores (always float64 in, float64 out)
 *
 * Memory model
 * ------------
 *   All functions work from the raw input buffer only.
 *   No O(n*p) temporary allocation. Thread-local partial-sum arrays are O(p)
 *   per thread, allocated and freed within the parallel region.
 *
 * OpenMP
 * ------
 *   Guarded by #ifdef _OPENMP throughout. Without OpenMP the parallel functions
 *   fall back to their single-threaded equivalents automatically. setup.py
 *   compiles with -fopenmp when the compiler supports it, otherwise without.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#  include <omp.h>
#endif

/* Tunable: number of columns processed together in the inner loop to help
 * the compiler auto-vectorise and keep accumulators in SIMD registers. */
#define INNER_BLOCK 16


/* =========================================================================
 * Utility: validate a C-contiguous float64 buffer with the stated shape.
 * Returns 0 on failure (exception already set), 1 on success.
 * ========================================================================= */
static int
_validate_f64_buffer(PyObject *obj, Py_buffer *view,
                     Py_ssize_t rows, Py_ssize_t cols,
                     const char *func_name)
{
    if (PyObject_GetBuffer(obj, view, PyBUF_C_CONTIGUOUS) < 0)
        return 0;
    if (view->itemsize != (Py_ssize_t)sizeof(double)
            || view->len != rows * cols * (Py_ssize_t)sizeof(double)) {
        PyBuffer_Release(view);
        PyErr_Format(PyExc_ValueError,
            "%s: expected C-contiguous float64 matrix (%zd x %zd)",
            func_name, rows, cols);
        return 0;
    }
    return 1;
}


/* =========================================================================
 * pearson_columns  (float64, single-threaded, original algorithm)
 *
 * Two-pass: compute column means, then covariances + sum-of-squares.
 * Memory: O(3*columns) auxiliary arrays.
 * ========================================================================= */
static PyObject *
pearson_columns(PyObject *self, PyObject *args)
{
    PyObject *data_obj;
    Py_buffer buf;
    Py_ssize_t rows, columns, anchor_index;

    (void)self;
    if (!PyArg_ParseTuple(args, "Onnn:pearson_columns",
                          &data_obj, &rows, &columns, &anchor_index))
        return NULL;
    if (rows < 2 || columns < 1) {
        PyErr_SetString(PyExc_ValueError,
            "data must contain at least two rows and one column");
        return NULL;
    }
    if (anchor_index < 0 || anchor_index >= columns) {
        PyErr_SetString(PyExc_IndexError, "anchor_index is out of range");
        return NULL;
    }
    if (!_validate_f64_buffer(data_obj, &buf, rows, columns, "pearson_columns"))
        return NULL;

    PyObject *output = PyBytes_FromStringAndSize(
        NULL, columns * (Py_ssize_t)sizeof(double));
    if (!output) { PyBuffer_Release(&buf); return NULL; }

    double *means   = (double *)PyMem_Calloc((size_t)columns, sizeof(double));
    double *col_sq  = (double *)PyMem_Calloc((size_t)columns, sizeof(double));
    double *cov     = (double *)PyMem_Calloc((size_t)columns, sizeof(double));
    double *corr    = (double *)PyBytes_AS_STRING(output);
    if (!means || !col_sq || !cov) {
        PyMem_Free(means); PyMem_Free(col_sq); PyMem_Free(cov);
        Py_DECREF(output); PyBuffer_Release(&buf);
        return PyErr_NoMemory();
    }

    const double *data = (const double *)buf.buf;
    double anchor_sq = 0.0;

    Py_BEGIN_ALLOW_THREADS

    /* Pass 1: column means */
    for (Py_ssize_t r = 0; r < rows; ++r) {
        const double *row = data + r * columns;
        for (Py_ssize_t c = 0; c < columns; ++c)
            means[c] += row[c];
    }
    for (Py_ssize_t c = 0; c < columns; ++c)
        means[c] /= (double)rows;

    /* Pass 2: covariance and sum-of-squares */
    for (Py_ssize_t r = 0; r < rows; ++r) {
        const double *row = data + r * columns;
        const double ac = row[anchor_index] - means[anchor_index];
        anchor_sq += ac * ac;
        for (Py_ssize_t c = 0; c < columns; ++c) {
            const double cc = row[c] - means[c];
            cov[c]    += ac * cc;
            col_sq[c] += cc * cc;
        }
    }

    /* Final correlations */
    for (Py_ssize_t c = 0; c < columns; ++c) {
        double denom = sqrt(anchor_sq * col_sq[c]);
        if (denom == 0.0) { corr[c] = Py_NAN; continue; }
        double r = cov[c] / denom;
        corr[c] = (r > 1.0) ? 1.0 : (r < -1.0) ? -1.0 : r;
    }

    Py_END_ALLOW_THREADS

    PyMem_Free(means); PyMem_Free(col_sq); PyMem_Free(cov);
    PyBuffer_Release(&buf);
    return output;
}


/* =========================================================================
 * pearson_columns_par  (float64, OpenMP row-parallel)
 *
 * Each OpenMP thread owns local partial-sum arrays for its row stripe.
 * After the parallel section the main thread reduces across threads.
 * Memory per thread: O(2*columns) for cov and col_sq partials.
 * Total extra memory: O(n_threads * 2 * columns) - ~64 MB for 8 threads, 1M cols.
 *
 * Falls back to single-threaded behaviour when _OPENMP is not defined.
 * ========================================================================= */
static PyObject *
pearson_columns_par(PyObject *self, PyObject *args)
{
    PyObject *data_obj;
    Py_buffer buf;
    Py_ssize_t rows, columns, anchor_index;

    (void)self;
    if (!PyArg_ParseTuple(args, "Onnn:pearson_columns_par",
                          &data_obj, &rows, &columns, &anchor_index))
        return NULL;
    if (rows < 2 || columns < 1) {
        PyErr_SetString(PyExc_ValueError,
            "data must contain at least two rows and one column");
        return NULL;
    }
    if (anchor_index < 0 || anchor_index >= columns) {
        PyErr_SetString(PyExc_IndexError, "anchor_index is out of range");
        return NULL;
    }
    if (!_validate_f64_buffer(data_obj, &buf, rows, columns, "pearson_columns_par"))
        return NULL;

    PyObject *output = PyBytes_FromStringAndSize(
        NULL, columns * (Py_ssize_t)sizeof(double));
    if (!output) { PyBuffer_Release(&buf); return NULL; }

    /* Shared arrays: means, global covariance and column sq, result */
    double *means  = (double *)PyMem_Calloc((size_t)columns, sizeof(double));
    double *cov    = (double *)PyMem_Calloc((size_t)columns, sizeof(double));
    double *col_sq = (double *)PyMem_Calloc((size_t)columns, sizeof(double));
    double *corr   = (double *)PyBytes_AS_STRING(output);
    if (!means || !cov || !col_sq) {
        PyMem_Free(means); PyMem_Free(cov); PyMem_Free(col_sq);
        Py_DECREF(output); PyBuffer_Release(&buf);
        return PyErr_NoMemory();
    }

    const double *data = (const double *)buf.buf;
    double anchor_sq = 0.0;
    int oom = 0;

    Py_BEGIN_ALLOW_THREADS

    /* Pass 1: column means (serial - the reduction is cheap) */
    for (Py_ssize_t r = 0; r < rows; ++r) {
        const double *row = data + r * columns;
        for (Py_ssize_t c = 0; c < columns; ++c)
            means[c] += row[c];
    }
    for (Py_ssize_t c = 0; c < columns; ++c)
        means[c] /= (double)rows;

    /* Pass 2: parallel over rows, thread-local accumulators */
#ifdef _OPENMP
    #pragma omp parallel reduction(+:anchor_sq)
    {
        double *lcov   = (double *)calloc((size_t)columns, sizeof(double));
        double *lcol_sq = (double *)calloc((size_t)columns, sizeof(double));
        if (!lcov || !lcol_sq) {
            free(lcov); free(lcol_sq);
            #pragma omp atomic write
            oom = 1;
        } else {
            #pragma omp for schedule(static)
            for (Py_ssize_t r = 0; r < rows; ++r) {
                const double *row = data + r * columns;
                const double ac = row[anchor_index] - means[anchor_index];
                anchor_sq += ac * ac;
                for (Py_ssize_t c = 0; c < columns; ++c) {
                    const double cc = row[c] - means[c];
                    lcov[c]    += ac * cc;
                    lcol_sq[c] += cc * cc;
                }
            }

            /* Reduction into shared arrays (critical section per thread) */
            #pragma omp critical
            {
                for (Py_ssize_t c = 0; c < columns; ++c) {
                    cov[c]    += lcov[c];
                    col_sq[c] += lcol_sq[c];
                }
            }
            free(lcov);
            free(lcol_sq);
        }
    } /* end omp parallel */
#else
    /* No OpenMP: single-threaded identical to pearson_columns */
    for (Py_ssize_t r = 0; r < rows; ++r) {
        const double *row = data + r * columns;
        const double ac = row[anchor_index] - means[anchor_index];
        anchor_sq += ac * ac;
        for (Py_ssize_t c = 0; c < columns; ++c) {
            const double cc = row[c] - means[c];
            cov[c]    += ac * cc;
            col_sq[c] += cc * cc;
        }
    }
#endif

    if (!oom) {
        for (Py_ssize_t c = 0; c < columns; ++c) {
            double denom = sqrt(anchor_sq * col_sq[c]);
            if (denom == 0.0) { corr[c] = Py_NAN; continue; }
            double r = cov[c] / denom;
            corr[c] = (r > 1.0) ? 1.0 : (r < -1.0) ? -1.0 : r;
        }
    }

    Py_END_ALLOW_THREADS

    PyMem_Free(means); PyMem_Free(cov); PyMem_Free(col_sq);
    PyBuffer_Release(&buf);

    if (oom) {
        Py_DECREF(output);
        return PyErr_NoMemory();
    }
    return output;
}


/* =========================================================================
 * pearson_columns_f32  (float32 input, float64 output)
 *
 * Reads float32 data (half the memory bandwidth of float64) and accumulates
 * internally in float64 for numerical stability.
 * Memory: O(3*columns) float64 auxiliary arrays.
 * ========================================================================= */
static PyObject *
pearson_columns_f32(PyObject *self, PyObject *args)
{
    PyObject *data_obj;
    Py_buffer buf;
    Py_ssize_t rows, columns, anchor_index;

    (void)self;
    if (!PyArg_ParseTuple(args, "Onnn:pearson_columns_f32",
                          &data_obj, &rows, &columns, &anchor_index))
        return NULL;
    if (rows < 2 || columns < 1) {
        PyErr_SetString(PyExc_ValueError,
            "data must contain at least two rows and one column");
        return NULL;
    }
    if (anchor_index < 0 || anchor_index >= columns) {
        PyErr_SetString(PyExc_IndexError, "anchor_index is out of range");
        return NULL;
    }

    if (PyObject_GetBuffer(data_obj, &buf, PyBUF_C_CONTIGUOUS) < 0)
        return NULL;
    if (buf.itemsize != (Py_ssize_t)sizeof(float)
            || buf.len != rows * columns * (Py_ssize_t)sizeof(float)) {
        PyBuffer_Release(&buf);
        PyErr_SetString(PyExc_ValueError,
            "pearson_columns_f32: expected C-contiguous float32 matrix");
        return NULL;
    }

    PyObject *output = PyBytes_FromStringAndSize(
        NULL, columns * (Py_ssize_t)sizeof(double));
    if (!output) { PyBuffer_Release(&buf); return NULL; }

    double *means  = (double *)PyMem_Calloc((size_t)columns, sizeof(double));
    double *cov    = (double *)PyMem_Calloc((size_t)columns, sizeof(double));
    double *col_sq = (double *)PyMem_Calloc((size_t)columns, sizeof(double));
    double *corr   = (double *)PyBytes_AS_STRING(output);
    if (!means || !cov || !col_sq) {
        PyMem_Free(means); PyMem_Free(cov); PyMem_Free(col_sq);
        Py_DECREF(output); PyBuffer_Release(&buf);
        return PyErr_NoMemory();
    }

    const float *data = (const float *)buf.buf;
    double anchor_sq = 0.0;
    int oom_f32 = 0;

    Py_BEGIN_ALLOW_THREADS

    /* Pass 1: column means (promote float32 to float64 on read) */
    for (Py_ssize_t r = 0; r < rows; ++r) {
        const float *row = data + r * columns;
        for (Py_ssize_t c = 0; c < columns; ++c)
            means[c] += (double)row[c];
    }
    for (Py_ssize_t c = 0; c < columns; ++c)
        means[c] /= (double)rows;

    /* Pass 2: covariance and sum-of-squares (parallel when OpenMP available) */
#ifdef _OPENMP
    #pragma omp parallel reduction(+:anchor_sq)
    {
        double *lc  = (double *)calloc((size_t)columns, sizeof(double));
        double *ls  = (double *)calloc((size_t)columns, sizeof(double));
        if (!lc || !ls) {
            free(lc); free(ls);
            #pragma omp atomic write
            oom_f32 = 1;
        } else {
            #pragma omp for schedule(static)
            for (Py_ssize_t r = 0; r < rows; ++r) {
                const float *row = data + r * columns;
                const double ac = (double)row[anchor_index] - means[anchor_index];
                anchor_sq += ac * ac;
                for (Py_ssize_t c = 0; c < columns; ++c) {
                    const double cc = (double)row[c] - means[c];
                    lc[c] += ac * cc;
                    ls[c] += cc * cc;
                }
            }
            #pragma omp critical
            {
                for (Py_ssize_t c = 0; c < columns; ++c) {
                    cov[c]    += lc[c];
                    col_sq[c] += ls[c];
                }
            }
            free(lc); free(ls);
        }
    }
#else
    for (Py_ssize_t r = 0; r < rows; ++r) {
        const float *row = data + r * columns;
        const double ac = (double)row[anchor_index] - means[anchor_index];
        anchor_sq += ac * ac;
        for (Py_ssize_t c = 0; c < columns; ++c) {
            const double cc = (double)row[c] - means[c];
            cov[c]    += ac * cc;
            col_sq[c] += cc * cc;
        }
    }
#endif

    if (!oom_f32) {
        for (Py_ssize_t c = 0; c < columns; ++c) {
            double denom = sqrt(anchor_sq * col_sq[c]);
            if (denom == 0.0) { corr[c] = Py_NAN; continue; }
            double r = cov[c] / denom;
            corr[c] = (r > 1.0) ? 1.0 : (r < -1.0) ? -1.0 : r;
        }
    }

    Py_END_ALLOW_THREADS

    PyMem_Free(means); PyMem_Free(cov); PyMem_Free(col_sq);
    PyBuffer_Release(&buf);
    if (oom_f32) {
        Py_DECREF(output);
        return PyErr_NoMemory();
    }
    return output;
}


/* =========================================================================
 * column_variances  (float64, two-pass, sample variance)
 *
 * Pass 1: column means.
 * Pass 2: sum-of-squared deviations.
 * variance[c] = SS[c] / (rows - 1)
 *
 * Returns bytes: float64 array of length columns.
 * ========================================================================= */
static PyObject *
column_variances(PyObject *self, PyObject *args)
{
    PyObject *data_obj;
    Py_buffer buf;
    Py_ssize_t rows, columns;

    (void)self;
    if (!PyArg_ParseTuple(args, "Onn:column_variances",
                          &data_obj, &rows, &columns))
        return NULL;
    if (rows < 2 || columns < 1) {
        PyErr_SetString(PyExc_ValueError,
            "column_variances: need at least 2 rows and 1 column");
        return NULL;
    }
    if (!_validate_f64_buffer(data_obj, &buf, rows, columns, "column_variances"))
        return NULL;

    PyObject *output = PyBytes_FromStringAndSize(
        NULL, columns * (Py_ssize_t)sizeof(double));
    if (!output) { PyBuffer_Release(&buf); return NULL; }

    double *means = (double *)PyMem_Calloc((size_t)columns, sizeof(double));
    double *var   = (double *)PyBytes_AS_STRING(output);
    if (!means) {
        Py_DECREF(output); PyBuffer_Release(&buf);
        return PyErr_NoMemory();
    }
    memset(var, 0, (size_t)columns * sizeof(double));

    const double *data = (const double *)buf.buf;
#ifdef _OPENMP
    int oom_var = 0;
#endif

    Py_BEGIN_ALLOW_THREADS

    /* Pass 1: means */
    for (Py_ssize_t r = 0; r < rows; ++r) {
        const double *row = data + r * columns;
        for (Py_ssize_t c = 0; c < columns; ++c)
            means[c] += row[c];
    }
    for (Py_ssize_t c = 0; c < columns; ++c)
        means[c] /= (double)rows;

    /* Pass 2: sum of squared deviations - OpenMP parallel over rows */
#ifdef _OPENMP
    #pragma omp parallel
    {
        double *lvar = (double *)calloc((size_t)columns, sizeof(double));
        if (!lvar) {
            #pragma omp atomic write
            oom_var = 1;
        } else {
            #pragma omp for schedule(static)
            for (Py_ssize_t r = 0; r < rows; ++r) {
                const double *row = data + r * columns;
                for (Py_ssize_t c = 0; c < columns; ++c) {
                    const double d = row[c] - means[c];
                    lvar[c] += d * d;
                }
            }
            #pragma omp critical
            {
                for (Py_ssize_t c = 0; c < columns; ++c)
                    var[c] += lvar[c];
            }
            free(lvar);
        }
    }
    if (!oom_var) {
        for (Py_ssize_t c = 0; c < columns; ++c)
            var[c] /= (double)(rows - 1);
    }
#else
    for (Py_ssize_t r = 0; r < rows; ++r) {
        const double *row = data + r * columns;
        for (Py_ssize_t c = 0; c < columns; ++c) {
            const double d = row[c] - means[c];
            var[c] += d * d;
        }
    }
    for (Py_ssize_t c = 0; c < columns; ++c)
        var[c] /= (double)(rows - 1);
#endif

    Py_END_ALLOW_THREADS

    PyMem_Free(means);
    PyBuffer_Release(&buf);
#ifdef _OPENMP
    if (oom_var) {
        Py_DECREF(output);
        return PyErr_NoMemory();
    }
#endif
    return output;
}


/* =========================================================================
 * column_variances_f32  (float32 input, float64 output)
 * ========================================================================= */
static PyObject *
column_variances_f32(PyObject *self, PyObject *args)
{
    PyObject *data_obj;
    Py_buffer buf;
    Py_ssize_t rows, columns;

    (void)self;
    if (!PyArg_ParseTuple(args, "Onn:column_variances_f32",
                          &data_obj, &rows, &columns))
        return NULL;
    if (rows < 2 || columns < 1) {
        PyErr_SetString(PyExc_ValueError,
            "column_variances_f32: need at least 2 rows and 1 column");
        return NULL;
    }
    if (PyObject_GetBuffer(data_obj, &buf, PyBUF_C_CONTIGUOUS) < 0)
        return NULL;
    if (buf.itemsize != (Py_ssize_t)sizeof(float)
            || buf.len != rows * columns * (Py_ssize_t)sizeof(float)) {
        PyBuffer_Release(&buf);
        PyErr_SetString(PyExc_ValueError,
            "column_variances_f32: expected C-contiguous float32 matrix");
        return NULL;
    }

    PyObject *output = PyBytes_FromStringAndSize(
        NULL, columns * (Py_ssize_t)sizeof(double));
    if (!output) { PyBuffer_Release(&buf); return NULL; }

    double *means = (double *)PyMem_Calloc((size_t)columns, sizeof(double));
    double *var   = (double *)PyBytes_AS_STRING(output);
    if (!means) {
        Py_DECREF(output); PyBuffer_Release(&buf);
        return PyErr_NoMemory();
    }
    memset(var, 0, (size_t)columns * sizeof(double));

    const float *data = (const float *)buf.buf;
#ifdef _OPENMP
    int oom_vf = 0;
#endif

    Py_BEGIN_ALLOW_THREADS

    for (Py_ssize_t r = 0; r < rows; ++r) {
        const float *row = data + r * columns;
        for (Py_ssize_t c = 0; c < columns; ++c)
            means[c] += (double)row[c];
    }
    for (Py_ssize_t c = 0; c < columns; ++c)
        means[c] /= (double)rows;

#ifdef _OPENMP
    #pragma omp parallel
    {
        double *lv = (double *)calloc((size_t)columns, sizeof(double));
        if (!lv) {
            #pragma omp atomic write
            oom_vf = 1;
        } else {
            #pragma omp for schedule(static)
            for (Py_ssize_t r = 0; r < rows; ++r) {
                const float *row = data + r * columns;
                for (Py_ssize_t c = 0; c < columns; ++c) {
                    const double d = (double)row[c] - means[c];
                    lv[c] += d * d;
                }
            }
            #pragma omp critical
            {
                for (Py_ssize_t c = 0; c < columns; ++c)
                    var[c] += lv[c];
            }
            free(lv);
        }
    }
    if (!oom_vf) {
        for (Py_ssize_t c = 0; c < columns; ++c)
            var[c] /= (double)(rows - 1);
    }
#else
    for (Py_ssize_t r = 0; r < rows; ++r) {
        const float *row = data + r * columns;
        for (Py_ssize_t c = 0; c < columns; ++c) {
            const double d = (double)row[c] - means[c];
            var[c] += d * d;
        }
    }
    for (Py_ssize_t c = 0; c < columns; ++c)
        var[c] /= (double)(rows - 1);
#endif

    Py_END_ALLOW_THREADS

    PyMem_Free(means);
    PyBuffer_Release(&buf);
#ifdef _OPENMP
    if (oom_vf) {
        Py_DECREF(output);
        return PyErr_NoMemory();
    }
#endif
    return output;
}


/* =========================================================================
 * vip_scores  Vectorised VIP (Variable Importance in Projection) kernel.
 *
 * Inputs (all float64, C-contiguous):
 *   t_data   : (n_samples, n_comp)   PLS/OPLS X-scores
 *   w_data   : (n_feat,   n_comp)   X-weight matrix
 *   q_data   : (n_comp,)            Y-loadings (1 per component)
 *   n_samples, n_feat, n_comp       dimension arguments
 *
 * Algorithm (identical to the numpy vectorised path):
 *   S[h]   = (t[:,h].t @ t[:,h]) * (q[h]^2)        <- per-component scale
 *   norms[h] = ||w[:,h]||
 *   VIP[i] = sqrt( n_feat * sum_h( S[h] * (w[i,h]/norms[h])^2 ) / sum(S) )
 *
 * OpenMP parallelises the outer feature loop (trivially parallel, no
 * shared writes between iterations).
 *
 * Returns bytes: float64 array of length n_feat.
 * ========================================================================= */
static PyObject *
vip_scores(PyObject *self, PyObject *args)
{
    PyObject *t_obj, *w_obj, *q_obj;
    Py_buffer t_buf, w_buf, q_buf;
    Py_ssize_t n_samples, n_feat, n_comp;

    (void)self;
    if (!PyArg_ParseTuple(args, "OOOnnn:vip_scores",
                          &t_obj, &w_obj, &q_obj,
                          &n_samples, &n_feat, &n_comp))
        return NULL;

    if (!_validate_f64_buffer(t_obj, &t_buf, n_samples, n_comp, "vip_scores t"))
        return NULL;
    if (!_validate_f64_buffer(w_obj, &w_buf, n_feat, n_comp, "vip_scores w")) {
        PyBuffer_Release(&t_buf); return NULL;
    }
    if (PyObject_GetBuffer(q_obj, &q_buf, PyBUF_C_CONTIGUOUS) < 0) {
        PyBuffer_Release(&t_buf); PyBuffer_Release(&w_buf); return NULL;
    }
    if (q_buf.itemsize != (Py_ssize_t)sizeof(double)
            || q_buf.len != n_comp * (Py_ssize_t)sizeof(double)) {
        PyBuffer_Release(&t_buf); PyBuffer_Release(&w_buf); PyBuffer_Release(&q_buf);
        PyErr_SetString(PyExc_ValueError,
            "vip_scores: q must be float64 array of length n_comp");
        return NULL;
    }

    PyObject *output = PyBytes_FromStringAndSize(
        NULL, n_feat * (Py_ssize_t)sizeof(double));
    if (!output) {
        PyBuffer_Release(&t_buf); PyBuffer_Release(&w_buf); PyBuffer_Release(&q_buf);
        return NULL;
    }

    /* Per-component scale S[h] and column norms - allocated on stack for small n_comp */
    double *S     = (double *)PyMem_Calloc((size_t)n_comp, sizeof(double));
    double *norms = (double *)PyMem_Calloc((size_t)n_comp, sizeof(double));
    if (!S || !norms) {
        PyMem_Free(S); PyMem_Free(norms);
        Py_DECREF(output);
        PyBuffer_Release(&t_buf); PyBuffer_Release(&w_buf); PyBuffer_Release(&q_buf);
        return PyErr_NoMemory();
    }

    const double *t_data = (const double *)t_buf.buf;
    const double *w_data = (const double *)w_buf.buf;
    const double *q_data = (const double *)q_buf.buf;
    double *vip          = (double *)PyBytes_AS_STRING(output);

    Py_BEGIN_ALLOW_THREADS

    /* Step 1: S[h] = ||t[:,h]||^2 * q[h]^2  (serial, n_comp is tiny: 2-10) */
    for (Py_ssize_t h = 0; h < n_comp; ++h) {
        double t_sq = 0.0;
        for (Py_ssize_t r = 0; r < n_samples; ++r) {
            double v = t_data[r * n_comp + h];
            t_sq += v * v;
        }
        S[h] = t_sq * q_data[h] * q_data[h];
    }

    /* Step 2: column norms of w  (serial, n_comp tiny) */
    for (Py_ssize_t h = 0; h < n_comp; ++h) {
        double sq = 0.0;
        for (Py_ssize_t i = 0; i < n_feat; ++i) {
            double v = w_data[i * n_comp + h];
            sq += v * v;
        }
        norms[h] = sqrt(sq);
        if (norms[h] == 0.0) norms[h] = 1.0;   /* guard zero-norm component */
    }

    double total_s = 0.0;
    for (Py_ssize_t h = 0; h < n_comp; ++h)
        total_s += S[h];

    /* Step 3: VIP per feature - trivially parallel, no shared writes */
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (Py_ssize_t i = 0; i < n_feat; ++i) {
        double acc = 0.0;
        const double *w_row = w_data + i * n_comp;
        for (Py_ssize_t h = 0; h < n_comp; ++h) {
            double w_norm = w_row[h] / norms[h];
            acc += S[h] * w_norm * w_norm;
        }
        vip[i] = (total_s > 0.0)
            ? sqrt((double)n_feat * acc / total_s)
            : 0.0;
    }

    Py_END_ALLOW_THREADS

    PyMem_Free(S);
    PyMem_Free(norms);
    PyBuffer_Release(&t_buf);
    PyBuffer_Release(&w_buf);
    PyBuffer_Release(&q_buf);
    return output;
}


/* =========================================================================
 * openmp_threads  Return number of available OpenMP threads (0 if disabled).
 * Useful for Python-side dispatch decisions.
 * ========================================================================= */
static PyObject *
openmp_threads(PyObject *self, PyObject *args)
{
    (void)self; (void)args;
#ifdef _OPENMP
    return PyLong_FromLong((long)omp_get_max_threads());
#else
    return PyLong_FromLong(0L);
#endif
}


/* =========================================================================
 * Method table and module definition
 * ========================================================================= */
static PyMethodDef native_methods[] = {
    {
        "pearson_columns", pearson_columns, METH_VARARGS,
        PyDoc_STR(
            "pearson_columns(data, rows, columns, anchor_index) -> bytes\n"
            "Single-threaded Pearson r between anchor column and all columns.\n"
            "Best for n*p <= 10_000_000 (fits L3 cache on most CPUs).")
    },
    {
        "pearson_columns_par", pearson_columns_par, METH_VARARGS,
        PyDoc_STR(
            "pearson_columns_par(data, rows, columns, anchor_index) -> bytes\n"
            "OpenMP row-parallel Pearson r. Use for n*p > 10_000_000.\n"
            "Falls back to single-threaded when OpenMP is unavailable.")
    },
    {
        "pearson_columns_f32", pearson_columns_f32, METH_VARARGS,
        PyDoc_STR(
            "pearson_columns_f32(data_f32, rows, columns, anchor_index) -> bytes\n"
            "float32 input, float64 result. Halves memory bandwidth vs f64 path.")
    },
    {
        "column_variances", column_variances, METH_VARARGS,
        PyDoc_STR(
            "column_variances(data, rows, columns) -> bytes\n"
            "Per-column sample variance. float64 input, float64 result.")
    },
    {
        "column_variances_f32", column_variances_f32, METH_VARARGS,
        PyDoc_STR(
            "column_variances_f32(data_f32, rows, columns) -> bytes\n"
            "Per-column sample variance. float32 input, float64 result.")
    },
    {
        "vip_scores", vip_scores, METH_VARARGS,
        PyDoc_STR(
            "vip_scores(t, w, q, n_samples, n_feat, n_comp) -> bytes\n"
            "Vectorised VIP. t(n,h), w(p,h), q(h,). Returns float64 array length p.\n"
            "OpenMP parallel over features when available.")
    },
    {
        "openmp_threads", openmp_threads, METH_NOARGS,
        PyDoc_STR(
            "openmp_threads() -> int\n"
            "Return omp_get_max_threads(). Returns 0 if OpenMP is not available.")
    },
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef native_module = {
    PyModuleDef_HEAD_INIT,
    "_native_backend",
    "Native numerical kernels for metbit: Pearson correlation, column variance,\n"
    "and vectorised VIP scores with optional OpenMP parallelism.",
    -1,
    native_methods
};

PyMODINIT_FUNC
PyInit__native_backend(void)
{
    return PyModule_Create(&native_module);
}
