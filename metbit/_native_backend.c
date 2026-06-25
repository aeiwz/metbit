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
 * nipals_full  –  Complete NIPALS-PLS1 loop in C.
 *
 * Signature (Python): nipals_full(x, y, rows, cols, tol, max_iter)
 *   x         : C-contiguous float64 (rows x cols)
 *   y         : C-contiguous float64 (rows,)
 *   rows, cols: Py_ssize_t
 *   tol       : double convergence tolerance
 *   max_iter  : int
 *
 * Returns: (bytes_w, bytes_u, c_double, bytes_t, n_iter_int)
 *   bytes_w   : float64 (cols,)  weights
 *   bytes_u   : float64 (rows,)  y-scores
 *   c_double  : float64          y-weight scalar
 *   bytes_t   : float64 (rows,)  x-scores
 *   n_iter_int: int               iterations used
 *
 * Memory: O(rows + cols) temporaries – no copy of X.
 * ========================================================================= */
static PyObject *
nipals_full(PyObject *self, PyObject *args)
{
    PyObject *x_obj, *y_obj;
    Py_ssize_t rows, cols;
    double tol;
    int max_iter;

    (void)self;
    if (!PyArg_ParseTuple(args, "OOnndi:nipals_full",
                          &x_obj, &y_obj, &rows, &cols, &tol, &max_iter))
        return NULL;

    Py_buffer x_buf, y_buf;

    if (!_validate_f64_buffer(x_obj, &x_buf, rows, cols, "nipals_full"))
        return NULL;
    /* validate y: 1-D float64 length rows */
    if (PyObject_GetBuffer(y_obj, &y_buf, PyBUF_C_CONTIGUOUS) < 0) {
        PyBuffer_Release(&x_buf); return NULL;
    }
    if (y_buf.itemsize != (Py_ssize_t)sizeof(double)
            || y_buf.len != rows * (Py_ssize_t)sizeof(double)) {
        PyBuffer_Release(&x_buf); PyBuffer_Release(&y_buf);
        PyErr_Format(PyExc_ValueError,
            "nipals_full: y must be float64 length %zd", rows);
        return NULL;
    }

    /* allocate outputs */
    PyObject *w_bytes = PyBytes_FromStringAndSize(NULL, cols * (Py_ssize_t)sizeof(double));
    PyObject *u_bytes = PyBytes_FromStringAndSize(NULL, rows * (Py_ssize_t)sizeof(double));
    PyObject *t_bytes = PyBytes_FromStringAndSize(NULL, rows * (Py_ssize_t)sizeof(double));
    if (!w_bytes || !u_bytes || !t_bytes) {
        Py_XDECREF(w_bytes); Py_XDECREF(u_bytes); Py_XDECREF(t_bytes);
        PyBuffer_Release(&x_buf); PyBuffer_Release(&y_buf);
        return PyErr_NoMemory();
    }

    const double *x = (const double *)x_buf.buf;
    const double *y = (const double *)y_buf.buf;
    double *w = (double *)PyBytes_AS_STRING(w_bytes);
    double *u = (double *)PyBytes_AS_STRING(u_bytes);
    double *t = (double *)PyBytes_AS_STRING(t_bytes);
    double c = 0.0;
    int n_iter = 0;

    Py_BEGIN_ALLOW_THREADS

    /* initialise u = y */
    for (Py_ssize_t i = 0; i < rows; ++i) u[i] = y[i];
    /* initialise w, t to zero */
    for (Py_ssize_t j = 0; j < cols; ++j) w[j] = 0.0;
    for (Py_ssize_t i = 0; i < rows; ++i) t[i] = 0.0;

    double d = tol * 10.0 + 1.0;

    while (d > tol && n_iter < max_iter) {

        /* w = X.T @ u / (u.u) */
        double utu = 0.0;
        for (Py_ssize_t i = 0; i < rows; ++i) utu += u[i] * u[i];
        if (utu < 1e-300) break;
        for (Py_ssize_t j = 0; j < cols; ++j) w[j] = 0.0;
        for (Py_ssize_t i = 0; i < rows; ++i) {
            const double *row = x + i * cols;
            double ui = u[i];
            Py_ssize_t j = 0;
            for (; j + INNER_BLOCK <= cols; j += INNER_BLOCK) {
                w[j   ] += ui * row[j   ]; w[j+1 ] += ui * row[j+1 ];
                w[j+2 ] += ui * row[j+2 ]; w[j+3 ] += ui * row[j+3 ];
                w[j+4 ] += ui * row[j+4 ]; w[j+5 ] += ui * row[j+5 ];
                w[j+6 ] += ui * row[j+6 ]; w[j+7 ] += ui * row[j+7 ];
                w[j+8 ] += ui * row[j+8 ]; w[j+9 ] += ui * row[j+9 ];
                w[j+10] += ui * row[j+10]; w[j+11] += ui * row[j+11];
                w[j+12] += ui * row[j+12]; w[j+13] += ui * row[j+13];
                w[j+14] += ui * row[j+14]; w[j+15] += ui * row[j+15];
            }
            for (; j < cols; ++j) w[j] += ui * row[j];
        }
        double inv_utu = 1.0 / utu;
        for (Py_ssize_t j = 0; j < cols; ++j) w[j] *= inv_utu;

        /* w /= ||w|| */
        double wtw = 0.0;
        for (Py_ssize_t j = 0; j < cols; ++j) wtw += w[j] * w[j];
        if (wtw < 1e-300) break;
        double inv_wnorm = 1.0 / sqrt(wtw);
        for (Py_ssize_t j = 0; j < cols; ++j) w[j] *= inv_wnorm;

        /* t = X @ w */
        for (Py_ssize_t i = 0; i < rows; ++i) {
            const double *row = x + i * cols;
            double acc = 0.0;
            Py_ssize_t j = 0;
            for (; j + INNER_BLOCK <= cols; j += INNER_BLOCK) {
                acc += row[j   ]*w[j   ] + row[j+1 ]*w[j+1 ]
                     + row[j+2 ]*w[j+2 ] + row[j+3 ]*w[j+3 ]
                     + row[j+4 ]*w[j+4 ] + row[j+5 ]*w[j+5 ]
                     + row[j+6 ]*w[j+6 ] + row[j+7 ]*w[j+7 ]
                     + row[j+8 ]*w[j+8 ] + row[j+9 ]*w[j+9 ]
                     + row[j+10]*w[j+10] + row[j+11]*w[j+11]
                     + row[j+12]*w[j+12] + row[j+13]*w[j+13]
                     + row[j+14]*w[j+14] + row[j+15]*w[j+15];
            }
            for (; j < cols; ++j) acc += row[j] * w[j];
            t[i] = acc;
        }

        /* c = t.y / (t.t) */
        double tty = 0.0, ttt = 0.0;
        for (Py_ssize_t i = 0; i < rows; ++i) { tty += t[i]*y[i]; ttt += t[i]*t[i]; }
        if (ttt < 1e-300) break;
        c = tty / ttt;

        /* convergence: d = ||u_new - u|| / ||u_new||   where u_new = y/c */
        if (fabs(c) < 1e-300) break;
        double inv_c = 1.0 / c;
        double diff2 = 0.0, unew2 = 0.0;
        for (Py_ssize_t i = 0; i < rows; ++i) {
            double un = y[i] * inv_c;
            double diff = un - u[i];
            diff2 += diff * diff;
            unew2 += un * un;
            u[i] = un;
        }
        d = (unew2 > 1e-300) ? sqrt(diff2 / unew2) : 0.0;
        ++n_iter;
    }

    Py_END_ALLOW_THREADS

    PyBuffer_Release(&x_buf);
    PyBuffer_Release(&y_buf);

    /* N steals the reference (no Py_INCREF), so the tuple owns the only
     * reference to each bytes object. Using O here would leak one ref per call. */
    return Py_BuildValue("(NNdNi)", w_bytes, u_bytes, c, t_bytes, n_iter);
}


/* =========================================================================
 * scale_transform  –  Scaler transform: out[i,j] = (X[i,j] - mean[j]) / s[j]
 *
 * Signature (Python): scale_transform(x, mean, s, rows, cols) -> bytes
 *   x, mean, s: float64 C-contiguous buffers
 *   rows, cols: Py_ssize_t
 *
 * Returns bytes (rows*cols float64) – a scaled copy of X.
 * s[j] is typically std[j] (standard scaling) or sqrt(std[j]) (pareto).
 * Zero entries in s are treated as 1.0 to avoid division by zero.
 * ========================================================================= */
static PyObject *
scale_transform(PyObject *self, PyObject *args)
{
    PyObject *x_obj, *mean_obj, *s_obj;
    Py_ssize_t rows, cols;

    (void)self;
    if (!PyArg_ParseTuple(args, "OOOnn:scale_transform",
                          &x_obj, &mean_obj, &s_obj, &rows, &cols))
        return NULL;

    Py_buffer x_buf, mean_buf, s_buf;
    if (!_validate_f64_buffer(x_obj,    &x_buf,    rows, cols, "scale_transform"))
        return NULL;
    if (PyObject_GetBuffer(mean_obj, &mean_buf, PyBUF_C_CONTIGUOUS) < 0) {
        PyBuffer_Release(&x_buf); return NULL;
    }
    if (PyObject_GetBuffer(s_obj, &s_buf, PyBUF_C_CONTIGUOUS) < 0) {
        PyBuffer_Release(&x_buf); PyBuffer_Release(&mean_buf); return NULL;
    }
    if (mean_buf.len != cols * (Py_ssize_t)sizeof(double)
            || s_buf.len != cols * (Py_ssize_t)sizeof(double)) {
        PyBuffer_Release(&x_buf); PyBuffer_Release(&mean_buf); PyBuffer_Release(&s_buf);
        PyErr_Format(PyExc_ValueError,
            "scale_transform: mean and s must be float64 length %zd", cols);
        return NULL;
    }

    PyObject *output = PyBytes_FromStringAndSize(
        NULL, rows * cols * (Py_ssize_t)sizeof(double));
    if (!output) {
        PyBuffer_Release(&x_buf); PyBuffer_Release(&mean_buf); PyBuffer_Release(&s_buf);
        return PyErr_NoMemory();
    }

    const double *x    = (const double *)x_buf.buf;
    const double *mean = (const double *)mean_buf.buf;
    const double *s    = (const double *)s_buf.buf;
    double       *out  = (double *)PyBytes_AS_STRING(output);

    /* Allocate inv_s BEFORE releasing the GIL – PyMem_Malloc requires the GIL. */
    double *inv_s = (double *)PyMem_Malloc((size_t)cols * sizeof(double));
    if (!inv_s) {
        PyBuffer_Release(&x_buf); PyBuffer_Release(&mean_buf); PyBuffer_Release(&s_buf);
        Py_DECREF(output);
        return PyErr_NoMemory();
    }
    for (Py_ssize_t j = 0; j < cols; ++j)
        inv_s[j] = (s[j] != 0.0) ? 1.0 / s[j] : 1.0;

    Py_BEGIN_ALLOW_THREADS

    for (Py_ssize_t i = 0; i < rows; ++i) {
        const double *xrow = x + i * cols;
        double       *orow = out + i * cols;
        Py_ssize_t j = 0;
        for (; j + INNER_BLOCK <= cols; j += INNER_BLOCK) {
            orow[j   ] = (xrow[j   ] - mean[j   ]) * inv_s[j   ];
            orow[j+1 ] = (xrow[j+1 ] - mean[j+1 ]) * inv_s[j+1 ];
            orow[j+2 ] = (xrow[j+2 ] - mean[j+2 ]) * inv_s[j+2 ];
            orow[j+3 ] = (xrow[j+3 ] - mean[j+3 ]) * inv_s[j+3 ];
            orow[j+4 ] = (xrow[j+4 ] - mean[j+4 ]) * inv_s[j+4 ];
            orow[j+5 ] = (xrow[j+5 ] - mean[j+5 ]) * inv_s[j+5 ];
            orow[j+6 ] = (xrow[j+6 ] - mean[j+6 ]) * inv_s[j+6 ];
            orow[j+7 ] = (xrow[j+7 ] - mean[j+7 ]) * inv_s[j+7 ];
            orow[j+8 ] = (xrow[j+8 ] - mean[j+8 ]) * inv_s[j+8 ];
            orow[j+9 ] = (xrow[j+9 ] - mean[j+9 ]) * inv_s[j+9 ];
            orow[j+10] = (xrow[j+10] - mean[j+10]) * inv_s[j+10];
            orow[j+11] = (xrow[j+11] - mean[j+11]) * inv_s[j+11];
            orow[j+12] = (xrow[j+12] - mean[j+12]) * inv_s[j+12];
            orow[j+13] = (xrow[j+13] - mean[j+13]) * inv_s[j+13];
            orow[j+14] = (xrow[j+14] - mean[j+14]) * inv_s[j+14];
            orow[j+15] = (xrow[j+15] - mean[j+15]) * inv_s[j+15];
        }
        for (; j < cols; ++j)
            orow[j] = (xrow[j] - mean[j]) * inv_s[j];
    }

    Py_END_ALLOW_THREADS

    PyMem_Free(inv_s);

    PyBuffer_Release(&x_buf);
    PyBuffer_Release(&mean_buf);
    PyBuffer_Release(&s_buf);
    return output;
}


/* =========================================================================
 * xcorr_max_shift  –  Find the integer shift maximising cross-correlation.
 *
 * Signature (Python): xcorr_max_shift(template, query, n, max_shift) -> (shift, corr)
 *   template, query: C-contiguous float64 length n
 *   n               : Py_ssize_t
 *   max_shift        : Py_ssize_t  search range [-max_shift, +max_shift]
 *
 * Returns (int shift, double best_corr).
 * Used by icoshift: shift > 0 means query is shifted right to align with template.
 * ========================================================================= */
static PyObject *
xcorr_max_shift(PyObject *self, PyObject *args)
{
    PyObject *tmpl_obj, *qry_obj;
    Py_ssize_t n, max_shift;

    (void)self;
    if (!PyArg_ParseTuple(args, "OOnn:xcorr_max_shift",
                          &tmpl_obj, &qry_obj, &n, &max_shift))
        return NULL;

    Py_buffer t_buf, q_buf;
    if (PyObject_GetBuffer(tmpl_obj, &t_buf, PyBUF_C_CONTIGUOUS) < 0) return NULL;
    if (PyObject_GetBuffer(qry_obj,  &q_buf, PyBUF_C_CONTIGUOUS) < 0) {
        PyBuffer_Release(&t_buf); return NULL;
    }
    if (t_buf.itemsize != (Py_ssize_t)sizeof(double)
            || t_buf.len != n * (Py_ssize_t)sizeof(double)
            || q_buf.itemsize != (Py_ssize_t)sizeof(double)
            || q_buf.len != n * (Py_ssize_t)sizeof(double)) {
        PyBuffer_Release(&t_buf); PyBuffer_Release(&q_buf);
        PyErr_Format(PyExc_ValueError,
            "xcorr_max_shift: template and query must be float64 length %zd", n);
        return NULL;
    }

    const double *tmpl = (const double *)t_buf.buf;
    const double *qry  = (const double *)q_buf.buf;
    Py_ssize_t best_shift = 0;
    double best_corr = -1e300;

    Py_BEGIN_ALLOW_THREADS

    for (Py_ssize_t sh = -max_shift; sh <= max_shift; ++sh) {
        double corr = 0.0;
        Py_ssize_t i_start = (sh >= 0) ? 0       : -sh;
        Py_ssize_t i_end   = (sh >= 0) ? n - sh  : n;
        if (i_start >= n || i_end <= 0) continue;
        if (i_end > n) i_end = n;
        /* query[i + sh] aligned with template[i] */
        for (Py_ssize_t i = i_start; i < i_end; ++i)
            corr += tmpl[i] * qry[i + sh];
        if (corr > best_corr) { best_corr = corr; best_shift = sh; }
    }

    Py_END_ALLOW_THREADS

    PyBuffer_Release(&t_buf);
    PyBuffer_Release(&q_buf);
    return Py_BuildValue("(nd)", (long long)best_shift, best_corr);
}


/* =========================================================================
 * pqn_median_quotient  –  One PQN quotient for a single sample.
 *
 * Signature (Python): pqn_median_quotient(sample, reference, n) -> double
 *   sample, reference: C-contiguous float64 length n
 *   n                : Py_ssize_t
 *
 * Returns the median of (sample[i] / reference[i]) over all i where
 * reference[i] != 0.  Returns 1.0 if no valid quotients exist.
 *
 * Uses an in-place partial sort (selection of the median element) to avoid
 * allocating a second temporary array: O(n log n) via qsort of a malloc'd buf.
 * ========================================================================= */
/* Lomuto partition – used by _quickselect. */
static size_t _qs_partition(double *arr, size_t left, size_t right) {
    double pivot = arr[right];
    size_t i = left;
    for (size_t j = left; j < right; ++j) {
        if (arr[j] <= pivot) {
            double tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
            ++i;
        }
    }
    double tmp = arr[i]; arr[i] = arr[right]; arr[right] = tmp;
    return i;
}

/* Hoare quickselect: rearranges arr[0..n-1] so that arr[k] is the k-th
 * smallest element (0-indexed). Average O(n), worst O(n^2). */
static double _quickselect(double *arr, size_t n, size_t k) {
    size_t left = 0, right = n - 1;
    while (left < right) {
        /* Median-of-three pivot to reduce degenerate worst case. */
        size_t mid = left + (right - left) / 2;
        if (arr[mid] < arr[left])  { double t = arr[mid];  arr[mid]  = arr[left];  arr[left]  = t; }
        if (arr[right] < arr[left]){ double t = arr[right]; arr[right]= arr[left];  arr[left]  = t; }
        if (arr[mid] < arr[right]) { double t = arr[mid];  arr[mid]  = arr[right]; arr[right] = t; }
        /* arr[right] is now the median-of-three pivot. */
        size_t p = _qs_partition(arr, left, right);
        if (p == k) break;
        else if (p < k) left  = p + 1;
        else            right = p > 0 ? p - 1 : 0;
    }
    return arr[k];
}

static PyObject *
pqn_median_quotient(PyObject *self, PyObject *args)
{
    PyObject *samp_obj, *ref_obj;
    Py_ssize_t n;

    (void)self;
    if (!PyArg_ParseTuple(args, "OOn:pqn_median_quotient",
                          &samp_obj, &ref_obj, &n))
        return NULL;

    Py_buffer s_buf, r_buf;
    if (PyObject_GetBuffer(samp_obj, &s_buf, PyBUF_C_CONTIGUOUS) < 0) return NULL;
    if (PyObject_GetBuffer(ref_obj,  &r_buf, PyBUF_C_CONTIGUOUS) < 0) {
        PyBuffer_Release(&s_buf); return NULL;
    }
    if (s_buf.itemsize != (Py_ssize_t)sizeof(double) || s_buf.len != n * (Py_ssize_t)sizeof(double)
        || r_buf.itemsize != (Py_ssize_t)sizeof(double) || r_buf.len != n * (Py_ssize_t)sizeof(double)) {
        PyBuffer_Release(&s_buf); PyBuffer_Release(&r_buf);
        PyErr_Format(PyExc_ValueError,
            "pqn_median_quotient: sample and reference must be float64 length %zd", n);
        return NULL;
    }

    const double *samp = (const double *)s_buf.buf;
    const double *ref  = (const double *)r_buf.buf;
    double result = 1.0;

    double *quotients = (double *)PyMem_Malloc((size_t)n * sizeof(double));
    if (!quotients) {
        PyBuffer_Release(&s_buf); PyBuffer_Release(&r_buf);
        return PyErr_NoMemory();
    }

    Py_BEGIN_ALLOW_THREADS

    Py_ssize_t cnt = 0;
    for (Py_ssize_t i = 0; i < n; ++i) {
        if (ref[i] != 0.0)
            quotients[cnt++] = samp[i] / ref[i];
    }
    if (cnt > 0) {
        size_t k = (size_t)cnt / 2;
        if (cnt % 2 == 1) {
            result = _quickselect(quotients, (size_t)cnt, k);
        } else {
            /* Even: median is mean of two middle elements. Need both. */
            double hi = _quickselect(quotients, (size_t)cnt, k);
            /* After first selection arr[k] is in place; find max of arr[0..k-1]. */
            double lo = quotients[0];
            for (size_t i = 1; i < k; ++i)
                if (quotients[i] > lo) lo = quotients[i];
            result = 0.5 * (lo + hi);
        }
    }

    Py_END_ALLOW_THREADS

    PyMem_Free(quotients);
    PyBuffer_Release(&s_buf);
    PyBuffer_Release(&r_buf);
    return PyFloat_FromDouble(result);
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
    {
        "nipals_full", nipals_full, METH_VARARGS,
        PyDoc_STR(
            "nipals_full(x, y, rows, cols, tol, max_iter)"
            " -> (bytes_w, bytes_u, c_float, bytes_t, n_iter)\n"
            "Complete NIPALS-PLS1 loop. x(rows,cols) float64, y(rows,) float64.\n"
            "Returns w(cols), u(rows), c scalar, t(rows) as bytes + iteration count.")
    },
    {
        "scale_transform", scale_transform, METH_VARARGS,
        PyDoc_STR(
            "scale_transform(x, mean, s, rows, cols) -> bytes\n"
            "Compute (X - mean) / s element-wise. Returns scaled float64 copy.\n"
            "s is std (standard scaling) or sqrt(std) (pareto scaling).")
    },
    {
        "xcorr_max_shift", xcorr_max_shift, METH_VARARGS,
        PyDoc_STR(
            "xcorr_max_shift(template, query, n, max_shift) -> (int shift, float corr)\n"
            "Find the integer shift in [-max_shift, max_shift] that maximises\n"
            "the dot-product cross-correlation of template and query.")
    },
    {
        "pqn_median_quotient", pqn_median_quotient, METH_VARARGS,
        PyDoc_STR(
            "pqn_median_quotient(sample, reference, n) -> float\n"
            "Median of sample[i]/reference[i] over non-zero reference entries.\n"
            "Used as the per-sample divisor in PQN normalization.")
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
