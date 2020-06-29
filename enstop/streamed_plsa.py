import numpy as np
import numba

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array, check_random_state
try:
    from sklearn.utils.validation import _check_sample_weight
except ImportError:
    from enstop.utils import _check_sample_weight
from scipy.sparse import issparse, csr_matrix, coo_matrix

from enstop.utils import normalize, coherence, mean_coherence, log_lift, mean_log_lift
from enstop.plsa import log_likelihood, plsa_init


@numba.njit(
    "f4[:,::1](i4[::1],i4[::1],f4[::1],f4[:,::1],f4[:,::1],f4[:,::1],i8,i8,f4)",
    locals={
        "k": numba.types.uint16,
        "w": numba.types.uint32,
        "d": numba.types.uint32,
        "z": numba.types.uint16,
        "nz_idx": numba.types.uint32,
        "norm": numba.types.float32,
    },
    fastmath=True,
    nogil=True,
    parallel=True,
)
def plsa_e_step_on_a_block(
    X_rows,
    X_cols,
    X_vals,
    p_w_given_z,
    p_z_given_d,
    p_z_given_wd_block,
    block_start,
    block_end,
    probability_threshold=1e-32,
):
    """Perform the E-step of pLSA optimization. This amounts to computing the
    probability of each topic given each word document pair. The computation
    implements

    P(z|w,d) = \frac{P(z|w)P(d|z)}{\sum_{z=1}^k P(z|w)P(d|z)}.

    This routine is optimized to work with sparse matrices such that P(z|w,d)
    is only computed for w, d such that X_{w,d} is non-zero, where X is the
    data matrix.

    To make this numba compilable the raw arrays defining the COO format sparse
    matrix must be passed separately.

    To keep memory use lower we only compute a block of P(z|w,d) -- specifically
    we compute it for all topics and a block of non-zeros of X. We can then use
    this block to complete a partial M step before computing the E step for
    the next block.


    Parameters
    ----------
    X_rows: array of shape (nnz,)
        For each non-zero entry of X, the row of the entry.

    X_cols: array of shape (nnz,)
        For each non-zero entry of X, the column of the
        entry.

    X_vals: array of shape (nnz,)
        For each non-zero entry of X, the value of entry.

    p_w_given_z: array of shape (n_topics, n_words)
        The current estimates of values for P(w|z)

    p_z_given_d: array of shape (n_docs, n_topics)
        The current estimates of values for P(z|d)

    p_z_given_wd_block: array of shape (block_size, n_topics)
        The result array to write new estimates of P(z|w,d) to.

    block_start: int
        The index into nen-zeros of X where this block starts

    block_end: int
        The index into nen-zeros of X where this block ends

    probability_threshold: float (optional, default=1e-32)
        Option to promote sparsity. If the value of P(w|z)P(z|d) falls below
        threshold then write a zero for P(z|w,d).

    """

    k = p_w_given_z.shape[0]

    for nz_idx in numba.prange(block_start, block_end):
        d = X_rows[nz_idx]
        w = X_cols[nz_idx]

        norm = 0.0
        for z in range(k):
            v = p_w_given_z[z, w] * p_z_given_d[d, z]
            if v > probability_threshold:
                p_z_given_wd_block[nz_idx - block_start, z] = v
                norm += v
            else:
                p_z_given_wd_block[nz_idx - block_start, z] = 0.0
        for z in range(k):
            if norm > 0:
                p_z_given_wd_block[nz_idx - block_start, z] /= norm

    return p_z_given_wd_block


@numba.njit(
    "void(i4[::1],i4[::1],f4[::1],f4[:,::1],f4[:,::1],f4[:,::1],f4[::1],f4[::1],i8,i8)",
    locals={
        "k": numba.types.uint16,
        "w": numba.types.uint32,
        "d": numba.types.uint32,
        "z": numba.types.uint16,
        "nz_idx": numba.types.uint32,
        "s": numba.types.float32,
    },
    fastmath=True,
    nogil=True,
)
def plsa_partial_m_step_on_a_block(
    X_rows,
    X_cols,
    X_vals,
    p_w_given_z,
    p_z_given_d,
    p_z_given_wd_block,
    norm_pwz,
    norm_pdz,
    block_start,
    block_end,
):
    """Perform a partial M-step of pLSA optimization. This amounts to using the
    estimates of P(z|w,d) to estimate the values P(w|z) and P(z|d). The computation
    implements

    P(w|z) = \frac{\sum_{d\in D} X_{w,d}P(z|w,d)}{\sum_{d,z} X_{w,d}P(z|w,d)}
    P(z|d) = \frac{\sum_{w\in V} X_{w,d}P(z|w,d)}{\sum_{w,d} X_{w,d}P(z|w,d)}

    This routine is optimized to work with sparse matrices such that P(z|w,d) is only
    computed for w, d such that X_{w,d} is non-zero, where X is the data matrix.

    To make this numba compilable the raw arrays defining the COO format sparse
    matrix must be passed separately.

    Note that in order to not store the entire P(z|w,d) matrix in memory at once
    we only process a block of it here. The normalization in the above formulas
    will actually be computed after all blocks have been completed.

    Parameters
    ----------
    X_rows: array of shape (nnz,)
        For each non-zero entry of X, the row of the entry.

    X_cols: array of shape (nnz,)
        For each non-zero entry of X, the column of the
        entry.

    X_vals: array of shape (nnz,)
        For each non-zero entry of X, the value of entry.

    p_w_given_z: array of shape (n_topics, n_words)
        The result array to write new estimates of P(w|z) to.

    p_z_given_d: array of shape (n_docs, n_topics)
        The result array to write new estimates of P(z|d) to.

    p_z_given_wd_block: array of shape (block_size, n_topics)
        The current estimates for P(z|w,d) for a block

    norm_pwz: array of shape (n_topics,)
        Auxilliary array used for storing row norms; this is passed in to save
        reallocations.

    norm_pdz: array of shape (n_docs,)
        Auxilliary array used for storing row norms; this is passed in to save
        reallocations.

    sample_weight: array of shape (n_docs,)
        Input document weights.

    block_start: int
        The index into nen-zeros of X where this block starts

    block_end: int
        The index into nen-zeros of X where this block ends

    """

    k = p_z_given_wd_block.shape[1]

    for nz_idx in range(block_start, block_end):
        d = X_rows[nz_idx]
        w = X_cols[nz_idx]
        x = X_vals[nz_idx]

        for z in range(k):
            s = x * p_z_given_wd_block[nz_idx - block_start, z]

            p_w_given_z[z, w] += s
            p_z_given_d[d, z] += s

            norm_pwz[z] += s
            norm_pdz[d] += s


@numba.njit(
    "void(i4[::1],i4[::1],f4[::1],f4[:,::1],f4[:,::1],f4[:,::1],f4[::1],f4[::1],f4[::1],i8,i8)",
    locals={
        "k": numba.types.uint16,
        "w": numba.types.uint32,
        "d": numba.types.uint32,
        "z": numba.types.uint16,
        "nz_idx": numba.types.uint32,
        "s": numba.types.float32,
    },
    fastmath=True,
    nogil=True,
)
def plsa_partial_m_step_on_a_block_w_sample_weight(
    X_rows,
    X_cols,
    X_vals,
    p_w_given_z,
    p_z_given_d,
    p_z_given_wd_block,
    norm_pwz,
    norm_pdz,
    sample_weight,
    block_start,
    block_end,
):
    """Perform a partial M-step of pLSA optimization. This amounts to using the
    estimates of P(z|w,d) to estimate the values P(w|z) and P(z|d). The computation
    implements

    P(w|z) = \frac{\sum_{d\in D} X_{w,d}P(z|w,d)}{\sum_{d,z} X_{w,d}P(z|w,d)}
    P(z|d) = \frac{\sum_{w\in V} X_{w,d}P(z|w,d)}{\sum_{w,d} X_{w,d}P(z|w,d)}

    This routine is optimized to work with sparse matrices such that P(z|w,d) is only
    computed for w, d such that X_{w,d} is non-zero, where X is the data matrix.

    To make this numba compilable the raw arrays defining the COO format sparse
    matrix must be passed separately.

    Note that in order to not store the entire P(z|w,d) matrix in memory at once
    we only process a block of it here. The normalization in the above formulas
    will actually be computed after all blocks have been completed.

    Parameters
    ----------
    X_rows: array of shape (nnz,)
        For each non-zero entry of X, the row of the entry.

    X_cols: array of shape (nnz,)
        For each non-zero entry of X, the column of the
        entry.

    X_vals: array of shape (nnz,)
        For each non-zero entry of X, the value of entry.

    p_w_given_z: array of shape (n_topics, n_words)
        The result array to write new estimates of P(w|z) to.

    p_z_given_d: array of shape (n_docs, n_topics)
        The result array to write new estimates of P(z|d) to.

    p_z_given_wd_block: array of shape (block_size, n_topics)
        The current estimates for P(z|w,d) for a block

    norm_pwz: array of shape (n_topics,)
        Auxilliary array used for storing row norms; this is passed in to save
        reallocations.

    norm_pdz: array of shape (n_docs,)
        Auxilliary array used for storing row norms; this is passed in to save
        reallocations.

    sample_weight: array of shape (n_docs,)
        Input document weights.

    block_start: int
        The index into nen-zeros of X where this block starts

    block_end: int
        The index into nen-zeros of X where this block ends

    """

    k = p_z_given_wd_block.shape[1]

    for nz_idx in range(block_start, block_end):
        d = X_rows[nz_idx]
        w = X_cols[nz_idx]
        x = X_vals[nz_idx]

        for z in range(k):
            s = x * p_z_given_wd_block[nz_idx - block_start, z]
            t = s * sample_weight[d]

            p_w_given_z[z, w] += t
            p_z_given_d[d, z] += s

            norm_pwz[z] += t
            norm_pdz[d] += s

@numba.njit(parallel=True, fastmath=True, nogil=True)
def plsa_em_step(
    X_rows,
    X_cols,
    X_vals,
    prev_p_w_given_z,
    prev_p_z_given_d,
    next_p_w_given_z,
    next_p_z_given_d,
    p_z_given_wd_block,
    norm_pwz,
    norm_pdz,
    e_step_thresh=1e-32,
):

    k = p_z_given_wd_block.shape[1]
    n = prev_p_z_given_d.shape[0]
    m = prev_p_w_given_z.shape[1]

    block_size = p_z_given_wd_block.shape[0]
    n_blocks = (X_vals.shape[0] // block_size) + 1

    # zero out the norms for recomputation
    norm_pdz[:] = 0.0
    norm_pwz[:] = 0.0

    # Loop over blocks doing E step on a block and a partial M step
    for block_index in range(n_blocks):
        block_start = block_index * block_size
        block_end = min(X_vals.shape[0], block_start + block_size)

        plsa_e_step_on_a_block(
            X_rows,
            X_cols,
            X_vals,
            prev_p_w_given_z,
            prev_p_z_given_d,
            p_z_given_wd_block,
            block_start,
            block_end,
            e_step_thresh,
        )
        plsa_partial_m_step_on_a_block(
            X_rows,
            X_cols,
            X_vals,
            next_p_w_given_z,
            next_p_z_given_d,
            p_z_given_wd_block,
            norm_pwz,
            norm_pdz,
            block_start,
            block_end,
        )

    # Once complete we can normalize to complete the M step
    for z in numba.prange(k):
        if norm_pwz[z] > 0:
            for w in range(m):
                next_p_w_given_z[z, w] /= norm_pwz[z]
        for d in range(n):
            if norm_pdz[d] > 0:
                next_p_z_given_d[d, z] /= norm_pdz[d]

    # Zero out the old matrices, we'll swap them on return and
    # these will become the new "next"
    prev_p_w_given_z[:] = 0.0
    prev_p_z_given_d[:] = 0.0

    return next_p_w_given_z, next_p_z_given_d, prev_p_w_given_z, prev_p_z_given_d


@numba.njit(parallel=True, fastmath=True, nogil=True)
def plsa_em_step_w_sample_weights(
    X_rows,
    X_cols,
    X_vals,
    prev_p_w_given_z,
    prev_p_z_given_d,
    next_p_w_given_z,
    next_p_z_given_d,
    p_z_given_wd_block,
    norm_pwz,
    norm_pdz,
    sample_weight,
    e_step_thresh=1e-32,
):

    k = p_z_given_wd_block.shape[1]
    n = prev_p_z_given_d.shape[0]
    m = prev_p_w_given_z.shape[1]

    block_size = p_z_given_wd_block.shape[0]
    n_blocks = (X_vals.shape[0] // block_size) + 1

    # zero out the norms for recomputation
    norm_pdz[:] = 0.0
    norm_pwz[:] = 0.0

    # Loop over blocks doing E step on a block and a partial M step
    for block_index in range(n_blocks):
        block_start = block_index * block_size
        block_end = min(X_vals.shape[0], block_start + block_size)

        plsa_e_step_on_a_block(
            X_rows,
            X_cols,
            X_vals,
            prev_p_w_given_z,
            prev_p_z_given_d,
            p_z_given_wd_block,
            block_start,
            block_end,
            e_step_thresh,
        )
        plsa_partial_m_step_on_a_block_w_sample_weight(
            X_rows,
            X_cols,
            X_vals,
            next_p_w_given_z,
            next_p_z_given_d,
            p_z_given_wd_block,
            norm_pwz,
            norm_pdz,
            sample_weight,
            block_start,
            block_end,
        )

    # Once complete we can normalize to complete the M step
    for z in numba.prange(k):
        if norm_pwz[z] > 0:
            for w in range(m):
                next_p_w_given_z[z, w] /= norm_pwz[z]
        for d in range(n):
            if norm_pdz[d] > 0:
                next_p_z_given_d[d, z] /= norm_pdz[d]

    # Zero out the old matrices, we'll swap them on return and
    # these will become the new "next"
    prev_p_w_given_z[:] = 0.0
    prev_p_z_given_d[:] = 0.0

    return next_p_w_given_z, next_p_z_given_d, prev_p_w_given_z, prev_p_z_given_d


@numba.njit(fastmath=True, nogil=True)
def plsa_fit_inner_blockwise(
    X_rows,
    X_cols,
    X_vals,
    p_w_given_z,
    p_z_given_d,
    sample_weight,
    block_size=65536,
    n_iter=100,
    n_iter_per_test=10,
    tolerance=0.001,
    e_step_thresh=1e-32,
    use_sample_weights=False,
):
    """Internal loop of EM steps required to optimize pLSA, along with relative
    convergence tests with respect to the log-likelihood of observing the data under
    the model.

    The EM looping will stop when either ``n_iter`` iterations have been reached,
    or if the relative improvement in log-likelihood over the last
    ``n_iter_per_test`` steps is under ``threshold``.

    This function is designed to wrap the internals of the EM process in a numba
    compilable loop, and is not the preferred entry point for fitting a plsa model.

    Parameters
    ----------
    X_rows: array of shape (nnz,)
        For each non-zero entry of X, the row of the entry.

    X_cols: array of shape (nnz,)
        For each non-zero entry of X, the column of the
        entry.

    X_vals: array of shape (nnz,)
        For each non-zero entry of X, the value of entry.

    p_w_given_z: array of shape (n_topics, n_words)
        The current estimates of values for P(w|z)

    p_z_given_d: array of shape (n_docs, n_topics)
        The current estimates of values for P(z|d)

    sample_weight: array of shape (n_docs,)
        Input document weights.

    block_size: int (optional, default=65536)
        The number of nonzero entries of X to process in a block. The larger this
        value the faster the compute may go, but at higher memory cost.

    n_iter: int
        The maximum number iterations of EM to perform

    n_iter_per_test: int
        The number of iterations between tests for
        relative improvement in log-likelihood.

    tolerance: float
        The threshold of relative improvement in
        log-likelihood required to continue iterations.

    e_step_thresh: float (optional, default=1e-32)
        Option to promote sparsity. If the value of P(w|z)P(z|d) in the E step falls
        below threshold then write a zero for P(z|w,d).

    Returns
    -------
    p_z_given_d, p_w_given_z: arrays of shapes (n_docs, n_topics) and (n_topics, n_words)
        The resulting model values of P(z|d) and P(w|z)

    """
    k = p_z_given_d.shape[1]
    n = p_z_given_d.shape[0]

    p_z_given_wd_block = np.zeros((block_size, k), dtype=np.float32)

    norm_pwz = np.zeros(k, dtype=np.float32)
    norm_pdz = np.zeros(n, dtype=np.float32)

    previous_log_likelihood = log_likelihood(
        X_rows, X_cols, X_vals, p_w_given_z, p_z_given_d, sample_weight,
    )

    next_p_w_given_z = np.zeros_like(p_w_given_z)
    next_p_z_given_d = np.zeros_like(p_z_given_d)

    for i in range(n_iter):

        if use_sample_weights:
            p_w_given_z, p_z_given_d, next_p_w_given_z, next_p_z_given_d = \
                plsa_em_step_w_sample_weights(
                X_rows,
                X_cols,
                X_vals,
                p_w_given_z,
                p_z_given_d,
                next_p_w_given_z,
                next_p_z_given_d,
                p_z_given_wd_block,
                norm_pwz,
                norm_pdz,
                sample_weight,
                e_step_thresh,
            )
        else:
            p_w_given_z, p_z_given_d, next_p_w_given_z, next_p_z_given_d = plsa_em_step(
                X_rows,
                X_cols,
                X_vals,
                p_w_given_z,
                p_z_given_d,
                next_p_w_given_z,
                next_p_z_given_d,
                p_z_given_wd_block,
                norm_pwz,
                norm_pdz,
                e_step_thresh,
            )

        if i % n_iter_per_test == 0:
            current_log_likelihood = log_likelihood(
                X_rows, X_cols, X_vals, p_w_given_z, p_z_given_d, sample_weight,
            )
            change = np.abs(current_log_likelihood - previous_log_likelihood)
            if change / np.abs(current_log_likelihood) < tolerance:
                break
            else:
                previous_log_likelihood = current_log_likelihood

    return p_z_given_d, p_w_given_z


def plsa_fit(
    X,
    k,
    sample_weight,
    init="random",
    block_size=65536,
    n_iter=100,
    n_iter_per_test=10,
    tolerance=0.001,
    e_step_thresh=1e-32,
    random_state=None,
):
    """Fit a pLSA model to a data matrix ``X`` with ``k`` topics, an initialized
    according to ``init``. This will run an EM method to optimize estimates of P(z|d)
    and P(w|z). The will perform at most ``n_iter`` EM step iterations,
    while checking for relative improvement of the log-likelihood of the data under
    the model every ``n_iter_per_test`` iterations, and stops early if that is under
    ``tolerance``.

    Parameters
    ----------
    X: sparse matrix of shape (n_docs, n_words)
        The data matrix pLSA is attempting to fit to.

    k: int
        The number of topics for pLSA to fit with.

    sample_weight: array of shape (n_docs,)
        Input document weights.

    init: string or tuple (optional, default="random")
        The intialization method to use. This should be one of:
            * ``"random"``
            * ``"nndsvd"``
            * ``"nmf"``
        or a tuple of two ndarrays of shape (n_docs, n_topics) and (n_topics, n_words).

    block_size: int (optional, default=65536)
        The number of nonzero entries of X to process in a block. The larger this
        value the faster the compute may go, but at higher memory cost.

    n_iter: int
        The maximum number iterations of EM to perform

    n_iter_per_test: int
        The number of iterations between tests for
        relative improvement in log-likelihood.

    tolerance: float
        The threshold of relative improvement in
        log-likelihood required to continue iterations.

    e_step_thresh: float (optional, default=1e-32)
        Option to promote sparsity. If the value of P(w|z)P(z|d) in the E step falls
        below threshold then write a zero for P(z|w,d).

    random_state: int, RandomState instance or None, (optional, default: None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. Used in in initialization.

    Returns
    -------
    p_z_given_d, p_w_given_z: arrays of shapes (n_docs, n_topics) and (n_topics, n_words)
        The resulting model values of P(z|d) and P(w|z)

    """

    rng = check_random_state(random_state)
    p_z_given_d, p_w_given_z = plsa_init(X, k, init=init, rng=rng)
    p_z_given_d = p_z_given_d.astype(np.float32, order="C")
    p_w_given_z = p_w_given_z.astype(np.float32, order="C")

    use_sample_weights = np.any(sample_weight != 1.0)

    A = X.tocoo().astype(np.float32)

    p_z_given_d, p_w_given_z = plsa_fit_inner_blockwise(
        A.row,
        A.col,
        A.data,
        p_w_given_z,
        p_z_given_d,
        sample_weight,
        block_size=block_size,
        n_iter=n_iter,
        n_iter_per_test=n_iter_per_test,
        tolerance=tolerance,
        e_step_thresh=e_step_thresh,
        use_sample_weights=use_sample_weights,
    )

    return p_z_given_d, p_w_given_z


@numba.njit(
    "void(i4[::1],i4[::1],f4[::1],f4[:,::1],f4[:,::1],f4[::1],f4[::1],i8,i8)",
    locals={
        "k": numba.types.uint16,
        "w": numba.types.uint32,
        "d": numba.types.uint32,
        "z": numba.types.uint16,
        "nz_idx": numba.types.uint32,
        "s": numba.types.float32,
    },
    fastmath=True,
    nogil=True,
)
def plsa_partial_refit_m_step_on_a_block(
    X_rows,
    X_cols,
    X_vals,
    p_z_given_d,
    p_z_given_wd_block,
    sample_weight,
    norm_pdz,
    block_start,
    block_end,
):
    """Perform a partial M-step of pLSA optimization. This amounts to using the
    estimates of P(z|w,d) to estimate the values P(w|z) and P(z|d). The computation
    implements

    P(w|z) = \frac{\sum_{d\in D} X_{w,d}P(z|w,d)}{\sum_{d,z} X_{w,d}P(z|w,d)}
    P(z|d) = \frac{\sum_{w\in V} X_{w,d}P(z|w,d)}{\sum_{w,d} X_{w,d}P(z|w,d)}

    This routine is optimized to work with sparse matrices such that P(z|w,d) is only
    computed for w, d such that X_{w,d} is non-zero, where X is the data matrix.

    To make this numba compilable the raw arrays defining the COO format sparse
    matrix must be passed separately.

    Note that in order to not store the entire P(z|w,d) matrix in memory at once
    we only process a block of it here. The normalization in the above formulas
    will actually be computed after all blocks have been completed.

    Parameters
    ----------
    X_rows: array of shape (nnz,)
        For each non-zero entry of X, the row of the entry.

    X_cols: array of shape (nnz,)
        For each non-zero entry of X, the column of the
        entry.

    X_vals: array of shape (nnz,)
        For each non-zero entry of X, the value of entry.

    p_z_given_d: array of shape (n_docs, n_topics)
        The result array to write new estimates of P(z|d) to.

    p_z_given_wd_block: array of shape (block_size, n_topics)
        The current estimates for P(z|w,d) for a block

    sample_weight: array of shape (n_docs,)
        Input document weights.

    norm_pdz: array of shape (n_docs,)
        Auxilliary array used for storing row norms; this is passed in to save
        reallocations.

    block_start: int
        The index into nen-zeros of X where this block starts

    block_end: int
        The index into nen-zeros of X where this block ends

    """

    k = p_z_given_wd_block.shape[1]

    for nz_idx in range(block_start, block_end):
        d = X_rows[nz_idx]
        w = X_cols[nz_idx]
        x = X_vals[nz_idx]

        for z in range(k):
            s = x * p_z_given_wd_block[nz_idx - block_start, z]
            p_z_given_d[d, z] += s
            norm_pdz[d] += s


@numba.njit()
def plsa_refit_em_step(
    X_rows,
    X_cols,
    X_vals,
    p_w_given_z,
    prev_p_z_given_d,
    next_p_z_given_d,
    p_z_given_wd_block,
    sample_weight,
    norm_pdz,
    e_step_thresh=1e-32,
):

    k = p_z_given_wd_block.shape[1]
    n = prev_p_z_given_d.shape[0]

    block_size = p_z_given_wd_block.shape[0]
    n_blocks = (X_vals.shape[0] // block_size) + 1

    # zero out the norms for recomputation
    norm_pdz[:] = 0.0

    # Loop over blocks doing E step on a block and a partial M step
    for block_index in range(n_blocks):
        block_start = block_index * block_size
        block_end = min(X_vals.shape[0], block_start + block_size)

        plsa_e_step_on_a_block(
            X_rows,
            X_cols,
            X_vals,
            p_w_given_z,
            prev_p_z_given_d,
            p_z_given_wd_block,
            block_start,
            block_end,
            e_step_thresh,
        )
        plsa_partial_refit_m_step_on_a_block(
            X_rows,
            X_cols,
            X_vals,
            p_w_given_z,
            next_p_z_given_d,
            p_z_given_wd_block,
            sample_weight,
            norm_pdz,
            block_start,
            block_end,
        )

    # Once complete we can normalize to complete the M step
    for z in numba.prange(k):
        for d in range(n):
            if norm_pdz[d] > 0:
                next_p_z_given_d[d, z] /= norm_pdz[d]

    # Zero out the old matrices, we'll swap them on return and
    # these will become the new "next"
    prev_p_z_given_d[:] = 0.0

    return next_p_z_given_d, prev_p_z_given_d


@numba.njit(locals={"e_step_thresh": numba.types.float32,}, fastmath=True, nogil=True)
def plsa_refit_inner_blockwise(
    X_rows,
    X_cols,
    X_vals,
    topics,
    p_z_given_d,
    sample_weight,
    block_size=65536,
    n_iter=50,
    n_iter_per_test=10,
    tolerance=0.005,
    e_step_thresh=1e-32,
):
    """Optimized routine for refitting values of P(z|d) given a fixed set of topics (
    i.e. P(w|z)). This allows fitting document vectors to a predefined set of topics
    (given, for example, by an ensemble result).

    This routine is optimized to work with sparse matrices and only compute values
    for w, d such that X_{w,d} is non-zero.

    To make this numba compilable the raw arrays defining the COO format sparse
    matrix must be passed separately.

    Parameters
    ----------
    X_rows: array of shape (nnz,)
        For each non-zero entry of X, the row of the entry.

    X_cols: array of shape (nnz,)
        For each non-zero entry of X, the column of the
        entry.

    X_vals: array of shape (nnz,)
        For each non-zero entry of X, the value of entry.

    topics: array of shape (n_topics, n_words)
        The fixed topics against which to fit the values of P(z|d).

    p_z_given_d: array of shape (n_docs, n_topics)
        The current estimates of values for P(z|d)

    sample_weight: array of shape (n_docs,)
        Input document weights.

    block_size: int (optional, default=65536)
        The number of nonzero entries of X to process in a block. The larger this
        value the faster the compute may go, but at higher memory cost.

    n_iter: int
        The maximum number iterations of EM to perform

    n_iter_per_test: int
        The number of iterations between tests for relative improvement in
        log-likelihood.

    tolerance: float
        The threshold of relative improvement in log-likelihood required to continue
        iterations.

    e_step_thresh: float (optional, default=1e-32)
        Option to promote sparsity. If the value of P(w|z)P(z|d) in the E step falls
        below threshold then write a zero for P(z|w,d).

    Returns
    -------
    p_z_given_d, p_w_given_z: arrays of shapes (n_docs, n_topics) and (n_topics, n_words)
        The resulting model values of P(z|d) and P(w|z)

    """
    k = topics.shape[0]
    p_z_given_wd_block = np.zeros((block_size, k), dtype=np.float32)

    norm_pdz = np.zeros(p_z_given_d.shape[0], dtype=np.float32)

    previous_log_likelihood = log_likelihood(
        X_rows, X_cols, X_vals, topics, p_z_given_d
    )

    next_p_z_given_d = np.zeros_like(p_z_given_d)

    for i in range(n_iter):

        p_z_given_d, next_p_z_given_d = plsa_refit_em_step(
            X_rows,
            X_cols,
            X_vals,
            topics,
            p_z_given_d,
            next_p_z_given_d,
            p_z_given_wd_block,
            sample_weight,
            norm_pdz,
        )

        if i % n_iter_per_test == 0:
            current_log_likelihood = log_likelihood(
                X_rows, X_cols, X_vals, topics, p_z_given_d, sample_weight,
            )
            if current_log_likelihood > 0:
                change = np.abs(current_log_likelihood - previous_log_likelihood)
                if change / np.abs(current_log_likelihood) < tolerance:
                    break
                else:
                    previous_log_likelihood = current_log_likelihood

    return p_z_given_d

def plsa_refit(
    X,
    topics,
    sample_weight,
    block_size=65536,
    n_iter=50,
    n_iter_per_test=10,
    tolerance=0.005,
    e_step_thresh=1e-32,
    random_state=None,
):
    """Routine for refitting values of P(z|d) given a fixed set of topics (
    i.e. P(w|z)). This allows fitting document vectors to a predefined set of topics
    (given, for example, by an ensemble result).

    Parameters
    ----------
    X: sparse matrix of shape (n_docs, n_words)
        The data matrix pLSA is attempting to fit to.

    topics: array of shape (n_topics, n_words)
        The fixed topics against which to fit the values of P(z|d).

    sample_weight: array of shape (n_docs,)
        Input document weights.

    block_size: int (optional, default=65536)
        The number of nonzero entries of X to process in a block. The larger this
        value the faster the compute may go, but at higher memory cost.

    n_iter: int
        The maximum number iterations of EM to perform

    n_iter_per_test: int
        The number of iterations between tests for relative improvement in
        log-likelihood.

    tolerance: float
        The threshold of relative improvement in log-likelihood required to continue
        iterations.

    e_step_thresh: float (optional, default=1e-32)
        Option to promote sparsity. If the value of P(w|z)P(z|d) in the E step falls
        below threshold then write a zero for P(z|w,d).

    random_state: int, RandomState instance or None, (optional, default: None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. Used in in initialization.

    Returns
    -------
    p_z_given_d, p_w_given_z: arrays of shapes (n_docs, n_topics) and (n_topics, n_words)
        The resulting model values of P(z|d) and P(w|z)

    """
    A = X.tocoo().astype(np.float32)
    k = topics.shape[0]

    rng = check_random_state(random_state)
    p_z_given_d = rng.rand(A.shape[0], k)
    normalize(p_z_given_d, axis=1)
    p_z_given_d = p_z_given_d.astype(np.float32)
    topics = topics.astype(np.float32)

    p_z_given_d = plsa_refit_inner_blockwise(
        A.row,
        A.col,
        A.data,
        topics,
        p_z_given_d,
        sample_weight,
        block_size=block_size,
        n_iter=n_iter,
        n_iter_per_test=n_iter_per_test,
        tolerance=tolerance,
        e_step_thresh=e_step_thresh,
    )

    return p_z_given_d


class StreamedPLSA(BaseEstimator, TransformerMixin):
    """Probabilistic Latent Semantic Analysis (pLSA)

    Given a bag-of-words matrix representation of a corpus of documents, where each row of the
    matrix represents a document, and the jth element of the ith row is the count of the number of
    times the jth vocabulary word occurs in the ith document, estimate matrices of conditional
    probabilities P(z|d) and P(w|z) such that the product matrix of probabilities P(w|d)
    maximises the likelihood of seeing the observed corpus data. Here P(z|d) represents the
    probability of topic z given document d, P(w|z) represents the probability of word w given
    topic z, and P(w|d) represents the probability of word w given document d.

    The algorithm proceeds using an Expectation-Maximization (EM) approach to attempt to maximise
    the likelihood of the observed data under the estimated model.

    The StreamedPLSA uses a block based approached to compute partial E-step M-step
    pairs to lower overall memory usage. This is particularly useful for very large
    training data and/or large numbers of topics.

    Parameters
    ----------
    n_components: int (optional, default=10)
        The number of topics to use in the matrix factorization.

    init: string or tuple (optional, default="random")
        The intialization method to use. This should be one of:
            * ``"random"``
            * ``"nndsvd"``
            * ``"nmf"``
        or a tuple of two ndarrays of shape (n_docs, n_topics) and (n_topics, n_words).

    block_size: int (optional, default=65536)
        The number of nonzero entries of X to process in a block. The larger this
        value the faster the compute may go, but at higher memory cost.

    n_iter: int
        The maximum number iterations of EM to perform

    n_iter_per_test: int
        The number of iterations between tests for relative improvement in
        log-likelihood.

    tolerance: float
        The threshold of relative improvement in log-likelihood required to continue
        iterations.

    e_step_thresh: float (optional, default=1e-32)
        Option to promote sparsity. If the value of P(w|z)P(z|d) in the E step falls
        below threshold then write a zero for P(z|w,d).

    random_state: int, RandomState instance or None, (optional, default: None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. Used in in initialization.

    Attributes
    ----------

    components_: array of shape (n_topics, n_words)
        The topic vectors produced by pLSA. Each row is a topic, which is a probability
        distribution, over the vocabulary, giving the probability of each word given the topic (
        P(w|z)).

    embedding_: array of shape (n_docs, n_topics)
        The document vectors produced by pLSA. Each row corresponds to a document, giving a
        probability distribution, over the topic space, specifying the probability of each topic
        occuring in the document (P(z|d)).

    training_data_: sparse matrix of shape (n_docs, n_words)
        The original training data saved in sparse matrix format.

    References
    ----------

    Hofmann, Thomas. "Probabilistic latent semantic analysis." Proceedings of the Fifteenth
    conference on Uncertainty in artificial intelligence. Morgan Kaufmann Publishers Inc., 1999.

    Hofmann, Thomas. "Unsupervised learning by probabilistic latent semantic analysis."
    Machine learning 42.1-2 (2001): 177-196.

    """

    def __init__(
        self,
        n_components=10,
        init="random",
        block_size=65536,
        n_iter=100,
        n_iter_per_test=10,
        tolerance=0.001,
        e_step_thresh=1e-32,
        transform_random_seed=42,
        random_state=None,
    ):

        self.n_components = n_components
        self.init = init
        self.block_size = block_size
        self.n_iter = n_iter
        self.n_iter_per_test = n_iter_per_test
        self.tolerance = tolerance
        self.e_step_thresh = e_step_thresh
        self.transform_random_seed = transform_random_seed
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        """Learn the pLSA model for the data X and return the document vectors.

        This is more efficient than calling fit followed by transform.

        Parameters
        ----------
        X: array or sparse matrix of shape (n_docs, n_words)
            The data matrix pLSA is attempting to fit to.

        y: Ignored

        sample_weight: array of shape (n_docs,)
            Input document weights.

        Returns
        -------
        self
        """
        self.fit_transform(X, sample_weight=sample_weight)
        return self

    def fit_transform(self, X, y=None, sample_weight=None):
        """Learn the pLSA model for the data X and return the document vectors.

        This is more efficient than calling fit followed by transform.

        Parameters
        ----------
        X: array or sparse matrix of shape (n_docs, n_words)
            The data matrix pLSA is attempting to fit to.

        y: Ignored

        sample_weight: array of shape (n_docs,)
            Input document weights.

        Returns
        -------
        embedding: array of shape (n_docs, n_topics)
            An embedding of the documents into a topic space.
        """

        X = check_array(X, accept_sparse="csr")

        if not issparse(X):
            X = csr_matrix(X)

        sample_weight = _check_sample_weight(
            sample_weight, X, dtype=np.float32)

        if np.any(X.data < 0):
            raise ValueError(
                "PLSA is only valid for matrices with non-negative " "entries"
            )

        row_sums = np.array(X.sum(axis=1).T)[0]
        good_rows = row_sums != 0

        if not np.all(good_rows):
            zero_rows_found = True
            data_for_fitting = X[good_rows]
        else:
            zero_rows_found = False
            data_for_fitting = X

        U, V = plsa_fit(
            data_for_fitting,
            self.n_components,
            sample_weight,
            init=self.init,
            block_size=self.block_size,
            n_iter=self.n_iter,
            n_iter_per_test=self.n_iter_per_test,
            tolerance=self.tolerance,
            e_step_thresh=self.e_step_thresh,
            random_state=self.random_state,
        )

        if zero_rows_found:
            self.embedding_ = np.zeros((X.shape[0], self.n_components))
            self.embedding_[good_rows] = U
        else:
            self.embedding_ = U

        self.components_ = V
        self.training_data_ = X

        return U

    def transform(self, X, y=None, sample_weight=None):
        """Transform the data X into the topic space of the fitted pLSA model.

        Parameters
        ----------
        X: array or sparse matrix of shape (n_docs, n_words)
            Corpus to be embedded into topic space

        y: Ignored

        Returns
        -------
        embedding: array of shape (n_docs, n_topics)
            An embedding of the documents X into the topic space.
        """
        X = check_array(X, accept_sparse="csr")
        sample_weight = _check_sample_weight(
            sample_weight, X, dtype=np.float32)
        random_state = check_random_state(self.transform_random_seed)

        if not issparse(X):
            X = coo_matrix(X)
        else:
            X = X.tocoo()

        result = plsa_refit(
            X,
            self.components_,
            sample_weight,
            block_size=self.block_size,
            n_iter=50,
            n_iter_per_test=5,
            tolerance=0.001,
            random_state=random_state,
        )

        return result

    def coherence(self, topic_num=None, n_words=20):
        """Compute the average coherence of fitted topics, or of a single individual topic.

        Parameters
        ----------
        topic_num: int (optional, default=None)
            The topic number to compute coherence for. If ``topic_num`` is None then the average
            coherence over all topics will be computed.

        n_words int (optional, default=20)
            The number of topic words to score against. The top ``n_words`` words from the selected
            topic will be used.

        Returns
        -------
        topic_coherence: float
            The requested coherence score.
        """

        # Test for errors
        if not isinstance(topic_num, int) and topic_num is not None:
            raise ValueError("Topic number must be an integer or None.")

        if topic_num is None:
            return mean_coherence(self.components_, self.training_data_, n_words)
        elif topic_num >= 0 and topic_num < self.n_components:
            return coherence(self.components_, topic_num, self.training_data_, n_words)
        else:
            raise ValueError(
                "Topic number must be in range 0 to {}".format(self.n_components)
            )

    def log_lift(self, topic_num=None, n_words=20):
        """Compute the average log lift of fitted topics, or of a single individual topic.

        Parameters
        ----------
        topic_num: int (optional, default=None)
            The topic number to compute log lift for. If ``topic_num`` is None then the average
            log lift over all topics will be computed.

        n_words int (optional, default=20)
            The number of topic words to score against. The top ``n_words`` words from the selected
            topic will be used.


        Returns
        -------
        log_lift: float
            The requested log lift score.
        """

        # Test for errors
        if not isinstance(topic_num, int) and topic_num is not None:
            raise ValueError("Topic number must be an integer or None.")

        if topic_num is None:
            return mean_log_lift(self.components_, self.training_data_, n_words)
        elif topic_num >= 0 and topic_num < self.n_components:
            return log_lift(self.components_, topic_num, self.training_data_, n_words)
        else:
            raise ValueError(
                "Topic number must be in range 0 to {}".format(self.n_components)
            )
