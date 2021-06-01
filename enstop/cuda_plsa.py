import numpy as np
import numba
import numba.cuda as cuda

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import _check_sample_weight
from scipy.sparse import issparse, csr_matrix, coo_matrix

from enstop.utils import (
    normalize,
    coherence,
    mean_coherence,
    log_lift,
    mean_log_lift,
    standardize_input,
)
from enstop.plsa import plsa_init
from enstop.block_parallel_plsa import log_likelihood_by_blocks


@cuda.jit()
def plsa_e_step(
    block_rows,
    block_cols,
    p_w_given_z_block,
    p_z_given_d_block,
    p_z_given_wd_block,
    e_step_thresh,
):
    i = cuda.blockIdx.x
    j = cuda.blockIdx.y
    nz_offset = cuda.threadIdx.x
    threads_per_blocks = cuda.blockDim.x
    k = p_z_given_d_block[i].shape[1]
    nnz = block_rows.shape[2]

    n_passes = (nnz // threads_per_blocks) + 1

    for n in range(n_passes):
        nz_idx = n * threads_per_blocks + nz_offset
        if nz_idx < nnz:
            if block_rows[i, j, nz_idx] < 0:
                break

            d = block_rows[i, j, nz_idx]
            w = block_cols[i, j, nz_idx]

            norm = 0.0
            for z in range(k):
                v = p_w_given_z_block[j, z, w] * p_z_given_d_block[i, d, z]
                if v > e_step_thresh:
                    p_z_given_wd_block[i, j, nz_idx, z] = v
                    norm += v
                else:
                    p_z_given_wd_block[i, j, nz_idx, z] = 0.0

            for z in range(k):
                if norm > 0.0:
                    p_z_given_wd_block[i, j, nz_idx, z] /= norm


@cuda.jit()
def plsa_partial_m_step(
    block_rows,
    block_cols,
    block_vals,
    p_w_given_z_block,
    p_z_given_d_block,
    result_p_w_given_z_block,
    result_p_z_given_d_block,
    p_z_given_wd_block,
    pwz_norms,
):
    z = cuda.threadIdx.x
    i = cuda.blockIdx.x
    j = cuda.blockIdx.y
    k = p_z_given_d_block[i].shape[1]
    nnz = block_rows.shape[2]

    if z < k:

        result_p_w_given_z_block[i, j, z, :] = 0.0
        result_p_z_given_d_block[j, i, :, z] = 0.0
        pwz_norms[i, j, z] = 0.0

        for nz_idx in range(block_rows[i, j].shape[0]):
            if block_rows[i, j, nz_idx] < 0:
                break

            d = block_rows[i, j, nz_idx]
            w = block_cols[i, j, nz_idx]
            x = block_vals[i, j, nz_idx]

            s = x * p_z_given_wd_block[i, j, nz_idx, z]

            result_p_w_given_z_block[i, j, z, w] += s
            result_p_z_given_d_block[j, i, d, z] += s

            pwz_norms[i, j, z] += s


@cuda.jit()
def normalize_m_step_p_z_given_d(blocked_next_p_z_given_d, p_z_given_d):
    d_offset = cuda.threadIdx.x
    i = cuda.blockIdx.x
    threads_per_block = cuda.blockDim.x
    k = p_z_given_d[i].shape[1]
    n_passes = ((p_z_given_d.shape[0] * p_z_given_d.shape[1]) // threads_per_block) + 1

    for n in range(n_passes):
        d = threads_per_block * n + d_offset
        if (
            i < blocked_next_p_z_given_d.shape[1]
            and d < blocked_next_p_z_given_d.shape[2]
        ):
            norm = 0.0
            for z in range(k):
                p_z_given_d[i, d, z] = 0.0
                for j in range(blocked_next_p_z_given_d.shape[0]):
                    p_z_given_d[i, d, z] += blocked_next_p_z_given_d[j, i, d, z]
                    norm += blocked_next_p_z_given_d[j, i, d, z]
            for z in range(k):
                p_z_given_d[i, d, z] /= norm


@cuda.jit()
def normalize_m_step_p_w_given_z(blocked_next_p_w_given_z, p_w_given_z, pwz_norms):
    w_offset = cuda.threadIdx.x
    i = cuda.blockIdx.x
    threads_per_block = cuda.blockDim.x
    k = p_w_given_z[i].shape[0]
    n_passes = ((p_w_given_z.shape[0] * p_w_given_z.shape[2]) // threads_per_block) + 1

    norms = cuda.local.array(1024, numba.float64)
    for z in range(k):
        norms[z] = 0.0
    for p in range(pwz_norms.shape[0]):
        for q in range(pwz_norms.shape[1]):
            for z in range(k):
                norms[z] += pwz_norms[p, q, z]

    for n in range(n_passes):
        w = n * threads_per_block + w_offset
        if (
            i < blocked_next_p_w_given_z.shape[1]
            and w < blocked_next_p_w_given_z.shape[3]
        ):
            for z in range(k):
                p_w_given_z[i, z, w] = 0.0
                for j in range(blocked_next_p_w_given_z.shape[0]):
                    p_w_given_z[i, z, w] += blocked_next_p_w_given_z[j, i, z, w]
            for z in range(k):
                p_w_given_z[i, z, w] /= norms[z]


def plsa_fit(
    data,
    k,
    n_row_blocks=8,
    n_col_blocks=8,
    init="random",
    n_iter=100,
    n_iter_per_test=10,
    tolerance=0.001,
    e_step_thresh=1e-32,
    random_state=None,
):
    rng = check_random_state(random_state)
    p_z_given_d_init, p_w_given_z_init = plsa_init(data, k, init=init, rng=rng)

    A = data.tocsr().astype(np.float32)

    n = A.shape[0]
    m = A.shape[1]

    block_row_size = np.uint16(np.ceil(A.shape[0] / n_row_blocks))
    block_col_size = np.uint16(np.ceil(A.shape[1] / n_col_blocks))

    p_z_given_d = np.zeros((block_row_size * n_row_blocks, k), dtype=np.float32)
    p_z_given_d[: p_z_given_d_init.shape[0]] = p_z_given_d_init
    p_z_given_d = p_z_given_d.reshape(n_row_blocks, block_row_size, k)

    p_w_given_z = np.zeros((k, block_col_size * n_col_blocks), dtype=np.float32)
    p_w_given_z[:, : p_w_given_z_init.shape[1]] = p_w_given_z_init
    p_w_given_z = np.transpose(
        p_w_given_z.T.reshape(n_col_blocks, block_col_size, k), axes=[0, 2, 1]
    ).astype(np.float32, order="C")

    A_blocks = [[0] * n_col_blocks for i in range(n_row_blocks)]
    max_nnz_per_block = 0
    for i in range(n_row_blocks):

        row_start = block_row_size * i
        row_end = min(row_start + block_row_size, n)

        for j in range(n_col_blocks):

            col_start = block_col_size * j
            col_end = min(col_start + block_col_size, m)

            A_blocks[i][j] = A[row_start:row_end, col_start:col_end].tocoo()
            if A_blocks[i][j].nnz > max_nnz_per_block:
                max_nnz_per_block = A_blocks[i][j].nnz

    block_rows_ndarray = np.full(
        (n_row_blocks, n_col_blocks, max_nnz_per_block), -1, dtype=np.int32
    )
    block_cols_ndarray = np.full(
        (n_row_blocks, n_col_blocks, max_nnz_per_block), -1, dtype=np.int32
    )
    block_vals_ndarray = np.zeros(
        (n_row_blocks, n_col_blocks, max_nnz_per_block), dtype=np.float32
    )
    for i in range(n_row_blocks):
        for j in range(n_col_blocks):
            nnz = A_blocks[i][j].nnz
            block_rows_ndarray[i, j, :nnz] = A_blocks[i][j].row
            block_cols_ndarray[i, j, :nnz] = A_blocks[i][j].col
            block_vals_ndarray[i, j, :nnz] = A_blocks[i][j].data

    n_d_blocks = block_rows_ndarray.shape[0]
    n_w_blocks = block_rows_ndarray.shape[1]
    block_size = block_rows_ndarray.shape[2]

    p_z_given_wd_block = np.zeros(
        (n_d_blocks, n_w_blocks, block_size, k), dtype=np.float32
    )

    blocked_next_p_w_given_z = np.zeros(
        (
            np.int64(n_d_blocks),
            np.int64(n_w_blocks),
            np.int64(k),
            np.int64(block_col_size),
        ),
        dtype=np.float32,
    )
    blocked_next_p_z_given_d = np.zeros(
        (
            np.int64(n_w_blocks),
            np.int64(n_d_blocks),
            np.int64(block_row_size),
            np.int64(k),
        ),
        dtype=np.float32,
    )
    norms_pwz = np.zeros((n_d_blocks, n_w_blocks, k), dtype=np.float64)

    previous_log_likelihood = log_likelihood_by_blocks(
        block_rows_ndarray,
        block_cols_ndarray,
        block_vals_ndarray,
        p_w_given_z,
        p_z_given_d,
    )

    d_block_rows_ndarray = cuda.to_device(block_rows_ndarray)
    d_block_cols_ndarray = cuda.to_device(block_cols_ndarray)
    d_block_vals_ndarray = cuda.to_device(block_vals_ndarray)
    d_blocked_next_p_w_given_z = cuda.to_device(blocked_next_p_w_given_z)
    d_blocked_next_p_z_given_d = cuda.to_device(blocked_next_p_z_given_d)
    d_p_z_given_wd_block = cuda.to_device(p_z_given_wd_block)
    d_p_w_given_z = cuda.to_device(p_w_given_z)
    d_p_z_given_d = cuda.to_device(p_z_given_d)
    d_norms_pwz = cuda.to_device(norms_pwz)

    n_d = p_z_given_d.shape[1]
    n_w = p_w_given_z.shape[2]

    for i in range(n_iter // n_iter_per_test):
        for j in range(n_iter_per_test):
            plsa_e_step[(n_d_blocks, n_w_blocks), 256](
                d_block_rows_ndarray,
                d_block_cols_ndarray,
                d_p_w_given_z,
                d_p_z_given_d,
                d_p_z_given_wd_block,
                e_step_thresh,
            )
            cuda.synchronize()
            plsa_partial_m_step[(n_d_blocks, n_w_blocks), k](
                d_block_rows_ndarray,
                d_block_cols_ndarray,
                d_block_vals_ndarray,
                d_p_w_given_z,
                d_p_z_given_d,
                d_blocked_next_p_w_given_z,
                d_blocked_next_p_z_given_d,
                d_p_z_given_wd_block,
                d_norms_pwz,
            )
            cuda.synchronize()
            normalize_m_step_p_z_given_d[n_d_blocks, 256](
                d_blocked_next_p_z_given_d, d_p_z_given_d
            )
            normalize_m_step_p_w_given_z[n_w_blocks, 256](
                d_blocked_next_p_w_given_z, d_p_w_given_z, d_norms_pwz
            )
            cuda.synchronize()

        p_z_given_d = d_p_z_given_d.copy_to_host()
        p_w_given_z = d_p_w_given_z.copy_to_host()
        current_log_likelihood = log_likelihood_by_blocks(
            block_rows_ndarray,
            block_cols_ndarray,
            block_vals_ndarray,
            p_w_given_z,
            p_z_given_d,
        )
        change = np.abs(current_log_likelihood - previous_log_likelihood)
        if change / np.abs(current_log_likelihood) < tolerance:
            break
        else:
            previous_log_likelihood = current_log_likelihood

    for i in range(n_iter % n_iter_per_test):
        plsa_e_step[(n_d_blocks, n_w_blocks), 256](
            d_block_rows_ndarray,
            d_block_cols_ndarray,
            d_p_w_given_z,
            d_p_z_given_d,
            d_p_z_given_wd_block,
            e_step_thresh,
        )
        cuda.synchronize()
        plsa_partial_m_step[(n_d_blocks, n_w_blocks), k](
            d_block_rows_ndarray,
            d_block_cols_ndarray,
            d_block_vals_ndarray,
            d_p_w_given_z,
            d_p_z_given_d,
            d_blocked_next_p_w_given_z,
            d_blocked_next_p_z_given_d,
            d_p_z_given_wd_block,
            d_norms_pwz,
        )
        cuda.synchronize()
        normalize_m_step_p_z_given_d[n_d_blocks, 256](
            d_blocked_next_p_z_given_d, d_p_z_given_d
        )
        normalize_m_step_p_w_given_z[n_w_blocks, 256](
            d_blocked_next_p_w_given_z, d_p_w_given_z, d_norms_pwz
        )
        cuda.synchronize()

    p_z_given_d = d_p_z_given_d.copy_to_host()
    p_w_given_z = d_p_w_given_z.copy_to_host()

    p_z_given_d = np.vstack(p_z_given_d)[:n, :]
    p_w_given_z = np.hstack(p_w_given_z)[:, :m]

    return p_z_given_d, p_w_given_z


class GPUPLSA(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_components=10,
        init="random",
        n_row_blocks=8,
        n_col_blocks=8,
        n_iter=100,
        n_iter_per_test=10,
        tolerance=0.001,
        e_step_thresh=1e-32,
        transform_random_seed=42,
        random_state=None,
    ):

        self.n_components = n_components
        self.init = init
        self.n_row_blocks = n_row_blocks
        self.n_col_blocks = n_col_blocks
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
        X = standardize_input(X)

        if not issparse(X):
            X = csr_matrix(X)

        sample_weight = _check_sample_weight(sample_weight, X, dtype=np.float32)

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
            n_row_blocks=self.n_row_blocks,
            n_col_blocks=self.n_col_blocks,
            init=self.init,
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

        return self.embedding_
