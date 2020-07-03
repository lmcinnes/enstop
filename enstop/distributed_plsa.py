import numpy as np
import numba

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array, check_random_state
from scipy.sparse import issparse, csr_matrix, coo_matrix

from enstop.utils import normalize, coherence, mean_coherence, log_lift, mean_log_lift
from enstop.plsa import plsa_init
from enstop.block_parallel_plsa import (
    plsa_e_step_on_a_block,
    plsa_partial_m_step_on_a_block,
    log_likelihood_by_blocks,
)

from dask import delayed
import dask.array as da


@delayed
@numba.njit(nogil=True, fastmath=True)
def plsa_em_step_block_kernel(
    row_block, col_block, val_block, p_w_given_z, p_z_given_d, e_step_thresh=1e-32,
):
    result_p_w_given_z = np.zeros_like(p_w_given_z)
    result_p_z_given_d = np.zeros_like(p_z_given_d)
    result_norm_pwz = np.zeros(p_w_given_z.shape[0], dtype=np.float32)
    result_norm_pdz = np.zeros(p_z_given_d.shape[0], dtype=np.float32)
    p_z_given_wd_block = np.zeros(
        (row_block.shape[0], p_w_given_z.shape[0]), dtype=np.float32
    )

    plsa_e_step_on_a_block(
        row_block,
        col_block,
        p_w_given_z,
        p_z_given_d,
        p_z_given_wd_block,
        e_step_thresh,
    )
    plsa_partial_m_step_on_a_block(
        row_block,
        col_block,
        val_block,
        result_p_w_given_z,
        result_p_z_given_d,
        p_z_given_wd_block,
        result_norm_pwz,
        result_norm_pdz,
    )

    return result_p_w_given_z, result_p_z_given_d, result_norm_pwz, result_norm_pdz


def plsa_em_step_dask(
    block_rows_ndarray,
    block_cols_ndarray,
    block_vals_ndarray,
    p_w_given_z,
    p_z_given_d,
    block_row_size,
    block_col_size,
    e_step_thresh=1e-32,
):
    n_d_blocks = block_rows_ndarray.shape[0]
    n_w_blocks = block_rows_ndarray.shape[1]

    n = p_z_given_d.shape[0]
    m = p_w_given_z.shape[1]
    k = p_z_given_d.shape[1]

    result_p_w_given_z = [[] for i in range(n_d_blocks)]
    result_p_z_given_d = [[] for i in range(n_w_blocks)]
    result_norm_pwz = []
    result_norm_pdz = [[] for i in range(n_d_blocks)]

    for i in numba.prange(n_d_blocks):

        row_start = block_row_size * i
        row_end = min(row_start + block_row_size, n)

        for j in range(n_w_blocks):
            col_start = block_col_size * j
            col_end = min(col_start + block_col_size, m)

            row_block = block_rows_ndarray[i, j]
            col_block = block_cols_ndarray[i, j]
            val_block = block_vals_ndarray[i, j]

            kernel_results = plsa_em_step_block_kernel(
                row_block,
                col_block,
                val_block,
                p_w_given_z[:, col_start:col_end],
                p_z_given_d[row_start:row_end, :],
                e_step_thresh=e_step_thresh,
            )

            result_p_w_given_z[j].append(
                da.from_delayed(
                    kernel_results[0], (k, block_col_size), dtype=np.float32
                )
            )
            result_p_z_given_d[i].append(
                da.from_delayed(
                    kernel_results[1], (block_row_size, k), dtype=np.float32
                )
            )
            result_norm_pwz.append(
                da.from_delayed(kernel_results[2], (k,), dtype=np.float32)
            )

            result_norm_pdz[i].append(
                da.from_delayed(kernel_results[3], (block_row_size,), dtype=np.float32)
            )

    p_w_given_z_blocks = [
        da.dstack(result_p_w_given_z[i]).sum(axis=-1) for i in range(n_w_blocks)
    ]
    p_z_given_d_blocks = [
        da.dstack(result_p_z_given_d[i]).sum(axis=-1) for i in range(n_d_blocks)
    ]
    norm_pdz_blocks = [
        da.dstack(result_norm_pdz[i]).sum(axis=-1) for i in range(n_d_blocks)
    ]

    p_w_given_z = (
        da.hstack(p_w_given_z_blocks) / da.dstack(result_norm_pwz).sum(axis=-1).T
    )
    p_z_given_d = da.vstack(p_z_given_d_blocks) / da.hstack(norm_pdz_blocks).T

    return p_w_given_z.compute(), p_z_given_d.compute()


def plsa_fit_inner_dask(
    block_rows_ndarray,
    block_cols_ndarray,
    block_vals_ndarray,
    p_w_given_z,
    p_z_given_d,
    block_row_size,
    block_col_size,
    n_iter=100,
    n_iter_per_test=10,
    tolerance=0.001,
    e_step_thresh=1e-32,
):
    previous_log_likelihood = log_likelihood_by_blocks(
        block_rows_ndarray,
        block_cols_ndarray,
        block_vals_ndarray,
        p_w_given_z,
        p_z_given_d,
        block_row_size,
        block_col_size,
    )

    for i in range(n_iter):
        p_w_given_z, p_z_given_d = plsa_em_step_dask(
            block_rows_ndarray,
            block_cols_ndarray,
            block_vals_ndarray,
            p_w_given_z,
            p_z_given_d,
            block_row_size,
            block_col_size,
            e_step_thresh=e_step_thresh,
        )
        if i % n_iter_per_test == 0:
            current_log_likelihood = log_likelihood_by_blocks(
                block_rows_ndarray,
                block_cols_ndarray,
                block_vals_ndarray,
                p_w_given_z,
                p_z_given_d,
                block_row_size,
                block_col_size,
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
    p_z_given_d_init, p_w_given_z_init = plsa_init(X, k, init=init, rng=rng)

    A = X.tocsr().astype(np.float32)

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

    p_z_given_d, p_w_given_z = plsa_fit_inner_dask(
        block_rows_ndarray,
        block_cols_ndarray,
        block_vals_ndarray,
        p_w_given_z,
        p_z_given_d,
        block_row_size,
        block_col_size,
        n_iter=n_iter,
        n_iter_per_test=n_iter_per_test,
        tolerance=tolerance,
        e_step_thresh=e_step_thresh,
    )

    return p_z_given_d, p_w_given_z


class DistributedPLSA(BaseEstimator, TransformerMixin):
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

        if not issparse(X):
            X = csr_matrix(X)

        if sample_weight is not None:
            NotImplementedError("Sample weights not supported in distributed")
        # sample_weight = _check_sample_weight(sample_weight, X, dtype=np.float32)

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
