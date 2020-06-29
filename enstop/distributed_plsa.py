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
