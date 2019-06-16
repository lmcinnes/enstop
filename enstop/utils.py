import numpy as np
import numba
from scipy.sparse import issparse, csc_matrix

@numba.njit(fastmath=True, nogil=True)
def normalize(ndarray, axis=0):
    """Normalize an array with respect to the l1-norm
    along an axis. Note that this procedure modifies
    the array **in place**.

    Parameters
    ----------
    ndarray: array of shape (n,m)
        The array to be normalized. Must be a 2D array.

    axis: int (optional, default=0)
        The axis to normalize with respect to. 0 means
        normalize columns, 1 means normalize rows.
    """
    # Compute marginal sum along axis
    marginal = np.zeros(ndarray.shape[1 - axis])
    for i in range(marginal.shape[0]):
        for j in range(ndarray.shape[axis]):
            if axis == 0:
                marginal[i] += ndarray[j, i]
            elif axis == 1:
                marginal[i] += ndarray[i, j]
            else:
                raise ValueError("axis must be 0 or 1")

    # Divide out by the marginal
    for i in range(marginal.shape[0]):
        for j in range(ndarray.shape[axis]):
            if marginal[i] > 0.0:
                if axis == 0:
                    ndarray[j, i] /= marginal[i]
                elif axis == 1:
                    ndarray[i, j] /= marginal[i]
                else:
                    raise ValueError("axis must be 0 or 1")


@numba.njit()
def _log_lift(topics, z, empirical_probs, n=-1):
    total_lift = 0.0
    if n <= 0:
        for w in range(topics.shape[1]):
            if empirical_probs[w] > 0:
                total_lift += topics[z, w] * 1.0 / empirical_probs[w]
        return np.log(total_lift * 1.0 / topics.shape[1])
    else:
        top_words = np.argsort(topics[z])[-n:]
        for i in range(n):
            w = top_words[i]
            if empirical_probs[w] > 0:
                total_lift += topics[z, w] * 1.0 / empirical_probs[w]
        return np.log(total_lift * 1.0 / n)


def log_lift(topics, z, data, n_words=-1):
    normalized_topics = topics.copy()
    normalize(normalized_topics, axis=1)
    empirical_probs = np.array(data.sum(axis=0)).squeeze().astype(np.float64)
    empirical_probs /= empirical_probs.sum()
    return _log_lift(normalized_topics, z, empirical_probs, n=n_words)


def mean_log_lift(topics, data, n_words=-1):
    normalized_topics = topics.copy()
    normalize(normalized_topics, axis=1)
    empirical_probs = np.array(data.sum(axis=0)).squeeze().astype(np.float64)
    empirical_probs /= empirical_probs.sum()
    return np.mean(
        [
            _log_lift(topics, z, empirical_probs, n=n_words)
            for z in range(topics.shape[0])
        ]
    )


@numba.njit()
def arr_intersect(ar1, ar2):
    aux = np.concatenate((ar1, ar2))
    aux.sort()
    return aux[:-1][aux[1:] == aux[:-1]]


@numba.njit()
def _coherence(topics, z, n, indices, indptr, n_docs_per_word):
    top_words = np.argsort(topics[z])[-n:]
    coherence = 0.0
    for i in range(n - 1):
        w = top_words[i]
        if n_docs_per_word[w] == 0:
            continue
        for j in range(i + 1, n):
            v = top_words[j]
            n_co_occur = arr_intersect(
                indices[indptr[w] : indptr[w + 1]], indices[indptr[v] : indptr[v + 1]]
            ).shape[0]
            coherence += np.log((n_co_occur + 1.0) / n_docs_per_word[w])
    return coherence


def coherence(topics, z, data, n_words=20):
    if not issparse(data):
        csc_data = csc_matrix(data)
    else:
        csc_data = data.tocsc()

    n_docs_per_word = np.array((data > 0).sum(axis=0)).squeeze()
    return _coherence(
        topics, z, n_words, csc_data.indices, csc_data.indptr, n_docs_per_word
    )


def mean_coherence(topics, data, n_words=20):
    if not issparse(data):
        csc_data = csc_matrix(data)
    else:
        csc_data = data.tocsc()

    n_docs_per_word = np.array((data > 0).sum(axis=0)).squeeze()
    return np.mean(
        [
            _coherence(
                topics, z, n_words, csc_data.indices, csc_data.indptr, n_docs_per_word
            )
            for z in range(topics.shape[0])
        ]
    )
