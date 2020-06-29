import numpy as np
import numba
from scipy.sparse import issparse, csc_matrix
from sklearn.utils.validation import check_array
import numbers

@numba.njit(fastmath=True, nogil=True)
def normalize(ndarray, axis=0):
    """Normalize an array with respect to the l1-norm along an axis. Note that this procedure
    modifies the array **in place**.

    Parameters
    ----------
    ndarray: array of shape (n,m)
        The array to be normalized. Must be a 2D array.

    axis: int (optional, default=0)
        The axis to normalize with respect to. 0 means normalize columns, 1 means normalize rows.
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
    """Internal method to compute the log lift given precomputed empirical probabilities. This
    routine is designed to be numba compilable for performance.

    Parameters
    ----------
    topics: array of shape (n_topics, n_words)
        The topic vectors to evaluate.

    z: int
        Which topic vector to evaluate. Must be
        in range(0, n_topics).

    empirical_probs: array of shape (n_words,)
        The empirical probability of word occurrence.

    n: int (optional, default=-1)
        The number of words to average over. If less than 0 it will evaluate over the entire
        vocabulary, otherwise it will select the top ``n`` words of the chosen topic.

    Returns
    -------
    log_lift: float
        The log lift of the ``z``th topic vector.
    """
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
    """Compute the log lift of a single topic given empirical data from which empirical
    probabilities of word occurrence can be computed.

     Parameters
     ----------
     topics: array of shape (n_topics, n_words)
         The topic vectors to evaluate.

     z: int
         Which topic vector to evaluate. Must be
         in range(0, n_topics).

     data: array or sparse matrix of shape (n_docs, n_words,)
         The empirical data of word occurrence in a corpus.

     n: int (optional, default=-1)
         The number of words to average over. If less than 0 it will evaluate over the entire
         vocabulary, otherwise it will select the top ``n`` words of the chosen topic.

     Returns
     -------
     log_lift: float
         The log lift of the ``z``th topic vector.
     """
    normalized_topics = topics.copy()
    normalize(normalized_topics, axis=1)
    empirical_probs = np.array(data.sum(axis=0)).squeeze().astype(np.float64)
    empirical_probs /= empirical_probs.sum()
    return _log_lift(normalized_topics, z, empirical_probs, n=n_words)


def mean_log_lift(topics, data, n_words=-1):
    """Compute the average log lift over all topics given empirical data from which empirical
    probabilities of word occurrence can be computed.

     Parameters
     ----------
     topics: array of shape (n_topics, n_words)
         The topic vectors to evaluate.

     data: array or sparse matrix of shape (n_docs, n_words,)
         The empirical data of word occurrence in a corpus.

     n: int (optional, default=-1)
         The number of words to average over. If less than 0 it will evaluate over the entire
         vocabulary, otherwise it will select the top ``n`` words of the chosen topic.

     Returns
     -------
     log_lift: float
         The average log lift over all topic vectors.
     """
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
    """Numba compilable equivalent of numpy's intersect1d"""
    aux = np.concatenate((ar1, ar2))
    aux.sort()
    return aux[:-1][aux[1:] == aux[:-1]]


@numba.njit()
def _coherence(topics, z, n, indices, indptr, n_docs_per_word):
    """Internal routine for computing the coherence of a given topic given raw data and the
    number of documents per vocabulary word. This routine makes use of scipy sparse matrix
    formats, but to be numba compilable it must make use of internal arrays thereof.

    Parameters
    ----------
    topics: array of shape (n_topics, n_words)
        The topic vectors for scoring

    z: int
        Which topic vector to score.

    n: int
        The number of topic words to score against. The top ``n`` words from the ``z``th topic
        will be used.

    indices: array of shape (nnz,)
        The indices array of a CSC format sparse matrix representation of the corpus data.

    indptr: array of shape(n_words - 1,)
        The indptr array of a CSC format sparse matrix representation of the corpus data.

    n_docs_per_word: array of shape (n_words,)
        The total number of documents for each vocabulary word (the column sum of the corpus data).


    Returns
    -------
    topic_coherence: float
        The coherence score of the ``z``th topic.
    """
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
    """Compute the coherence of a single topic given empirical data.

    Parameters
    ----------
    topics: array of shape (n_topics, n_words)
        The topic vectors for scoring

    z: int
        Which topic vector to score.

    data: array or sparse matrix of shape (n_doc, n_words)
        The empirical data of word occurrence in a corpus.

    n_words: int (optional, default=20)
        The number of topic words to score against. The top ``n_words`` words from the ``z``th topic
        will be used.

    Returns
    -------
    topic_coherence: float
        The coherence score of the ``z``th topic.
    """
    if not issparse(data):
        csc_data = csc_matrix(data)
    else:
        csc_data = data.tocsc()

    n_docs_per_word = np.array((data > 0).sum(axis=0)).squeeze()
    return _coherence(
        topics, z, n_words, csc_data.indices, csc_data.indptr, n_docs_per_word
    )


def mean_coherence(topics, data, n_words=20):
    """Compute the average coherence of all topics given empirical data.

    Parameters
    ----------
    topics: array of shape (n_topics, n_words)
        The topic vectors for scoring

    data: array or sparse matrix of shape (n_doc, n_words)
        The empirical data of word occurrence in a corpus.

    n_words: int (optional, default=20)
        The number of topic words to score against. The top ``n_words`` words of each topic
        will be used.

    Returns
    -------
    topic_coherence: float
        The average coherence score of all the topics.
    """
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

####
# Taken from sklearn as a fallback option; by default we import their latest version
####
def _check_sample_weight(sample_weight, X, dtype=None):
    """Validate sample weights.

    Note that passing sample_weight=None will output an array of ones.
    Therefore, in some cases, you may want to protect the call with:
    if sample_weight is not None:
        sample_weight = _check_sample_weight(...)

    Parameters
    ----------
    sample_weight : {ndarray, Number or None}, shape (n_samples,)
       Input sample weights.

    X : nd-array, list or sparse matrix
        Input data.

    dtype: dtype
       dtype of the validated `sample_weight`.
       If None, and the input `sample_weight` is an array, the dtype of the
       input is preserved; otherwise an array with the default numpy dtype
       is be allocated.  If `dtype` is not one of `float32`, `float64`,
       `None`, the output will be of dtype `float64`.

    Returns
    -------
    sample_weight : ndarray, shape (n_samples,)
       Validated sample weight. It is guaranteed to be "C" contiguous.
    """
    n_samples = X.shape[0]

    if dtype is not None and dtype not in [np.float32, np.float64]:
        dtype = np.float64

    if sample_weight is None:
        sample_weight = np.ones(n_samples, dtype=dtype)
    elif isinstance(sample_weight, numbers.Number):
        sample_weight = np.full(n_samples, sample_weight, dtype=dtype)
    else:
        if dtype is None:
            dtype = [np.float64, np.float32]
        sample_weight = check_array(
            sample_weight, accept_sparse=False, ensure_2d=False, dtype=dtype,
            order="C"
        )
        if sample_weight.ndim != 1:
            raise ValueError("Sample weights must be 1D array or scalar")

        if sample_weight.shape != (n_samples,):
            raise ValueError("sample_weight.shape == {}, expected {}!"
                             .format(sample_weight.shape, (n_samples,)))
    return sample_weight
