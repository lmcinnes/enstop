import numpy as np
import numba
from warnings import warn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.decomposition import NMF, non_negative_factorization
from scipy.sparse import issparse, csr_matrix, coo_matrix
import dask

try:
    import joblib

    _HAVE_JOBLIB = True
except ImportError:
    warn("Joblib could not be loaded; joblib parallelism will not be available")
    _HAVE_JOBLIB = False
from hdbscan._hdbscan_linkage import mst_linkage_core, label
from hdbscan.hdbscan_ import _tree_to_labels
import hdbscan
import umap

# TODO: Once umap 0.4 is released enable this...
# from umap.distances import hellinger

from enstop.utils import normalize, coherence, mean_coherence, log_lift, mean_log_lift
from enstop.plsa import plsa_fit, plsa_refit


def plsa_topics(X, k, **kwargs):
    """Perform a boostrap sample from a corpus of documents and fit the sample using
    pLSA to give a set of topic vectors such that the (z,w) entry of the returned
    array is the probability P(w|z) of word w occuring given the zth topic.

    Parameters
    ----------
    X: sparse matrix of shape (n_docs, n_words)
        The bag of words representation of the corpus of documents.

    k: int
        The number of topics to generate.

    kwargs:
        Further keyword arguments that can be passed on th the ``plsa_fit`` function.
        Possibilities include:
            * ``init``
            * ``n_iter``
            * ``n_iter_per_test``
            * ``tolerance``
            * ``e_step_threshold``

    Returns
    -------
    topics: array of shape (k, n_words)
        The topics generated from the bootstrap sample.
    """
    A = X.tocsr()
    bootstrap_sample_indices = np.random.randint(0, A.shape[0], size=A.shape[0])
    B = A[bootstrap_sample_indices]
    doc_topic, topic_vocab = plsa_fit(
        B,
        k,
        init=kwargs.get("init", "nndsvd"),
        n_iter=kwargs.get("n_iter", 100),
        n_iter_per_test=kwargs.get("n_iter_per_test", 10),
        tolerance=kwargs.get("tolerance", 0.001),
        e_step_thresh=kwargs.get("e_step_thresh", 1e-16),
    )
    return topic_vocab


def nmf_topics(X, k, **kwargs):
    """Perform a boostrap sample from a corpus of documents and fit the sample using
    NMF to give a set of topic vectors, normalized such that the(z,w) entry of the
    returned array is the probability P(w|z) of word w occuring given the zth topic.

    Parameters
    ----------
    X: sparse matrix of shape (n_docs, n_words)
        The bag of words representation of the corpus of documents.

    k: int
        The number of topics to generate.

    kwargs:
        Further keyword arguments that can be passed on th the ``plsa_fit`` function.
        Possibilities include:
            * ``init``
            * ``beta_loss``
            * ``alpha``
            * ``solver``

    Returns
    -------
    topics: array of shape (k, n_words)
        The topics generated from the bootstrap sample.
    """
    A = X.tocsr()
    bootstrap_sample_indices = np.random.randint(0, A.shape[0], size=A.shape[0])
    B = A[bootstrap_sample_indices]
    nmf = NMF(
        n_components=k,
        init=kwargs.get("init", "nndsvd"),
        beta_loss=kwargs.get("beta_loss", 1),
        alpha=kwargs.get("alpha", 0.0),
        solver=kwargs.get("solver", "mu"),
    ).fit(B)
    topics = nmf.components_.copy()
    normalize(topics, axis=1)
    return topics


def ensemble_of_topics(
    X, k, model="plsa", n_jobs=4, n_runs=16, parallelism="dask", **kwargs
):

    if model == "plsa":
        create_topics = plsa_topics
    elif model == "nmf":
        create_topics = nmf_topics
    else:
        raise ValueError('Model must be one of "plsa" or "nmf"')

    if parallelism == "dask":
        dask_topics = dask.delayed(create_topics)
        staged_topics = [dask_topics(X, k, **kwargs) for i in range(n_runs)]
        topics = dask.compute(*staged_topics, scheduler="threads", num_workers=n_jobs)
    elif parallelism == "joblib" and _HAVE_JOBLIB:
        joblib_topics = joblib.delayed(create_topics)
        topics = joblib.Parallel(n_jobs=n_jobs, prefer="threads")(
            joblib_topics(X, k, **kwargs) for i in range(n_runs)
        )
    elif parallelism == "joblib" and not _HAVE_JOBLIB:
        raise ValueError("Joblib was not correctly imported and is unavailable")
    else:
        raise ValueError(
            "Unrecognized parallelism {}; should be one of {}".format(
                parallelism, ("dask", "joblib")
            )
        )

    return np.vstack(topics)


@numba.njit(fastmath=True, nogil=True)
def kl_divergence(a, b):
    result = 0.0
    for i in range(a.shape[0]):
        if a[i] > 0.0 and b[i] > 0.0:
            result += a[i] * (np.log2(a[i]) - np.log2(b[i]))
    return result


@numba.njit(fastmath=True, parallel=True)
def all_pairs_kl_divergence(distributions):
    n = distributions.shape[0]
    result = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            result[i, j] = kl_divergence(distributions[i], distributions[j])
    return result


# TODO: Once umap 0.4 is released enable this...
# @numba.njit(fastmath=True, parallel=True)
# def all_pairs_hellinger_distance(distributions):
#     n = distributions.shape[0]
#     result = np.zeros((n, n))
#     for i in range(n):
#         for j in range(n):
#             result[i, j] = hellinger(distributions[i], distributions[j])
#     return result


def generate_combined_topics_kl(all_topics, min_samples=5, min_cluster_size=5):
    divergence_matrix = all_pairs_kl_divergence(all_topics)
    core_divergences = np.sort(divergence_matrix, axis=1)[:, min_samples]
    tiled_core_divergences = np.tile(core_divergences, (core_divergences.shape[0], 1))
    mutual_reachability = np.dstack(
        [
            divergence_matrix,
            divergence_matrix.T,
            tiled_core_divergences,
            tiled_core_divergences.T,
        ]
    ).max(axis=-1)
    mst_data = mst_linkage_core(mutual_reachability)
    mst_order = np.argsort(mst_data.T[2])
    mst_data = mst_data[mst_order]
    single_linkage_tree = label(mst_data)
    labels, probs, stabs, ctree, stree = _tree_to_labels(
        all_topics,
        single_linkage_tree,
        min_cluster_size=min_cluster_size,
        cluster_selection_method="leaf",
    )
    result = np.empty((labels.max() + 1, all_topics.shape[1]), dtype=np.float32)
    for i in range(labels.max() + 1):
        result[i] = np.mean(np.sqrt(all_topics[labels == i]), axis=0) ** 2
        result[i] /= result[i].sum()

    return result

# TODO: Once umap 0.4 is released enable this...
# def generate_combined_topics_hellinger(all_topics, min_samples=5, min_cluster_size=5):
#     distance_matrix = all_pairs_hellinger_distance(all_topics)
#     labels = hdbscan.HDBSCAN(
#         min_samples=min_samples, min_cluster_size=min_cluster_size, metric="precomputed"
#     ).fit_predict(distance_matrix)
#     result = np.empty((labels.max() + 1, all_topics.shape[1]), dtype=np.float32)
#     for i in range(labels.max() + 1):
#         result[i] = np.mean(np.sqrt(all_topics[labels == i]), axis=0) ** 2
#         result[i] /= result[i].sum()
#
#     return result


# TODO: Once umap 0.4 is released enable hellinger distance...
def generate_combined_topics_hellinger_umap(
    all_topics, min_samples=5, min_cluster_size=5, n_neighbors=15, reduced_dim=5
):
    embedding = umap.UMAP(
        n_neighbors=n_neighbors, n_components=reduced_dim, metric="cosine"
    ).fit_transform(all_topics)
    labels = hdbscan.HDBSCAN(
        min_samples=min_samples, min_cluster_size=min_cluster_size
    ).fit_predict(embedding)
    result = np.empty((labels.max() + 1, all_topics.shape[1]), dtype=np.float32)
    for i in range(labels.max() + 1):
        result[i] = np.mean(np.sqrt(all_topics[labels == i]), axis=0) ** 2
        result[i] /= result[i].sum()

    return result


_topic_combiner = {
    "kl_divergence": generate_combined_topics_kl,
    # "hellinger": generate_combined_topics_hellinger,
    "hellinger_umap": generate_combined_topics_hellinger_umap,
}


def ensemble_fit(
    X,
    estimated_n_topics=10,
    model="plsa",
    init="nndsvd",
    min_samples=3,
    min_cluster_size=5,
    n_starts=16,
    n_jobs=8,
    parallelism="dask",
    topic_combination="hellinger_umap",
    n_iter=100,
    n_iter_per_test=10,
    tolerance=0.001,
    e_step_thresh=1e-16,
    lift_factor=1,
    beta_loss=1,
    alpha=0.0,
    solver="mu",
):

    X = check_array(X, accept_sparse="csr")

    if issparse(X):
        X_coo = X.tocoo()
    else:
        X_coo = coo_matrix(X)

    all_topics = ensemble_of_topics(
        X_coo,
        estimated_n_topics,
        model,
        n_jobs,
        n_starts,
        parallelism,
        init=init,
        n_iter=n_iter,
        n_iter_per_test=n_iter_per_test,
        tolerance=tolerance,
        e_step_thresh=e_step_thresh,
        lift_factor=1,
        beta_loss=beta_loss,
        alpha=alpha,
        solver=solver,
    )

    if topic_combination in _topic_combiner:
        cluster_topics = _topic_combiner[topic_combination]
    else:
        raise ValueError(
            "topic_combination must be one of {}".format(tuple(_topic_combiner.keys()))
        )

    stable_topics = cluster_topics(all_topics, min_samples, min_cluster_size)

    if lift_factor != 1:
        stable_topics **= lift_factor
        normalize(stable_topics, axis=1)

    if model == "plsa":
        doc_vectors = plsa_refit(
            X_coo.row,
            X_coo.col,
            X_coo.data,
            X_coo.shape[0],
            X_coo.shape[1],
            stable_topics,
            e_step_thresh=e_step_thresh,
        )
    elif model == "nmf":
        doc_vectors, _, _ = non_negative_factorization(
            X,
            H=stable_topics,
            n_components=stable_topics.shape[0],
            update_H=False,
            beta_loss=beta_loss,
            alpha=alpha,
            solver=solver,
        )
    else:
        raise ValueError('Model must be one of "plsa" or "nmf"')

    return doc_vectors, stable_topics


class EnsembleTopics(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_components=10,
        model="plsa",
        init="nndsvd",
        n_starts=16,
        min_samples=3,
        min_cluster_size=5,
        n_jobs=8,
        parallelism="dask",
        topic_combination="hellinger_umap",
        n_iter=100,
        n_iter_per_test=10,
        tolerance=0.001,
        e_step_thresh=1e-16,
        lift_factor=1,
        beta_loss=1,
        alpha=0.0,
        solver="mu",
    ):

        self.n_components = n_components
        self.model = model
        self.init = init
        self.n_starts = n_starts
        self.min_samples = min_samples
        self.min_cluster_size = min_cluster_size
        self.n_jobs = n_jobs
        self.parallelism = parallelism
        self.topic_combination = topic_combination
        self.n_iter = n_iter
        self.n_iter_per_test = n_iter_per_test
        self.tolerance = tolerance
        self.e_step_thresh = e_step_thresh
        self.lift_factor = lift_factor
        self.beta_loss = beta_loss
        self.alpha = alpha
        self.solver = solver

    def fit(self, X, y=None):
        self.fit_transform(X)
        return self

    def fit_transform(self, X, y=None):

        X = check_array(X, accept_sparse="csr")

        if not issparse(X):
            X = csr_matrix(X)

        U, V = ensemble_fit(
            X,
            self.n_components,
            self.model,
            self.init,
            self.min_samples,
            self.min_cluster_size,
            self.n_starts,
            self.n_jobs,
            self.parallelism,
            self.topic_combination,
            self.n_iter,
            self.n_iter_per_test,
            self.tolerance,
            self.e_step_thresh,
            self.lift_factor,
            self.beta_loss,
            self.alpha,
            self.solver,
        )
        self.components_ = V
        self.embedding_ = U
        self.training_data_ = X
        self.n_components_ = self.components_.shape[0]

        return U

    def transform(self, X, y=None):
        X = check_array(X, accept_sparse="csr")

        if not issparse(X):
            X = coo_matrix(X)
        else:
            X = X.tocoo()

        n, m = X.shape

        result = plsa_refit(
            X.row,
            X.col,
            X.vals,
            n,
            m,
            self.components_,
            n_iter=50,
            n_iter_per_test=5,
            tolerance=0.001,
        )

        return result

    def coherence(self, topic_num=None, n_words=20):

        # Test for errors
        if not isinstance(topic_num, int) and topic_num is not None:
            raise ValueError("Topic number must be an integer or None.")

        if topic_num is None:
            return mean_coherence(
                self.components_, self.training_data_, n_words=n_words
            )
        elif topic_num >= 0 and topic_num < self.n_components:
            return coherence(
                self.components_, topic_num, self.training_data_, n_words=n_words
            )
        else:
            raise ValueError(
                "Topic number must be in range 0 to {}".format(self.n_components)
            )

    def log_lift(self, topic_num=None, n_words=20):

        # Test for errors
        if not isinstance(topic_num, int) and topic_num is not None:
            raise ValueError("Topic number must be an integer or None.")

        if topic_num is None:
            return mean_log_lift(self.components_, self.training_data_, n_words=n_words)
        elif topic_num >= 0 and topic_num < self.n_components:
            return log_lift(
                self.components_, topic_num, self.training_data_, n_words=n_words
            )
        else:
            raise ValueError(
                "Topic number must be in range 0 to {}".format(self.n_components)
            )
