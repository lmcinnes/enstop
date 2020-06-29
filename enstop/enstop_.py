import numpy as np
import numba
from warnings import warn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array, check_random_state
try:
    from sklearn.utils.validation import _check_sample_weight
except ImportError:
    from enstop.utils import _check_sample_weight
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


@numba.njit()
def hellinger(x, y):
    result = 0.0
    l1_norm_x = 0.0
    l1_norm_y = 0.0

    for i in range(x.shape[0]):
        result += np.sqrt(x[i] * y[i])
        l1_norm_x += x[i]
        l1_norm_y += y[i]

    if l1_norm_x == 0 and l1_norm_y == 0:
        return 0.0
    elif l1_norm_x == 0 or l1_norm_y == 0:
        return 1.0
    else:
        return np.sqrt(1 - result / np.sqrt(l1_norm_x * l1_norm_y))


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
            * ``random_state``

    Returns
    -------
    topics: array of shape (k, n_words)
        The topics generated from the bootstrap sample.
    """
    A = X.tocsr()
    if kwargs.get("bootstrap", True):
        rng = check_random_state(kwargs.get("random_state", None))
        bootstrap_sample_indices = rng.randint(0, A.shape[0], size=A.shape[0])
        B = A[bootstrap_sample_indices]
    else:
        B = A
    sample_weight = _check_sample_weight(None, B, dtype=np.float32)
    doc_topic, topic_vocab = plsa_fit(
        B,
        k,
        sample_weight,
        init=kwargs.get("init", "random"),
        n_iter=kwargs.get("n_iter", 100),
        n_iter_per_test=kwargs.get("n_iter_per_test", 10),
        tolerance=kwargs.get("tolerance", 0.001),
        e_step_thresh=kwargs.get("e_step_thresh", 1e-16),
        random_state=kwargs.get("random_state", None),
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
        Further keyword arguments that can be passed on th the ``NMF`` class.
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
    if kwargs.get("bootstrap", True):
        rng = check_random_state(kwargs.get("random_state", None))
        bootstrap_sample_indices = rng.randint(0, A.shape[0], size=A.shape[0])
        B = A[bootstrap_sample_indices]
    else:
        B = A
    nmf = NMF(
        n_components=k,
        init=kwargs.get("init", "nndsvd"),
        beta_loss=kwargs.get("beta_loss", 1),
        alpha=kwargs.get("alpha", 0.0),
        solver=kwargs.get("solver", "mu"),
        random_state=kwargs.get("random_state", None),
    ).fit(B)
    topics = nmf.components_.copy()
    normalize(topics, axis=1)
    return topics


def ensemble_of_topics(
    X, k, model="plsa", n_jobs=4, n_runs=16, parallelism="dask", **kwargs
):
    """Generate a large number of topic vectors by running an ensemble of
    bootstrap samples of a given corpus. Exploit the embarrassingly parallel nature of the problem
    using wither joblib or dask. Support for both pLSA and NMF approaches to topic generation are
    available. The sklearn implementation of NMF is used for NMF modeling.

    Parameters
    ----------
    X: sparse matrix of shape (n_docs, n_words)
        The bag-of-words matrix for the corpus to train on

    k: int
        The number of topics to generate per bootstrap sampled run.

    model: string (optional, default="plsa")
        The topic modeling method to use (either "plsa" or "nmf")

    n_jobs: int (optional, default=4)
        The number of jobs to run in parallel.

    n_runs: int (optional, default=16)
        The number of bootstrapped sampled runs to use for topic generation.

    parallelism: string (optional, default="dask")
        The parallelism model to use. Should be one of "dask" or "joblib".

    kwargs:
        Extra keyword based arguments to pass on to the pLSA or NMF models.

    Returns
    -------
    topics: array of shape (n_runs * k, n_words)
        The full set of all topics generated by all the topic modeling runs.

    """

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
    elif parallelism == "none":
        topics = []
        for i in range(n_runs):
            topics.append(create_topics(X, k, **kwargs))
    else:
        raise ValueError(
            "Unrecognized parallelism {}; should be one of {}".format(
                parallelism, ("dask", "joblib")
            )
        )

    return np.vstack(topics)


@numba.njit(fastmath=True, nogil=True)
def kl_divergence(a, b):
    """Compute the KL-divergence between two multinomial distributions."""
    result = 0.0
    for i in range(a.shape[0]):
        if a[i] > 0.0 and b[i] > 0.0:
            result += a[i] * (np.log2(a[i]) - np.log2(b[i]))
    return result


@numba.njit(fastmath=True, parallel=True)
def all_pairs_kl_divergence(distributions):
    """Compute all pairwise KL-divergences between a set of multinomial distributions."""
    n = distributions.shape[0]
    result = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            result[i, j] = kl_divergence(distributions[i], distributions[j])
    return result


@numba.njit(fastmath=True, parallel=True)
def all_pairs_hellinger_distance(distributions):
    """Compute all pairwise Hellinger distances between a set of multinomial distributions."""
    n = distributions.shape[0]
    result = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            result[i, j] = hellinger(distributions[i], distributions[j])
    return result


def generate_combined_topics_kl(all_topics, min_samples=5, min_cluster_size=5):
    """Given a large list of topics select out a small list of stable topics
    by clustering the topics with HDBSCAN using KL-divergence as a distance
    measure between topics.


    Parameters
    ----------
    all_topics: array of shape (N, n_words)
        The set of topics to be clustered.

    min_samples: int (optional, default=5)
        The min_samples parameter to use for HDBSCAN clustering.

    min_cluster_size: int (optional, default=5)
        The min_cluster_size parameter to use for HDBSCAN clustering

    Returns
    -------
    stable_topics: array of shape (M, n_words)
        A set of M topics, one for each cluster found by HDBSCAN.
    """
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


def generate_combined_topics_hellinger(all_topics, min_samples=5, min_cluster_size=5):
    """Given a large list of topics select out a small list of stable topics
    by clustering the topics with HDBSCAN using Hellinger as a distance
    measure between topics.


    Parameters
    ----------
    all_topics: array of shape (N, n_words)
        The set of topics to be clustered.

    min_samples: int (optional, default=5)
        The min_samples parameter to use for HDBSCAN clustering.

    min_cluster_size: int (optional, default=5)
        The min_cluster_size parameter to use for HDBSCAN clustering

    Returns
    -------
    stable_topics: array of shape (M, n_words)
        A set of M topics, one for each cluster found by HDBSCAN.
    """
    distance_matrix = all_pairs_hellinger_distance(all_topics)
    labels = hdbscan.HDBSCAN(
        min_samples=min_samples,
        min_cluster_size=min_cluster_size,
        metric="precomputed",
        cluster_selection_method="leaf",
    ).fit_predict(distance_matrix)
    result = np.empty((labels.max() + 1, all_topics.shape[1]), dtype=np.float32)
    for i in range(labels.max() + 1):
        result[i] = np.mean(np.sqrt(all_topics[labels == i]), axis=0) ** 2
        result[i] /= result[i].sum()

    return result


def generate_combined_topics_hellinger_umap(
    all_topics, min_samples=5, min_cluster_size=5, n_neighbors=15, reduced_dim=5
):
    """Given a large list of topics select out a small list of stable topics
    by mapping the topics to a low dimensional space with UMAP (using
    Hellinger distance) and then clustering the topics with HDBSCAN using
    Euclidean distance in the embedding space to measure distance between topics.


    Parameters
    ----------
    all_topics: array of shape (N, n_words)
        The set of topics to be clustered.

    min_samples: int (optional, default=5)
        The min_samples parameter to use for HDBSCAN clustering.

    min_cluster_size: int (optional, default=5)
        The min_cluster_size parameter to use for HDBSCAN clustering

    n_neighbors: int (optional, default=15)
        The n_neighbors value to use with UMAP.

    reduced_dim: int (optional, default=5)
        The dimension of the embedding space to use.

    Returns
    -------
    stable_topics: array of shape (M, n_words)
        A set of M topics, one for each cluster found by HDBSCAN.
    """
    embedding = umap.UMAP(
        n_neighbors=n_neighbors, n_components=reduced_dim, metric=hellinger
    ).fit_transform(all_topics)
    clusterer = hdbscan.HDBSCAN(
        min_samples=min_samples,
        min_cluster_size=min_cluster_size,
        cluster_selection_method="leaf",
        allow_single_cluster=True,
    ).fit(embedding)
    labels = clusterer.labels_
    membership_strengths = clusterer.probabilities_
    result = np.empty((labels.max() + 1, all_topics.shape[1]), dtype=np.float32)
    for i in range(labels.max() + 1):
        mask = labels == i
        result[i] = (
            np.average(
                np.sqrt(all_topics[mask]), axis=0, weights=membership_strengths[mask]
            )
            ** 2
        )
        result[i] /= result[i].sum()

    return result


_topic_combiner = {
    "kl_divergence": generate_combined_topics_kl,
    "hellinger": generate_combined_topics_hellinger,
    "hellinger_umap": generate_combined_topics_hellinger_umap,
}


def ensemble_fit(
    X,
    estimated_n_topics=10,
    model="plsa",
    init="random",
    min_samples=3,
    min_cluster_size=4,
    n_starts=16,
    n_jobs=1,
    parallelism="dask",
    topic_combination="hellinger_umap",
    bootstrap=True,
    n_iter=100,
    n_iter_per_test=10,
    tolerance=0.001,
    e_step_thresh=1e-16,
    lift_factor=1,
    beta_loss=1,
    alpha=0.0,
    solver="mu",
    random_state=None,
):
    """Generate a set of stable topics by using an ensemble of topic models and then clustering
    the results and generating representative topics for each cluster. The generate a set of
    document vectors based on the selected stable topics.

    Parameters
    ----------
    X: array or sparse matrix of shape (n_docs, n_words)
        The bag-of-words matrix for the corpus to train on.

    estimated_n_topics: int (optional, default=10)
        The estimated number of topics. Note that the final number of topics produced can differ
        from this value, and may be more or less than the provided value. Instead this value
        provides the algorithm with a suggestion of the approximate number of topics to use.

    model: string (optional, default="plsa")
        The topic modeling method to use (either "plsa" or "nmf")

    init: string or tuple (optional, default="random")
        The intialization method to use. This should be one of:
            * ``"random"``
            * ``"nndsvd"``
            * ``"nmf"``
        or a tuple of two ndarrays of shape (n_docs, n_topics) and (n_topics, n_words).

    int (optional, default=3)
        The min_samples parameter to use for HDBSCAN clustering.

    min_cluster_size: int (optional, default=4)
        The min_cluster_size parameter to use for HDBSCAN clustering

    n_starts: int (optional, default=16)
        The number of bootstrap sampled topic models to run -- the size of the ensemble.

    n_jobs: int (optional, default=8)
        The number of parallel jobs to run at a time.

    parallelism: string (optional, default="dask")
        The parallelism model to use. Should be one of "dask" or "joblib" or "none".

    topic_combination: string (optional, default="hellinger_umap")
        The method of comnining ensemble topics into a set of stable topics. Should be one of:
            * ``"hellinger_umap"``
            * ``"hellinger"``
            * ``"kl_divergence"``

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

    lift_factor: int (optional, default=1)
        Importance factor to apply to lift -- if high lift value are important to
        you then larger lift factors will be beneficial.

    beta_loss: float or string, (optional, default 'kullback-leibler')
        The beta loss to use if using NMF for topic modeling.

    alpha: float (optional, default=0.0)
        The alpha parameter defining regularization if using NMF for topic modeling.

    solver: string, (optional, default="mu")
        The choice of solver if using NMF for topic modeling. Should be either "cd" or "mu".

    random_state int, RandomState instance or None, (optional, default: None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. Used in in initialization.

    Returns
    -------
    doc_vectors, stable_topics: arrays of shape (n_docs, M) and (M, n_words)
        The vectors giving the probability of topics for each document, and the stable topics
        produced by the ensemble.
    """

    X = check_array(X, accept_sparse="csr", dtype=np.float32)

    if issparse(X):
        X_coo = X.tocoo()
    else:
        X_coo = coo_matrix(X, dtype=np.float32)

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
        bootstrap=bootstrap,
        lift_factor=1,
        beta_loss=beta_loss,
        alpha=alpha,
        solver=solver,
        random_state=random_state,
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
        sample_weight = _check_sample_weight(None, X, dtype=np.float32)
        doc_vectors = plsa_refit(
            X, stable_topics, sample_weight, e_step_thresh=e_step_thresh,
            random_state=random_state,
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
    """Ensemble Topic Modelling (EnsTop)

    Given a bag-of-words matrix representation of a corpus of documents, where each row of the
    matrix represents a document, and the jth element of the ith row is the count of the number of
    times the jth vocabulary word occurs in the ith document, build an ensemble of different
    topic models from bootstrap samples of the corpus, and then select a set of representative
    stable topics by clustering the topic produced.

    By default this will use pLSA for topic modelling. In that case the result will be matrices
    of conditional probabilities P(z|d) and P(w|z) such that the product matrix of probabilities
    P(w|d) maximises the likelihood of seeing the observed corpus data. Here P(z|d) represents
    the probability of topic z given document d, P(w|z) represents the probability of word w
    given topic z, and P(w|d) represents the probability of word w given document d.

    Parameters
    ----------
    n_components: int (optional, default=10)
        The estimated number of topics. Note that the final number of topics produced can differ
        from this value, and may be more or less than the provided value. Instead this value
        provides the algorithm with a suggestion of the approximate number of topics to use.

    model: string (optional, default="plsa")
        The topic modeling method to use (either "plsa" or "nmf")

    init: string or tuple (optional, default="random")
        The intialization method to use. This should be one of:
            * ``"random"``
            * ``"nndsvd"``
            * ``"nmf"``
        or a tuple of two ndarrays of shape (n_docs, n_topics) and (n_topics, n_words).

    int (optional, default=3)
        The min_samples parameter to use for HDBSCAN clustering.

    min_cluster_size: int (optional, default=4)
        The min_cluster_size parameter to use for HDBSCAN clustering

    n_starts: int (optional, default=16)
        The number of bootstrap sampled topic models to run -- the size of the ensemble.

    n_jobs: int (optional, default=8)
        The number of parallel jobs to run at a time.

    parallelism: string (optional, default="dask")
        The parallelism model to use. Should be one of "dask" or "joblib".

    topic_combination: string (optional, default="hellinger_umap")
        The method of comnining ensemble topics into a set of stable topics. Should be one of:
            * ``"hellinger_umap"``
            * ``"hellinger"``
            * ``"kl_divergence"``

    bootstrap: bool (optional, default=True)
        Whether to use bootstrap resampling of documents for greater randomization. In general
        this is a good idea that helps to prevent overfitting, however for small document
        collections, or for other reasons, this might not be desireable.

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

    lift_factor: int (optional, default=1)
        Importance factor to apply to lift -- if high lift value are important to
        you then larger lift factors will be beneficial.

    beta_loss: float or string, (optional, default 'kullback-leibler')
        The beta loss to use if using NMF for topic modeling.

    alpha: float (optional, default=0.0)
        The alpha parameter defining regularization if using NMF for topic modeling.

    solver: string, (optional, default="mu")
        The choice of solver if using NMF for topic modeling. Should be either "cd" or "mu".

    random_state int, RandomState instance or None, (optional, default: None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. Used in in initialization.

    Attributes
    ----------

    n_components_: int
        The actual number of stable topics generated by the ensemble.

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
        model="plsa",
        init="random",
        n_starts=16,
        min_samples=3,
        min_cluster_size=5,
        n_jobs=8,
        parallelism="dask",
        topic_combination="hellinger_umap",
        bootstrap=True,
        n_iter=80,
        n_iter_per_test=10,
        tolerance=0.001,
        e_step_thresh=1e-32,
        lift_factor=1,
        beta_loss=1,
        alpha=0.0,
        solver="mu",
        transform_random_seed=42,
        random_state=None,
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
        self.bootstrap = bootstrap
        self.n_iter = n_iter
        self.n_iter_per_test = n_iter_per_test
        self.tolerance = tolerance
        self.e_step_thresh = e_step_thresh
        self.lift_factor = lift_factor
        self.beta_loss = beta_loss
        self.alpha = alpha
        self.solver = solver
        self.transform_random_seed = transform_random_seed
        self.random_state = random_state

    def fit(self, X, y=None):
        """Learn the ensemble model for the data X and return the document vectors.

        This is more efficient than calling fit followed by transform.

        Parameters
        ----------
        X: array or sparse matrix of shape (n_docs, n_words)
            The data matrix pLSA is attempting to fit to.

        y: Ignored

        Returns
        -------
        self
        """
        self.fit_transform(X)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Learn the ensemble model for the data X and return the document vectors.

        This is more efficient than calling fit followed by transform.

        Parameters
        ----------
        X: array or sparse matrix of shape (n_docs, n_words)
            The data matrix pLSA is attempting to fit to.

        y: Ignored

        Returns
        -------
        embedding: array of shape (n_docs, n_topics)
            An embedding of the documents into a topic space.
        """
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
            self.bootstrap,
            self.n_iter,
            self.n_iter_per_test,
            self.tolerance,
            self.e_step_thresh,
            self.lift_factor,
            self.beta_loss,
            self.alpha,
            self.solver,
            self.random_state,
        )
        self.components_ = V
        self.embedding_ = U
        self.training_data_ = X
        self.n_components_ = self.components_.shape[0]

        return U

    def transform(self, X, y=None):
        """Transform the data X into the topic space of the fitted ensemble model.

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
        random_state = check_random_state(self.transform_random_seed)

        if not issparse(X):
            X = coo_matrix(X)
        else:
            X = X.tocoo()

        result = plsa_refit(
            X,
            self.components_,
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
            return mean_log_lift(self.components_, self.training_data_, n_words=n_words)
        elif topic_num >= 0 and topic_num < self.n_components:
            return log_lift(
                self.components_, topic_num, self.training_data_, n_words=n_words
            )
        else:
            raise ValueError(
                "Topic number must be in range 0 to {}".format(self.n_components)
            )
