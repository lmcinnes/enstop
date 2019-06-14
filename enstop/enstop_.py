import numpy as np
import numba
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.extmath import randomized_svd
from sklearn.decomposition import NMF, non_negative_factorization
from scipy.sparse import issparse, csr_matrix, coo_matrix, csc_matrix
import dask
import joblib
from hdbscan._hdbscan_linkage import mst_linkage_core, label
from hdbscan.hdbscan_ import _tree_to_labels
import hdbscan
import umap
from umap.distances import hellinger


@numba.njit(fastmath=True, nogil=True)
def normalize(ndarray, axis=0):
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


@numba.njit(fastmath=True, nogil=True)
def plsa_e_step(
    X_rows,
    X_cols,
    X_vals,
    p_w_given_z,
    p_z_given_d,
    p_z_given_wd,
    probability_threshold=1e-16,
):

    k = p_w_given_z.shape[0]

    for nz_idx in range(X_vals.shape[0]):
        d = X_rows[nz_idx]
        w = X_cols[nz_idx]

        norm = 0.0
        for z in range(k):
            v = p_w_given_z[z, w] * p_z_given_d[d, z]
            if v > probability_threshold:
                p_z_given_wd[nz_idx, z] = v
                norm += p_z_given_wd[nz_idx, z]
            else:
                p_z_given_wd[nz_idx, z] = 0.0
        for z in range(k):
            if norm > 0:
                p_z_given_wd[nz_idx, z] /= norm

    return p_z_given_wd


@numba.njit(fastmath=True, nogil=True)
def plsa_m_step(
    X_rows,
    X_cols,
    X_vals,
    p_w_given_z,
    p_z_given_d,
    p_z_given_wd,
    norm_pwz,
    norm_pdz,
    n,
    m,
):

    k = p_z_given_wd.shape[1]

    p_w_given_z[:] = 0.0
    p_z_given_d[:] = 0.0

    norm_pwz[:] = 0.0
    norm_pdz[:] = 0.0

    for nz_idx in range(X_vals.shape[0]):
        d = X_rows[nz_idx]
        w = X_cols[nz_idx]
        x = X_vals[nz_idx]

        for z in range(k):
            s = x * p_z_given_wd[nz_idx, z]

            p_w_given_z[z, w] += s
            p_z_given_d[d, z] += s

            norm_pwz[z] += s
            norm_pdz[d] += s

    for z in range(k):
        for w in range(m):
            if norm_pwz[z] > 0:
                p_w_given_z[z, w] /= norm_pwz[z]
        for d in range(n):
            if norm_pdz[d] > 0:
                p_z_given_d[d, z] /= norm_pdz[d]

    return p_w_given_z, p_z_given_d


@numba.njit(fastmath=True, nogil=True)
def log_likelihood(X_rows, X_cols, X_vals, p_w_given_z, p_z_given_d):

    result = 0.0
    k = p_w_given_z.shape[0]

    for nz_idx in range(X_vals.shape[0]):
        d = X_rows[nz_idx]
        w = X_cols[nz_idx]
        x = X_vals[nz_idx]

        p_w_given_d = 0.0
        for z in range(k):
            p_w_given_d += p_w_given_z[z, w] * p_z_given_d[d, z]

        result += x * np.log(p_w_given_d)

    return result


@numba.njit(fastmath=True, nogil=True)
def norm(x):

    result = 0.0

    for i in range(x.shape[0]):
        result += x[i] ** 2

    return np.sqrt(result)


@numba.jit(fastmath=True)
def plsa_init(X, k, init="nnsvd"):

    n = X.shape[0]
    m = X.shape[1]

    if init == "random":
        p_w_given_z = np.random.random((k, m))
        p_z_given_d = np.random.random((n, k))

    elif init == "nndsvd":
        # Taken from sklearn NMF implementation
        U, S, V = randomized_svd(X, k)
        p_z_given_d, p_w_given_z = np.zeros(U.shape), np.zeros(V.shape)

        # The leading singular triplet is non-negative
        # so it can be used as is for initialization.
        p_z_given_d[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
        p_w_given_z[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])

        for j in range(1, k):
            x, y = U[:, j], V[j, :]

            # extract positive and negative parts of column vectors
            x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
            x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))

            # and their norms
            x_p_nrm, y_p_nrm = norm(x_p), norm(y_p)
            x_n_nrm, y_n_nrm = norm(x_n), norm(y_n)

            m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

            # choose update
            if m_p > m_n:
                u = x_p / x_p_nrm
                v = y_p / y_p_nrm
                sigma = m_p
            else:
                u = x_n / x_n_nrm
                v = y_n / y_n_nrm
                sigma = m_n

            lbd = np.sqrt(S[j] * sigma)
            p_z_given_d[:, j] = lbd * u
            p_w_given_z[j, :] = lbd * v

    elif init == "nmf":
        p_z_given_d, p_w_given_z, _ = non_negative_factorization(
            X,
            n_components=k,
            init="nndsvd",
            solver="cd",
            beta_loss=2,
            tol=1e-2,
            max_iter=100,
        )
    elif isinstance(init, tuple) or isinstance(init, list):
        p_z_given_d, p_w_given_z = init
    else:
        raise ValueError("Unrecognized init {}".format(init))

    normalize(p_w_given_z, axis=1)
    normalize(p_z_given_d, axis=1)

    return p_z_given_d, p_w_given_z


@numba.njit(fastmath=True, nogil=True)
def plsa_fit_inner(
    X_rows,
    X_cols,
    X_vals,
    n,
    m,
    k,
    p_w_given_z,
    p_z_given_d,
    n_iter=100,
    n_iter_per_test=10,
    tolerance=0.001,
    e_step_thresh=1e-16,
):

    p_z_given_wd = np.zeros((X_vals.shape[0], k))

    norm_pwz = np.zeros(k)
    norm_pdz = np.zeros(n)

    previous_log_likelihood = log_likelihood(
        X_rows, X_cols, X_vals, p_w_given_z, p_z_given_d
    )

    for i in range(n_iter):

        plsa_e_step(
            X_rows,
            X_cols,
            X_vals,
            p_w_given_z,
            p_z_given_d,
            p_z_given_wd,
            e_step_thresh,
        )
        plsa_m_step(
            X_rows,
            X_cols,
            X_vals,
            p_w_given_z,
            p_z_given_d,
            p_z_given_wd,
            norm_pwz,
            norm_pdz,
            n,
            m,
        )

        if i % n_iter_per_test == 0:
            current_log_likelihood = log_likelihood(
                X_rows, X_cols, X_vals, p_w_given_z, p_z_given_d
            )
            change = np.abs(current_log_likelihood - previous_log_likelihood)
            if change / np.abs(current_log_likelihood) < tolerance:
                break
            else:
                previous_log_likelihood = current_log_likelihood

    return p_z_given_d, p_w_given_z


@numba.jit()
def plsa_fit(
    X,
    k,
    init="nndsvd",
    n_iter=100,
    n_iter_per_test=10,
    tolerance=0.001,
    e_step_thresh=1e-16,
):

    n = X.shape[0]
    m = X.shape[1]

    p_z_given_d, p_w_given_z = plsa_init(X, k, init=init)

    A = X.tocoo()

    p_z_given_d, p_w_given_z = plsa_fit_inner(
        A.row,
        A.col,
        A.data,
        n,
        m,
        k,
        p_w_given_z,
        p_z_given_d,
        n_iter,
        n_iter_per_test,
        tolerance,
        e_step_thresh,
    )

    return p_z_given_d, p_w_given_z


@numba.njit(nogil=True)
def plsa_refit_m_step(
    X_rows, X_cols, X_vals, p_w_given_z, p_z_given_d, p_z_given_wd, norm_pdz, n, m
):

    k = p_z_given_wd.shape[1]

    p_z_given_d[:] = 0.0
    norm_pdz[:] = 0.0

    for nz_idx in range(X_vals.shape[0]):
        d = X_rows[nz_idx]
        w = X_cols[nz_idx]
        x = X_vals[nz_idx]

        for z in range(k):
            s = x * p_z_given_wd[nz_idx, z]
            p_z_given_d[d, z] += s
            norm_pdz[d] += s

    for z in range(k):
        for d in range(n):
            if norm_pdz[d] > 0:
                p_z_given_d[d, z] /= norm_pdz[d]

    return p_w_given_z, p_z_given_d


@numba.njit(nogil=True)
def plsa_refit(
    X_rows,
    X_cols,
    X_vals,
    n,
    m,
    topics,
    n_iter=50,
    n_iter_per_test=5,
    tolerance=0.001,
    e_step_thresh=1e-16,
):

    k = topics.shape[0]

    p_z_given_d = np.random.random((n, k))
    p_z_given_wd = np.zeros((X_vals.shape[0], k))

    norm_pdz = np.zeros(n)

    normalize(p_z_given_d, axis=1)

    previous_log_likelihood = log_likelihood(
        X_rows, X_cols, X_vals, topics, p_z_given_d
    )

    for i in range(n_iter):

        plsa_e_step(
            X_rows, X_cols, X_vals, topics, p_z_given_d, p_z_given_wd, e_step_thresh
        )
        plsa_refit_m_step(
            X_rows, X_cols, X_vals, topics, p_z_given_d, p_z_given_wd, norm_pdz, n, m
        )

        if i % n_iter_per_test == 0:
            current_log_likelihood = log_likelihood(
                X_rows, X_cols, X_vals, topics, p_z_given_d
            )
            if current_log_likelihood > 0:
                change = np.abs(current_log_likelihood - previous_log_likelihood)
                if change / np.abs(current_log_likelihood) < tolerance:
                    break
                else:
                    previous_log_likelihood = current_log_likelihood

    return p_z_given_d


def plsa_topics(X, k, **kwargs):
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
    elif parallelism == "joblib":
        joblib_topics = joblib.delayed(create_topics)
        topics = joblib.Parallel(n_jobs=n_jobs, prefer="threads")(
            joblib_topics(X, k, **kwargs) for i in range(n_runs)
        )
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


@numba.njit(fastmath=True, parallel=True)
def all_pairs_hellinger_distance(distributions):
    n = distributions.shape[0]
    result = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            result[i, j] = hellinger(distributions[i], distributions[j])
    return result


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


def generate_combined_topics_hellinger(all_topics, min_samples=5, min_cluster_size=5):
    distance_matrix = all_pairs_hellinger_distance(all_topics)
    labels = hdbscan.HDBSCAN(
        min_samples=min_samples, min_cluster_size=min_cluster_size, metric="precomputed"
    ).fit_predict(distance_matrix)
    result = np.empty((labels.max() + 1, all_topics.shape[1]), dtype=np.float32)
    for i in range(labels.max() + 1):
        result[i] = np.mean(np.sqrt(all_topics[labels == i]), axis=0) ** 2
        result[i] /= result[i].sum()

    return result


def generate_combined_topics_hellinger_umap(
    all_topics, min_samples=5, min_cluster_size=5, n_neighbors=15, reduced_dim=5
):
    embedding = umap.UMAP(
        n_neighbors=n_neighbors, n_components=reduced_dim, metric="hellinger"
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
    "hellinger": generate_combined_topics_hellinger,
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


class PLSA(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_components=10,
        init="nndsvd",
        n_iter=100,
        n_iter_per_test=10,
        tolerance=0.001,
        e_step_thresh=1e-16,
    ):

        self.n_components = n_components
        self.init = init
        self.n_iter = n_iter
        self.n_iter_per_test = n_iter_per_test
        self.tolerance = tolerance
        self.e_step_thresh = e_step_thresh

    def fit(self, X, y=None):
        self.fit_transform(X)
        return self

    def fit_transform(self, X, y=None):

        X = check_array(X, accept_sparse="csr")

        if not issparse(X):
            X = csr_matrix(X)

        U, V = plsa_fit(
            X,
            self.n_components,
            self.init,
            self.n_iter,
            self.n_iter_per_test,
            self.tolerance,
            self.e_step_thresh,
        )
        self.components_ = V
        self.embedding_ = U
        self.training_data_ = X

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
            return mean_coherence(self.components_, self.training_data_, n_words)
        elif topic_num >= 0 and topic_num < self.n_components:
            return coherence(self.components_, topic_num, self.training_data_, n_words)
        else:
            raise ValueError(
                "Topic number must be in range 0 to {}".format(self.n_components)
            )

    def log_lift(self, topic_num=None, n_words=20):

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
