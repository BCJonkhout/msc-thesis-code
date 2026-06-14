import logging
import random
import sys
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import tiktoken
import umap
from sklearn.mixture import GaussianMixture

# RAPTOR_Clustering.perform_clustering recurses to break up any cluster whose
# concatenated text exceeds max_length_in_cluster. On a large, low-diversity
# leaf set (e.g. a long NovelQA novel) the GMM can keep returning a single
# cluster equal to its input, so the original code recursed on the same set
# forever and overflowed the stack ("RecursionError: maximum recursion depth
# exceeded", observed on novels B12/B24/B38/B42). The convergence guard in
# perform_clustering (recurse only on a strictly-smaller subset) makes the
# recursion provably terminating; this depth ceiling is a cheap backstop, set
# well above the ~log2(tokens/max_len) levels a legitimate split needs.
_MAX_RECLUSTER_DEPTH = 24
# Belt-and-suspenders against any other deep recursion in the build/pickle path.
sys.setrecursionlimit(max(sys.getrecursionlimit(), 10_000))

# Initialize logging
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

from .tree_structures import Node
# Import necessary methods from other modules
from .utils import get_embeddings

# Seed the deterministic parts that are safe to seed (Python random, numpy,
# and the GMM random_state below). We deliberately do NOT pass random_state
# to umap.UMAP: doing so forces UMAP onto a numba-compiled deterministic
# code path that segfaults (Windows access violation 0xC0000005) during
# "Constructing Layer 0" on larger leaf sets, killing the build process.
# Cross-answerer determinism does not depend on UMAP being seeded: each
# tree is built once and stored in the on-disk preprocess cache, and every
# answerer reuses that byte-identical cached artefact -- so the disk cache,
# not UMAP's RNG, is the determinism mechanism.
RANDOM_SEED = 224
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def global_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: Optional[int] = None,
    metric: str = "cosine",
) -> np.ndarray:
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    reduced_embeddings = umap.UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)
    return reduced_embeddings


def local_cluster_embeddings(
    embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine"
) -> np.ndarray:
    reduced_embeddings = umap.UMAP(
        n_neighbors=num_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)
    return reduced_embeddings


def get_optimal_clusters(
    embeddings: np.ndarray, max_clusters: int = 50, random_state: int = RANDOM_SEED
) -> int:
    max_clusters = min(max_clusters, len(embeddings))
    n_clusters = np.arange(1, max_clusters)
    bics = []
    for n in n_clusters:
        # reg_covar above the sklearn 1e-6 default prevents "ill-defined
        # empirical covariance" failures when a candidate cluster collapses on
        # degenerate embeddings (seen on some long-document, e.g. NovelQA, leaf
        # sets). Negligible for well-conditioned clusters, so well-formed cached
        # trees are unaffected.
        gm = GaussianMixture(n_components=n, random_state=random_state, reg_covar=1e-4)
        gm.fit(embeddings)
        bics.append(gm.bic(embeddings))
    optimal_clusters = n_clusters[np.argmin(bics)]
    return optimal_clusters


def GMM_cluster(embeddings: np.ndarray, threshold: float, random_state: int = 0):
    n_clusters = get_optimal_clusters(embeddings)
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state, reg_covar=1e-4)
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters


def perform_clustering(
    embeddings: np.ndarray, dim: int, threshold: float, verbose: bool = False
) -> List[np.ndarray]:
    reduced_embeddings_global = global_cluster_embeddings(embeddings, min(dim, len(embeddings) -2))
    global_clusters, n_global_clusters = GMM_cluster(
        reduced_embeddings_global, threshold
    )

    if verbose:
        logging.info(f"Global Clusters: {n_global_clusters}")

    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0

    for i in range(n_global_clusters):
        global_cluster_embeddings_ = embeddings[
            np.array([i in gc for gc in global_clusters])
        ]
        if verbose:
            logging.info(
                f"Nodes in Global Cluster {i}: {len(global_cluster_embeddings_)}"
            )
        if len(global_cluster_embeddings_) == 0:
            continue
        if len(global_cluster_embeddings_) <= dim + 1:
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
            n_local_clusters = 1
        else:
            reduced_embeddings_local = local_cluster_embeddings(
                global_cluster_embeddings_, dim
            )
            local_clusters, n_local_clusters = GMM_cluster(
                reduced_embeddings_local, threshold
            )

        if verbose:
            logging.info(f"Local Clusters in Global Cluster {i}: {n_local_clusters}")

        for j in range(n_local_clusters):
            local_cluster_embeddings_ = global_cluster_embeddings_[
                np.array([j in lc for lc in local_clusters])
            ]
            indices = np.where(
                (embeddings == local_cluster_embeddings_[:, None]).all(-1)
            )[1]
            for idx in indices:
                all_local_clusters[idx] = np.append(
                    all_local_clusters[idx], j + total_clusters
                )

        total_clusters += n_local_clusters

    if verbose:
        logging.info(f"Total Clusters: {total_clusters}")
    return all_local_clusters


class ClusteringAlgorithm(ABC):
    @abstractmethod
    def perform_clustering(self, embeddings: np.ndarray, **kwargs) -> List[List[int]]:
        pass


class RAPTOR_Clustering(ClusteringAlgorithm):
    def perform_clustering(
        nodes: List[Node],
        embedding_model_name: str,
        max_length_in_cluster: int = 3500,
        tokenizer=tiktoken.get_encoding("cl100k_base"),
        reduction_dimension: int = 10,
        threshold: float = 0.1,
        verbose: bool = False,
        _depth: int = 0,
    ) -> List[List[Node]]:
        # Get the embeddings from the nodes
        embeddings = np.array([node.embeddings[embedding_model_name] for node in nodes])

        # Perform the clustering
        clusters = perform_clustering(
            embeddings, dim=reduction_dimension, threshold=threshold
        )

        # Initialize an empty list to store the clusters of nodes
        node_clusters = []
        n_input = len(nodes)

        # Iterate over each unique label in the clusters
        for label in np.unique(np.concatenate(clusters)):
            # Get the indices of the nodes that belong to this cluster
            indices = [i for i, cluster in enumerate(clusters) if label in cluster]

            # Add the corresponding nodes to the node_clusters list
            cluster_nodes = [nodes[i] for i in indices]

            # Base case: if the cluster only has one node, do not attempt to recluster it
            if len(cluster_nodes) == 1:
                node_clusters.append(cluster_nodes)
                continue

            # Calculate the total length of the text in the nodes
            total_length = sum(
                [len(tokenizer.encode(node.text)) for node in cluster_nodes]
            )

            # If the total length exceeds the maximum allowed length, recluster
            # this cluster -- but ONLY if doing so can make progress. Recurse
            # solely on a strictly-smaller subset (this level actually split
            # the input) and below the depth ceiling; otherwise the recursion
            # on a degenerate, non-splitting set never terminates and overflows
            # the stack. When we cannot split further, accept the oversized
            # cluster as-is: the summariser just receives a longer context,
            # which is bounded by the document and far cheaper than a crash.
            can_make_progress = (
                len(cluster_nodes) < n_input and _depth < _MAX_RECLUSTER_DEPTH
            )
            if total_length > max_length_in_cluster and can_make_progress:
                if verbose:
                    logging.info(
                        f"reclustering cluster with {len(cluster_nodes)} nodes "
                        f"(depth {_depth})"
                    )
                node_clusters.extend(
                    RAPTOR_Clustering.perform_clustering(
                        cluster_nodes,
                        embedding_model_name,
                        max_length_in_cluster,
                        _depth=_depth + 1,
                    )
                )
            else:
                node_clusters.append(cluster_nodes)

        return node_clusters
