# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Global decision-predicate explanations for tree ensembles.

This module provides a decision predicate graph (DPG) style global
explainer: every split of every tree contributes its predicate
``(feature, threshold)`` to a forest-wide aggregate, and consecutive
splits along root-to-leaf paths contribute directed co-occurrence
edges. The result is a structure-level explanation of the whole
ensemble -- which features and thresholds dominate the partitioning --
that complements the local, per-sample SHAP explainers in this package.

The construction is adapted from the Decision Predicate Graph method
extended to isolation forests in:

    "Extending Decision Predicate Graphs for Comprehensive Explanation
    of Isolation Forest", arXiv:2505.04019.

Unlike the reference method, predicates are weighted by occurrence
counts and node depth rather than by training-set traversal counts,
which the Treelite model export does not carry.
"""

from typing import NamedTuple

import numpy as np
import treelite

__all__ = [
    "Predicate",
    "PredicateEdge",
    "PredicateGraph",
    "build_predicate_graph",
]


class Predicate(NamedTuple):
    """One split predicate aggregated over the whole forest.

    Attributes
    ----------
    feature : int
        Index of the feature the split tests.
    threshold : float
        Threshold of the split; samples go left when
        ``x[feature] <= threshold``.
    count : int
        Number of tree nodes in the forest using this predicate.
    mean_depth : float
        Mean depth (root = 0) of the nodes using this predicate.
    """

    feature: int
    threshold: float
    count: int
    mean_depth: float


class PredicateEdge(NamedTuple):
    """A directed co-occurrence of two predicates along tree paths.

    An edge ``parent -> child`` with ``count = k`` means a node testing
    ``child`` is the direct child of a node testing ``parent`` in ``k``
    places across the forest.
    """

    parent: Predicate
    child: Predicate
    count: int


class PredicateGraph:
    """Global decision-predicate graph of a tree ensemble.

    Attributes
    ----------
    predicates : list of Predicate
        Unique split predicates, sorted by descending ``count``.
    edges : list of PredicateEdge
        Directed parent->child co-occurrences, sorted by descending
        ``count``.
    feature_importances_ : np.ndarray of shape (n_features,)
        Share of all split nodes that test each feature.
    n_features : int
        Number of features referenced by the ensemble.
    n_trees : int
        Number of trees in the ensemble.
    n_splits : int
        Total number of split nodes across the forest.
    """

    def __init__(self, predicates, edges, n_features, n_trees, n_splits):
        self.predicates = list(predicates)
        self.edges = list(edges)
        self.n_features = n_features
        self.n_trees = n_trees
        self.n_splits = n_splits
        importances = np.zeros(n_features, dtype=np.float64)
        for predicate in self.predicates:
            if 0 <= predicate.feature < n_features:
                importances[predicate.feature] += predicate.count
        total = importances.sum()
        if total > 0:
            importances /= total
        self.feature_importances_ = importances

    def top_predicates(self, n=10):
        """Return the ``n`` most frequently used predicates."""
        return self.predicates[:n]

    def to_dict(self):
        """Return a JSON-serializable view of the graph."""
        return {
            "n_features": self.n_features,
            "n_trees": self.n_trees,
            "n_splits": self.n_splits,
            "feature_importances": self.feature_importances_.tolist(),
            "predicates": [p._asdict() for p in self.predicates],
            "edges": [
                {
                    "parent": e.parent._asdict(),
                    "child": e.child._asdict(),
                    "count": e.count,
                }
                for e in self.edges
            ],
        }

    def __repr__(self):
        return (
            f"PredicateGraph(n_trees={self.n_trees}, "
            f"n_predicates={len(self.predicates)}, "
            f"n_edges={len(self.edges)})"
        )


def build_predicate_graph(model):
    """Build a global decision-predicate graph from a tree ensemble.

    Parameters
    ----------
    model : treelite.Model
        A fitted ensemble, e.g. from
        :meth:`cuml.ensemble.IsolationForest.as_treelite`. Any object
        exposing ``estimators_`` in the shape of the output of
        ``treelite.sklearn.export_model`` is also accepted.

    Returns
    -------
    PredicateGraph
        The aggregated predicate graph of the ensemble.
    """
    if hasattr(model, "estimators_"):
        exported = model
        n_features = None
    else:
        exported = treelite.sklearn.export_model(model)
        n_features = getattr(model, "num_feature", None)
        if n_features is not None:
            n_features = int(n_features)

    counts = {}
    depth_sums = {}
    edge_counts = {}
    n_splits = 0
    max_feature = -1

    for estimator in exported.estimators_:
        tree = estimator.tree_
        feature = np.asarray(tree.feature)
        threshold = np.asarray(tree.threshold)
        children_left = np.asarray(tree.children_left)
        children_right = np.asarray(tree.children_right)

        stack = [(0, 0)]
        while stack:
            node, depth = stack.pop()
            left = int(children_left[node])
            if left == -1:
                continue
            key = (int(feature[node]), float(threshold[node]))
            counts[key] = counts.get(key, 0) + 1
            depth_sums[key] = depth_sums.get(key, 0) + depth
            max_feature = max(max_feature, key[0])
            n_splits += 1
            for child in (left, int(children_right[node])):
                if int(children_left[child]) != -1:
                    child_key = (
                        int(feature[child]),
                        float(threshold[child]),
                    )
                    edge = (key, child_key)
                    edge_counts[edge] = edge_counts.get(edge, 0) + 1
                stack.append((child, depth + 1))

    if n_features is None:
        n_features = max_feature + 1
    else:
        n_features = max(n_features, max_feature + 1)

    predicates = [
        Predicate(
            feature=key[0],
            threshold=key[1],
            count=counts[key],
            mean_depth=depth_sums[key] / counts[key],
        )
        for key in counts
    ]
    predicates.sort(key=lambda p: (-p.count, p.feature, p.threshold))
    by_key = {(p.feature, p.threshold): p for p in predicates}
    edges = [
        PredicateEdge(parent=by_key[parent], child=by_key[child], count=count)
        for (parent, child), count in edge_counts.items()
    ]
    edges.sort(
        key=lambda e: (
            -e.count,
            e.parent.feature,
            e.parent.threshold,
            e.child.feature,
            e.child.threshold,
        )
    )
    return PredicateGraph(
        predicates,
        edges,
        n_features,
        n_trees=len(exported.estimators_),
        n_splits=n_splits,
    )
