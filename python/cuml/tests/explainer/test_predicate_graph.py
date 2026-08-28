# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for the global decision-predicate graph explainer."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from cuml.explainer.predicate_graph import (
    PredicateGraph,
    build_predicate_graph,
)


def _fake_tree(feature, threshold, children_left, children_right):
    """A stand-in for one tree of ``treelite.sklearn.export_model``."""
    return SimpleNamespace(
        tree_=SimpleNamespace(
            feature=np.asarray(feature),
            threshold=np.asarray(threshold, dtype=np.float64),
            children_left=np.asarray(children_left),
            children_right=np.asarray(children_right),
        )
    )


@pytest.fixture
def exported_forest():
    """Two tiny trees shaped like ``treelite.sklearn.export_model``.

    Tree 0 splits on (feature 0, 0.5) at the root, then on
    (feature 1, 1.5) in its left subtree. Tree 1 repeats the root split
    of tree 0 with two leaves.
    """
    tree0 = _fake_tree(
        feature=[0, 1, -2, -2, -2],
        threshold=[0.5, 1.5, -2.0, -2.0, -2.0],
        children_left=[1, 2, -1, -1, -1],
        children_right=[4, 3, -1, -1, -1],
    )
    tree1 = _fake_tree(
        feature=[0, -2, -2],
        threshold=[0.5, -2.0, -2.0],
        children_left=[1, -1, -1],
        children_right=[2, -1, -1],
    )
    return SimpleNamespace(estimators_=[tree0, tree1])


def test_build_from_exported_forest(exported_forest):
    graph = build_predicate_graph(exported_forest)

    assert isinstance(graph, PredicateGraph)
    assert graph.n_trees == 2
    assert graph.n_splits == 3
    assert graph.n_features == 2

    by_key = {(p.feature, p.threshold): p for p in graph.predicates}
    assert by_key[(0, 0.5)].count == 2
    assert by_key[(0, 0.5)].mean_depth == 0.0
    assert by_key[(1, 1.5)].count == 1
    assert by_key[(1, 1.5)].mean_depth == 1.0

    # The root predicate repeats across trees, so it ranks first.
    assert graph.top_predicates(1) == [by_key[(0, 0.5)]]

    np.testing.assert_allclose(
        graph.feature_importances_, [2.0 / 3.0, 1.0 / 3.0]
    )


def test_edges_record_parent_child_cooccurrence(exported_forest):
    graph = build_predicate_graph(exported_forest)

    assert len(graph.edges) == 1
    edge = graph.edges[0]
    assert (edge.parent.feature, edge.parent.threshold) == (0, 0.5)
    assert (edge.child.feature, edge.child.threshold) == (1, 1.5)
    assert edge.count == 1


def test_to_dict_is_serializable(exported_forest):
    graph = build_predicate_graph(exported_forest)
    payload = graph.to_dict()
    assert json.loads(json.dumps(payload))["n_splits"] == 3
    assert len(payload["predicates"]) == 2
    assert len(payload["edges"]) == 1
