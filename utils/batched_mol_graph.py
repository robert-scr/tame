from dataclasses import dataclass
from typing import List

import numpy as np

from utils.molecular_graph import MolGraph


@dataclass(frozen=True)
class BatchedMolGraph:
    X: np.ndarray                 # (sumN, F)
    edge_index: np.ndarray        # (2, sumE)
    edge_weight: np.ndarray       # (sumE,)
    batch: np.ndarray             # (sumN,) graph-id per node
    ptr: np.ndarray               # (B+1,) prefix sums of nodes


def batch_graphs(graphs: List[MolGraph]) -> BatchedMolGraph:
    if len(graphs) == 0:
        raise ValueError("graphs must be non-empty")

    Xs = []
    eis = []
    ews = []
    batches = []
    ptr = [0]

    node_offset = 0
    for g_id, g in enumerate(graphs):
        n = int(g.n_nodes)
        Xs.append(g.X)

        # shift edge indices by node_offset
        ei = g.edge_index.copy()
        ei = ei + node_offset
        eis.append(ei)
        ews.append(g.edge_weight)

        batches.append(np.full((n,), g_id, dtype=np.int64))
        node_offset += n
        ptr.append(node_offset)

    X = np.concatenate(Xs, axis=0)
    edge_index = np.concatenate(eis, axis=1) if eis else np.zeros((2, 0), dtype=np.int64)
    edge_weight = np.concatenate(ews, axis=0) if ews else np.zeros((0,), dtype=np.float32)
    batch = np.concatenate(batches, axis=0)

    return BatchedMolGraph(X=X, edge_index=edge_index, edge_weight=edge_weight, batch=batch, ptr=np.asarray(ptr, dtype=np.int64))