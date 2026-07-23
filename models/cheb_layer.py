
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

SparseOrDense = Union[torch.Tensor, torch.sparse.Tensor]


def scaled_laplacian_from_edges(
	edge_index: torch.Tensor,
	edge_weight: Optional[torch.Tensor],
	n_nodes: int,
	*,
	lambda_max: float = 2.0,
	device: Optional[torch.device] = None,
	dtype: torch.dtype = torch.float32,
) -> torch.sparse.Tensor:
	"""Build the ChebNet scaled Laplacian from an edge list.

	We use the normalized Laplacian:

		L = I - D^{-1/2} A D^{-1/2}

	and the ChebNet scaling:

		L_tilde = (2 / lambda_max) * L - I

	For the normalized Laplacian, the spectrum is in [0, 2]. In practice, ChebNet
	often uses the shortcut lambda_max = 2.0 (no eigendecomposition needed).

	This implementation constructs L_tilde as a sparse COO tensor without
	materializing dense NxN matrices.

	Notes:
	- edge_index is expected to already contain both directions for an undirected graph.
	- for batched graphs (offset node indices), this naturally yields a block-diagonal operator.
	"""
	if device is None:
		device = edge_index.device

	edge_index = edge_index.to(device=device)
	if edge_weight is None:
		edge_weight = torch.ones(edge_index.size(1), device=device, dtype=dtype)
	else:
		edge_weight = edge_weight.to(device=device, dtype=dtype)

	i = edge_index[0].long()
	j = edge_index[1].long()

	# Degree: deg[u] = sum_v A_{u,v}. For undirected graphs with both directions stored,
	# this is the standard node degree (weighted).
	deg = torch.zeros(n_nodes, device=device, dtype=dtype)
	deg.scatter_add_(0, i, edge_weight)

	with torch.no_grad():
		inv_sqrt_deg = deg.rsqrt()  # 1/sqrt(deg); deg=0 -> inf
		inv_sqrt_deg[~torch.isfinite(inv_sqrt_deg)] = 0.0

	# Normalized adjacency weights: A_norm(u,v) = invsqrt(deg_u) * A(u,v) * invsqrt(deg_v)
	w_norm = inv_sqrt_deg[i] * edge_weight * inv_sqrt_deg[j]

	scale = 2.0 / float(lambda_max)

	# Off-diagonals of L_tilde: -(2/lmax) * A_norm
	off_vals = (-scale) * w_norm

	# Diagonal of L_tilde: (2/lmax)*I - I = (2/lmax - 1)*I
	diag_idx = torch.arange(n_nodes, device=device, dtype=torch.long)
	diag_indices = torch.stack([diag_idx, diag_idx], dim=0)
	diag_vals = torch.full((n_nodes,), (scale - 1.0), device=device, dtype=dtype)

	indices = torch.cat([edge_index.long(), diag_indices], dim=1)
	values = torch.cat([off_vals, diag_vals], dim=0)
	return torch.sparse_coo_tensor(indices, values, (n_nodes, n_nodes), device=device, dtype=dtype).coalesce()


class ChebLayer(nn.Module):
	"""Chebyshev spectral graph convolution (ChebNet) layer.

	Given scaled Laplacian L_tilde and node features X:
		T_0(X) = X
		T_1(X) = L_tilde X
		T_k(X) = 2 L_tilde T_{k-1}(X) - T_{k-2}(X)

	Output:
		Y = sum_{k=0..K-1} T_k(X) W_k + b

	Shapes:
		X:        (N, Fin)
		L_tilde:  (N, N) sparse/dense
		Y:        (N, Fout)
	"""

	def __init__(self, in_channels: int, out_channels: int, K: int, bias: bool = True):
		super().__init__()
		if K < 1:
			raise ValueError("K must be >= 1")

		self.in_channels = int(in_channels)
		self.out_channels = int(out_channels)
		self.K = int(K)

		self.weight = nn.Parameter(torch.empty(self.K, self.in_channels, self.out_channels))
		self.bias = nn.Parameter(torch.zeros(self.out_channels)) if bias else None
		self.reset_parameters()

	def reset_parameters(self) -> None:
		nn.init.xavier_uniform_(self.weight)
		if self.bias is not None:
			nn.init.zeros_(self.bias)

	@staticmethod
	def _spmm(M: SparseOrDense, X: torch.Tensor) -> torch.Tensor:
		if M.is_sparse:
			return torch.sparse.mm(M, X)
		return M @ X

	def forward(self, X: torch.Tensor, L_tilde: SparseOrDense) -> torch.Tensor:
		if X.dim() != 2:
			raise ValueError(f"X must be 2D (N, Fin), got {tuple(X.shape)}")
		if X.size(-1) != self.in_channels:
			raise ValueError(
				f"X has Fin={X.size(-1)} but layer expects in_channels={self.in_channels}"
			)

		# T_0
		Tkm2 = X
		out = Tkm2 @ self.weight[0]

		if self.K == 1:
			return out + self.bias if self.bias is not None else out

		# T_1
		Tkm1 = self._spmm(L_tilde, X)
		out = out + (Tkm1 @ self.weight[1])

		for k in range(2, self.K):
			Tk = 2.0 * self._spmm(L_tilde, Tkm1) - Tkm2
			out = out + (Tk @ self.weight[k])
			Tkm2, Tkm1 = Tkm1, Tk

		if self.bias is not None:
			out = out + self.bias
		return out


def _to_torch_edges(g) -> tuple[torch.Tensor, torch.Tensor]:
	"""Small adapter: MolGraph/BatchedMolGraph (numpy) -> torch tensors."""
	edge_index = torch.from_numpy(g.edge_index.astype(np.int64, copy=False))
	edge_weight = torch.from_numpy(g.edge_weight.astype(np.float32, copy=False))
	return edge_index, edge_weight


if __name__ == "__main__":
	# Minimal sanity example:
	# 1) SMILES -> MolGraph (numpy)
	# 2) Batch 2 graphs by offsetting indices (block-diagonal operator)
	# 3) Build L_tilde as torch sparse
	# 4) Run a ChebLayer forward pass
	import sys
	from pathlib import Path

	repo_root = Path(__file__).resolve().parents[1]
	if str(repo_root) not in sys.path:
		sys.path.insert(0, str(repo_root))

	from utils.molecular_graph import smiles_to_graph
	from utils.batched_mol_graph import batch_graphs

	g1 = smiles_to_graph("CCO")
	g2 = smiles_to_graph("c1ccccc1")
	batch = batch_graphs([g1, g2])

	X = torch.from_numpy(batch.X).float()
	edge_index, edge_weight = _to_torch_edges(batch)
	L_tilde = scaled_laplacian_from_edges(edge_index, edge_weight, n_nodes=X.size(0))

	layer = ChebLayer(in_channels=X.size(1), out_channels=16, K=3)
	Y = layer(X, L_tilde)

	print("X:", tuple(X.shape))
	print("edge_index:", tuple(edge_index.shape), "edge_weight:", tuple(edge_weight.shape))
	print("L_tilde:", tuple(L_tilde.shape), "nnz:", int(L_tilde._nnz()))
	print("Y:", tuple(Y.shape))
