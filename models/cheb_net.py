
from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
	from models.cheb_layer import ChebLayer, scaled_laplacian_from_edges
except ModuleNotFoundError:
	# Allow running this file directly: `python models/cheb_net.py`
	import sys
	from pathlib import Path

	repo_root = Path(__file__).resolve().parents[1]
	if str(repo_root) not in sys.path:
		sys.path.insert(0, str(repo_root))
	from models.cheb_layer import ChebLayer, scaled_laplacian_from_edges


PoolType = Literal["sum", "mean", "set2set"]


def _scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
	"""Scatter sum using native PyTorch (autograd safe).

	Args:
		src: Source tensor (N,) or (N, C).
		index: Index tensor (N,).
		dim_size: Size of output dimension 0.

	Returns:
		Summed tensor (dim_size,) or (dim_size, C).
	"""
	if src.dim() == 1:
		src = src.unsqueeze(-1)
		squeeze = True
	else:
		squeeze = False

	out = torch.zeros(dim_size, src.size(-1), device=src.device, dtype=src.dtype)
	expanded_index = index.unsqueeze(-1).expand_as(src)
	out = out.scatter_add(0, expanded_index, src)

	if squeeze:
		out = out.squeeze(-1)
	return out


def _scatter_softmax(src: torch.Tensor, index: torch.Tensor, num_graphs: int) -> torch.Tensor:
	"""Compute softmax over groups defined by index (numerically stable, autograd safe).

	Args:
		src: Source tensor (N,).
		index: Index tensor (N,) indicating group membership.
		num_graphs: Number of groups.

	Returns:
		Softmax values (N,).
	"""
	# Compute max per group for numerical stability
	src_max = torch.full((num_graphs,), float("-inf"), device=src.device, dtype=src.dtype)
	src_max = src_max.scatter_reduce(0, index, src, reduce="amax", include_self=True)
	src_max = torch.where(src_max == float("-inf"), torch.zeros_like(src_max), src_max)

	# Subtract max and exponentiate
	src_shifted = src - src_max[index]
	exp_src = torch.exp(src_shifted)

	# Sum per group
	exp_sum = _scatter_sum(exp_src, index, num_graphs)

	# Normalize
	return exp_src / (exp_sum[index] + 1e-8)


class Set2SetPooling(nn.Module):
	"""Set2Set pooling layer (Vinyals et al., 2015) - Pure PyTorch implementation.

	Produces a graph-level embedding by iteratively attending over node features
	using an LSTM. Output dimension is 2 * input_dim.

	Reference: "Order Matters: Sequence to sequence for sets" (Vinyals et al., 2015)

	Args:
		input_dim: Dimension of input node features.
		n_iters: Number of LSTM iterations (processing steps, typically 3-6).
		n_layers: Number of LSTM layers.
	"""

	def __init__(self, input_dim: int, n_iters: int = 3, n_layers: int = 1) -> None:
		super().__init__()
		self.input_dim = input_dim
		self.output_dim = 2 * input_dim
		self.n_iters = n_iters
		self.n_layers = n_layers

		self.lstm = nn.LSTM(
			input_size=self.output_dim,
			hidden_size=input_dim,
			num_layers=n_layers,
			batch_first=True,
		)

	def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
		"""Forward pass.

		Args:
			x: Node features (N, input_dim).
			batch: Graph assignment for each node (N,).

		Returns:
			Graph-level embeddings (B, 2 * input_dim).
		"""
		if batch.dtype != torch.long:
			batch = batch.long()

		B = int(batch.max().item()) + 1
		device = x.device
		dtype = x.dtype

		# Initialize LSTM hidden state
		h = torch.zeros(self.n_layers, B, self.input_dim, device=device, dtype=dtype)
		c = torch.zeros(self.n_layers, B, self.input_dim, device=device, dtype=dtype)

		# Initialize query vector q_star
		q_star = torch.zeros(B, self.output_dim, device=device, dtype=dtype)

		for _ in range(self.n_iters):
			# LSTM step: input is q_star, output is new query q
			q, (h, c) = self.lstm(q_star.unsqueeze(1), (h, c))
			q = q.squeeze(1)  # (B, input_dim)

			# Compute attention scores: e_i = x_i · q_{batch[i]}
			q_expanded = q[batch]  # (N, input_dim)
			e = (x * q_expanded).sum(dim=-1)  # (N,)

			# Softmax over nodes within each graph
			a = _scatter_softmax(e, batch, B)  # (N,)

			# Weighted sum of node features per graph (read vector)
			r = _scatter_sum(a.unsqueeze(-1) * x, batch, B)  # (B, input_dim)

			# Concatenate q and r to form new q_star
			q_star = torch.cat([q, r], dim=-1)  # (B, 2 * input_dim)

		return q_star


def global_pool(x: torch.Tensor, batch: torch.Tensor, *, pool: PoolType = "mean") -> torch.Tensor:
	"""Global pooling over nodes to graph embeddings.

	x:     (N, C)
	batch: (N,) graph id per node in [0, B-1]
	returns: (B, C)
	"""
	if batch.numel() == 0:
		raise ValueError("batch must be non-empty")

	if batch.dtype != torch.long:
		batch = batch.long()

	B = int(batch.max().item()) + 1
	out = torch.zeros((B, x.size(-1)), device=x.device, dtype=x.dtype)
	out.index_add_(0, batch, x)

	if pool == "sum":
		return out

	if pool == "mean":
		counts = torch.zeros((B,), device=x.device, dtype=x.dtype)
		ones = torch.ones((batch.numel(),), device=x.device, dtype=x.dtype)
		counts.index_add_(0, batch, ones)
		counts = counts.clamp_min(1.0).unsqueeze(-1)
		return out / counts

	# Note: set2set pooling is handled separately via Set2Set module
	raise ValueError(f"Unknown pool={pool!r}. For 'set2set', use the Set2Set module directly.")


class ChebNet(nn.Module):
	"""Minimal Chebyshev-GCN for graph-level prediction.

	Pipeline:
	  X -> [ChebLayer + ReLU + Dropout] * num_layers -> global pool -> MLP head

	Intended usage with your batching:
	  - build BatchedMolGraph via utils.batched_mol_graph.batch_graphs
	  - convert X/edge_index/edge_weight/batch to torch
	  - build L_tilde with scaled_laplacian_from_edges(...) once per batch
	"""

	def __init__(
		self,
		in_channels: int,
		hidden_channels: int,
		out_channels: int,
		*,
		K: int = 3,
		num_layers: int = 2,
		dropout: float = 0.1,
		pool: PoolType = "mean",
		lambda_max: float = 2.0,
		set2set_processing_steps: int = 3,
	) -> None:
		super().__init__()
		if num_layers < 1:
			raise ValueError("num_layers must be >= 1")

		self.pool: PoolType = pool
		self.dropout = float(dropout)
		self.lambda_max = float(lambda_max)

		layers = []
		for layer_idx in range(num_layers):
			fin = in_channels if layer_idx == 0 else hidden_channels
			fout = hidden_channels
			layers.append(ChebLayer(fin, fout, K=K))
		self.layers = nn.ModuleList(layers)

		# Set2Set pooling outputs 2x the input dimension
		self.set2set_pooling: Optional[Set2SetPooling] = None
		if pool == "set2set":
			self.set2set_pooling = Set2SetPooling(hidden_channels, n_iters=set2set_processing_steps)
			head_input_dim = 2 * hidden_channels
		else:
			head_input_dim = hidden_channels

		self.head = nn.Sequential(
			nn.Linear(head_input_dim, hidden_channels),
			nn.ReLU(),
			nn.Dropout(self.dropout),
			nn.Linear(hidden_channels, out_channels),
		)

	def forward(
		self,
		x: torch.Tensor,
		edge_index: torch.Tensor,
		edge_weight: Optional[torch.Tensor],
		batch: torch.Tensor,
		*,
		L_tilde: Optional[torch.sparse.Tensor] = None,
	) -> torch.Tensor:
		"""Forward pass.

		x:          (N, Fin)
		edge_index: (2, E)
		edge_weight:(E,) or None
		batch:      (N,)
		L_tilde:    optional precomputed scaled Laplacian for this batch
		returns:    (B, out_channels) graph-level predictions
		"""
		if L_tilde is None:
			L_tilde = scaled_laplacian_from_edges(
				edge_index=edge_index,
				edge_weight=edge_weight,
				n_nodes=int(x.size(0)),
				lambda_max=self.lambda_max,
				device=x.device,
				dtype=x.dtype,
			)

		h = x
		for layer in self.layers:
			h = layer(h, L_tilde)
			h = F.relu(h)
			h = F.dropout(h, p=self.dropout, training=self.training)

		if self.pool == "set2set" and self.set2set_pooling is not None:
			g = self.set2set_pooling(h, batch)
		else:
			g = global_pool(h, batch, pool=self.pool)  # type: ignore[arg-type]
		return self.head(g)


def _to_torch_batched(bg) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	"""Adapter for utils.batched_mol_graph.BatchedMolGraph (numpy) -> torch tensors."""
	x = torch.from_numpy(bg.X.astype(np.float32, copy=False))
	edge_index = torch.from_numpy(bg.edge_index.astype(np.int64, copy=False))
	edge_weight = torch.from_numpy(bg.edge_weight.astype(np.float32, copy=False))
	batch = torch.from_numpy(bg.batch.astype(np.int64, copy=False))
	return x, edge_index, edge_weight, batch


if __name__ == "__main__":
	# Minimal runnable example: two SMILES, batched, graph-level output.
	import sys
	from pathlib import Path

	repo_root = Path(__file__).resolve().parents[1]
	if str(repo_root) not in sys.path:
		sys.path.insert(0, str(repo_root))

	from utils.molecular_graph import smiles_to_graph
	from utils.batched_mol_graph import batch_graphs

	g1 = smiles_to_graph("CCO")
	g2 = smiles_to_graph("c1ccccc1")
	bg = batch_graphs([g1, g2])

	x, edge_index, edge_weight, batch = _to_torch_batched(bg)

	model = ChebNet(
		in_channels=x.size(1),
		hidden_channels=32,
		out_channels=1,
		K=3,
		num_layers=2,
		pool="mean",
	)

	with torch.no_grad():
		y = model(x, edge_index, edge_weight, batch)

	print("x:", tuple(x.shape))
	print("batch graphs:", int(batch.max().item()) + 1)
	print("y:", tuple(y.shape))
