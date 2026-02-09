#!/usr/bin/env python3
"""
mhc-single/run.py

Minimal, self-contained runner to compare residual / HC / mHC / mHC-lite on
FineWeb10B GPT-2 token shards (uint16 .bin files).

Single GPU:
  python mhc-single/run.py --data_root /path/to/DATA_ROOT

DDP:
  torchrun --standalone --nproc_per_node=8 mhc-single/run.py --data_root /path/to/DATA_ROOT
  # if torchrun is not available:
  python -m torch.distributed.run --standalone --nproc_per_node=8 mhc-single/run.py --data_root /path/to/DATA_ROOT
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import glob
import inspect
import json
import math
import os
import random
import time
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# einops is used for the hyper-connection implementations (ported from mhc-lite)
from einops import rearrange, repeat, reduce, einsum
from einops.layers.torch import Rearrange


# match mhc-lite: suppress noisy FutureWarning (e.g., GradScaler deprecation)
warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------------------------------------------------------
# Utils

def now_ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def parse_csv(s: str) -> list[str]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    # de-dupe while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


class JsonlWriter:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._f = open(path, "a", encoding="utf-8")

    def write(self, obj: dict[str, Any]) -> None:
        self._f.write(json.dumps(obj, ensure_ascii=True) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()


# -----------------------------------------------------------------------------
# DDP utilities

def ddp_detect() -> bool:
    return int(os.environ.get("RANK", "-1")) != -1 and int(os.environ.get("WORLD_SIZE", "-1")) != -1


@dataclass
class DDPInfo:
    ddp: bool
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    master: bool = True
    device: torch.device = torch.device("cpu")
    backend: str = "nccl"


def ddp_init(backend: str | None = None) -> DDPInfo:
    ddp = ddp_detect()
    if not ddp:
        dev = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
        return DDPInfo(ddp=False, device=dev, backend=backend or ("nccl" if dev.type == "cuda" else "gloo"))

    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ["WORLD_SIZE"])
    dev = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    if backend is None:
        backend = "nccl" if dev.type == "cuda" else "gloo"

    if dev.type == "cuda":
        torch.cuda.set_device(dev)

    dist.init_process_group(backend=backend)
    dist.barrier()

    return DDPInfo(
        ddp=True,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        master=(rank == 0),
        device=dev,
        backend=backend,
    )


def ddp_barrier(ddp: DDPInfo) -> None:
    if not ddp.ddp:
        return
    import torch.distributed as dist

    dist.barrier()


def ddp_destroy(ddp: DDPInfo) -> None:
    if not ddp.ddp:
        return
    import torch.distributed as dist

    dist.barrier()
    dist.destroy_process_group()


# -----------------------------------------------------------------------------
# FineWeb shard loader (uint16 token shards)

def resolve_shards(data_root: str, pattern: str) -> list[str]:
    full = os.path.join(os.path.abspath(os.path.expanduser(data_root)), pattern)
    paths = sorted(glob.glob(full, recursive=True))
    return paths


def shard_token_counts(paths: list[str]) -> list[int]:
    # FineWeb shards are uint16 token IDs.
    return [os.path.getsize(p) // 2 for p in paths]


def resolve_shards_auto(data_root: str, train_glob: str, val_glob: str) -> tuple[list[str], list[str], str, str]:
    # Try a few common layouts to make ModelScope dataset repos easy to use.
    tried: list[tuple[str, str, int, int]] = []

    def _try(tg: str, vg: str):
        tp = resolve_shards(data_root, tg)
        vp = resolve_shards(data_root, vg)
        tried.append((tg, vg, len(tp), len(vp)))
        if tp and vp:
            return tp, vp, tg, vg
        return None

    seen: set[tuple[str, str]] = set()
    patterns: list[tuple[str, str]] = [
        (train_glob, val_glob),
        ("fineweb_train_*.bin", "fineweb_val_*.bin"),
        ("data/fineweb10B/fineweb_train_*.bin", "data/fineweb10B/fineweb_val_*.bin"),
        ("**/fineweb_train_*.bin", "**/fineweb_val_*.bin"),
    ]
    for tg, vg in patterns:
        key = (tg, vg)
        if key in seen:
            continue
        seen.add(key)
        got = _try(tg, vg)
        if got is not None:
            return got

    msg = ["no FineWeb shard files found under data_root. Tried patterns:"]
    for tg, vg, nt, nv in tried:
        msg.append(f"  - train_glob={tg!r} ({nt} files), val_glob={vg!r} ({nv} files)")
    raise FileNotFoundError("\n".join(msg))


# -----------------------------------------------------------------------------
# Hyper-Connections (ported, minimal)

def exists(v):
    return v is not None


def divisible_by(num, den):
    return (num % den) == 0


def default(v, d):
    return v if exists(v) else d


def add(x, y):
    return x + y


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1) * self.scale * (self.gamma + 1)


class Residual(nn.Module):
    def __init__(self, *args, branch: nn.Module | None = None, residual_transform: nn.Module | None = None, **kwargs):
        super().__init__()
        self.branch = branch
        self.residual_transform = default(residual_transform, nn.Identity())

    def width_connection(self, residuals):
        return residuals, residuals, dict()

    def depth_connection(self, branch_output, residuals):
        return branch_output + self.residual_transform(residuals)

    def forward(self, residuals, *branch_args, **branch_kwargs):
        branch_input, residuals, residual_kwargs = self.width_connection(residuals)

        def add_residual_fn(branch_out):
            # In this runner, branches always return a Tensor.
            return self.depth_connection(branch_out, residuals, **residual_kwargs)

        if not exists(self.branch):
            return branch_input, add_residual_fn

        branch_output = self.branch(branch_input, *branch_args, **branch_kwargs)
        return add_residual_fn(branch_output)


class HyperConnections(nn.Module):
    def __init__(
        self,
        num_residual_streams: int,
        *,
        dim: int,
        branch: nn.Module | None = None,
        layer_index: int | None = None,
        tanh: bool = True,
        channel_first: bool = False,
        dropout: float = 0.0,
        residual_transform: nn.Module | None = None,
        add_branch_out_to_residual: bool = True,
        num_input_views: int = 1,
        depth_residual_fn=add,
        num_fracs: int = 1,
    ):
        super().__init__()
        self.branch = branch
        self.act = nn.Tanh() if tanh else nn.Identity()

        assert num_fracs >= 1
        self.num_fracs = num_fracs
        self.has_fracs = num_fracs > 1

        self.split_fracs = Rearrange("b ... (f d) -> b ... f d", f=num_fracs)
        self.merge_fracs = Rearrange("b ... f d -> b ... (f d)")
        assert divisible_by(dim, num_fracs)
        dim_eff = dim // num_fracs

        self.norm = RMSNorm(dim_eff)
        assert num_residual_streams > 0
        self.num_residual_streams = num_residual_streams
        init_residual_index = default(layer_index, random.randrange(num_residual_streams)) % num_residual_streams

        num_residual_streams_fracs = num_residual_streams * num_fracs
        num_input_views_fracs = num_input_views * num_fracs
        assert num_input_views >= 1
        self.num_input_views = num_input_views

        init_alpha0 = torch.zeros((num_residual_streams_fracs, num_input_views_fracs))
        init_alpha0[init_residual_index, :] = 1.0
        self.static_alpha = nn.Parameter(torch.cat((init_alpha0, torch.eye(num_residual_streams_fracs)), dim=1))
        self.dynamic_alpha_fn = nn.Parameter(torch.zeros(dim_eff, num_residual_streams_fracs + num_input_views_fracs))
        self.dynamic_alpha_scale = nn.Parameter(torch.ones(()) * 1e-2)

        self.add_branch_out_to_residual = add_branch_out_to_residual
        if add_branch_out_to_residual:
            self.static_beta = nn.Parameter(torch.ones(num_residual_streams_fracs))
            dynamic_beta_shape = (dim_eff,) if num_fracs == 1 else (dim_eff, num_fracs)
            self.dynamic_beta_fn = nn.Parameter(torch.zeros(dynamic_beta_shape))
            self.dynamic_beta_scale = nn.Parameter(torch.ones(()) * 1e-2)

        self.dropout = nn.Dropout(dropout)
        self.channel_first = channel_first
        self.residual_transform = default(residual_transform, nn.Identity())
        self.depth_residual_fn = depth_residual_fn

    def width_connection(self, residuals):
        streams = self.num_residual_streams

        if self.channel_first:
            residuals = rearrange(residuals, "b d ... -> b ... d")

        residuals = self.split_fracs(residuals)
        residuals = rearrange(residuals, "(b s) ... d -> b ... s d", s=streams)

        normed = self.norm(residuals)

        wc_weight = self.act(normed @ self.dynamic_alpha_fn)
        dynamic_alpha = wc_weight * self.dynamic_alpha_scale
        static_alpha = rearrange(self.static_alpha, "(f s) d -> f s d", s=streams)
        alpha = dynamic_alpha + static_alpha
        alpha = self.split_fracs(alpha)

        beta = None
        if self.add_branch_out_to_residual:
            dc_weight = self.act(normed @ self.dynamic_beta_fn)
            if not self.has_fracs:
                dc_weight = rearrange(dc_weight, "... -> ... 1")
            dynamic_beta = dc_weight * self.dynamic_beta_scale
            static_beta = rearrange(self.static_beta, "... (s f) -> ... s f", s=streams)
            beta = dynamic_beta + static_beta

        mix_h = einsum(alpha, residuals, "... f1 s f2 t, ... f1 s d -> ... f2 t d")

        if self.num_input_views == 1:
            branch_input, residuals = mix_h[..., 0, :], mix_h[..., 1:, :]
        else:
            branch_input, residuals = mix_h[..., : self.num_input_views, :], mix_h[..., self.num_input_views :, :]
            branch_input = rearrange(branch_input, "b ... v d -> v b ... d")

        if self.channel_first:
            branch_input = rearrange(branch_input, "b ... d -> b d ...")

        branch_input = self.merge_fracs(branch_input)

        residuals = rearrange(residuals, "b ... s d -> (b s) ... d")
        if self.channel_first:
            residuals = rearrange(residuals, "b ... d -> b d ...")
        residuals = self.merge_fracs(residuals)
        return branch_input, residuals, dict(beta=beta)

    def depth_connection(self, branch_output, residuals, *, beta):
        assert self.add_branch_out_to_residual
        branch_output = self.split_fracs(branch_output)
        if self.channel_first:
            branch_output = rearrange(branch_output, "b d ... -> b ... d")
        output = einsum(branch_output, beta, "b ... f1 d, b ... f1 s f2 -> b ... f2 s d")
        output = rearrange(output, "b ... s d -> (b s) ... d")
        output = self.merge_fracs(output)
        if self.channel_first:
            output = rearrange(output, "b ... d -> b d ...")
        residuals = self.depth_residual_fn(output, residuals)
        return self.dropout(residuals)

    def forward(self, residuals, *branch_args, **branch_kwargs):
        branch_input, residuals, residual_kwargs = self.width_connection(residuals)

        def add_residual_fn(branch_out):
            if not self.add_branch_out_to_residual:
                return branch_out
            # In this runner, branches always return a Tensor.
            return self.depth_connection(branch_out, residuals, **residual_kwargs)

        if not exists(self.branch):
            return branch_input, add_residual_fn

        branch_output = self.branch(branch_input, *branch_args, **branch_kwargs)
        return add_residual_fn(branch_output)


def sinkhorn_knopps(log_alpha: torch.Tensor, iters: int = 20) -> torch.Tensor:
    for _ in range(iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
    return log_alpha.exp()


class ManifoldConstrainedHyperConnections(nn.Module):
    def __init__(
        self,
        num_residual_streams: int,
        *,
        dim: int,
        branch: nn.Module | None = None,
        layer_index: int | None = None,
        channel_first: bool = False,
        dropout: float = 0.0,
        residual_transform: nn.Module | None = None,
        add_branch_out_to_residual: bool = True,
        num_input_views: int = 1,
        depth_residual_fn=add,
        num_fracs: int = 1,
        sinkhorn_iters: int = 20,
    ):
        super().__init__()
        self.branch = branch
        assert num_fracs >= 1
        self.num_fracs = num_fracs
        self.has_fracs = num_fracs > 1

        self.split_fracs = Rearrange("b ... (f d) -> b ... f d", f=num_fracs)
        self.merge_fracs = Rearrange("b ... f d -> b ... (f d)")
        assert divisible_by(dim, num_fracs)
        dim_eff = dim // num_fracs

        assert num_residual_streams > 0
        self.num_residual_streams = num_residual_streams
        init_residual_index = default(layer_index, random.randrange(num_residual_streams)) % num_residual_streams

        num_residual_streams_fracs = num_residual_streams * num_fracs
        num_input_views_fracs = num_input_views * num_fracs
        self.num_input_views = num_input_views

        self.norm = RMSNorm(dim_eff * num_residual_streams_fracs)

        init_alpha0 = torch.ones((num_residual_streams_fracs, num_input_views_fracs)) * -1
        init_alpha0[init_residual_index, :] = 1.0
        init_alpha1 = torch.ones((num_residual_streams_fracs, num_residual_streams_fracs)) * -8
        init_alpha1.fill_diagonal_(0.0)
        self.static_alpha = nn.Parameter(torch.cat((init_alpha0, init_alpha1), dim=1))

        self.dynamic_alpha_fn = nn.Parameter(
            torch.zeros(
                dim_eff * num_residual_streams,
                num_fracs * (num_residual_streams * num_residual_streams + num_residual_streams * num_input_views),
            )
        )
        self.pre_branch_scale = nn.Parameter(torch.ones(1) * 1e-2)
        self.residual_scale = nn.Parameter(torch.ones(1) * 1e-2)

        self.add_branch_out_to_residual = add_branch_out_to_residual
        if add_branch_out_to_residual:
            beta_init = torch.ones(num_residual_streams_fracs) * -1.0
            beta_init[init_residual_index] = 1.0
            self.static_beta = nn.Parameter(beta_init)
            self.dynamic_beta_fn = nn.Parameter(torch.zeros(dim_eff * num_residual_streams, num_fracs * num_residual_streams))
            self.h_post_scale = nn.Parameter(torch.ones(()) * 1e-2)

        self.sinkhorn_iters = sinkhorn_iters
        self.dropout = nn.Dropout(dropout)
        self.channel_first = channel_first
        self.residual_transform = default(residual_transform, nn.Identity())
        self.depth_residual_fn = depth_residual_fn

    def width_connection(self, residuals):
        streams = self.num_residual_streams

        if self.channel_first:
            residuals = rearrange(residuals, "b d ... -> b ... d")

        residuals = self.split_fracs(residuals)
        residuals = rearrange(residuals, "(b s) ... d -> b ... s d", s=streams)

        normed = rearrange(residuals, "b ... s d -> b ... (s d)", s=streams)
        normed = self.norm(normed)

        wc_weight = normed @ self.dynamic_alpha_fn
        wc_weight = rearrange(wc_weight, "... (s t) -> ... s t", s=streams)

        pre_branch_scale = repeat(self.pre_branch_scale, "1 -> v", v=self.num_input_views * self.num_fracs)
        residual_scale = repeat(self.residual_scale, "1 -> s", s=self.num_fracs * streams)
        alpha_scale = torch.cat((pre_branch_scale, residual_scale))
        dynamic_alpha = wc_weight * alpha_scale

        static_alpha = rearrange(self.static_alpha, "(f s) t -> f s t", s=streams)
        alpha = dynamic_alpha + static_alpha
        alpha = self.split_fracs(alpha)

        alpha_pre, alpha_residual = alpha[..., : self.num_input_views], alpha[..., self.num_input_views :]
        alpha_pre = alpha_pre.sigmoid()

        alpha_residual = rearrange(alpha_residual, "... f s g t -> ... f g s t")
        alpha_residual = sinkhorn_knopps(alpha_residual, self.sinkhorn_iters)
        alpha_residual = rearrange(alpha_residual, "... f g s t -> ... f s g t")

        alpha = torch.cat((alpha_pre, alpha_residual), dim=-1)

        beta = None
        if self.add_branch_out_to_residual:
            dc_weight = normed @ self.dynamic_beta_fn
            dc_weight = rearrange(dc_weight, "... (s f) -> ... s f", s=streams)
            dynamic_beta = dc_weight * self.h_post_scale
            static_beta = rearrange(self.static_beta, "... (s f) -> ... s f", s=streams)
            beta = dynamic_beta + static_beta
            beta = beta.sigmoid() * 2

        mix_h = einsum(alpha, residuals, "... f1 s f2 t, ... f1 s d -> ... f2 t d")

        if self.num_input_views == 1:
            branch_input, residuals = mix_h[..., 0, :], mix_h[..., 1:, :]
        else:
            branch_input, residuals = mix_h[..., : self.num_input_views, :], mix_h[..., self.num_input_views :, :]
            branch_input = rearrange(branch_input, "b ... v d -> v b ... d")

        if self.channel_first:
            branch_input = rearrange(branch_input, "b ... d -> b d ...")

        branch_input = self.merge_fracs(branch_input)

        residuals = rearrange(residuals, "b ... s d -> (b s) ... d")
        if self.channel_first:
            residuals = rearrange(residuals, "b ... d -> b d ...")
        residuals = self.merge_fracs(residuals)
        return branch_input, residuals, dict(beta=beta)

    def depth_connection(self, branch_output, residuals, *, beta):
        assert self.add_branch_out_to_residual
        branch_output = self.split_fracs(branch_output)
        if self.channel_first:
            branch_output = rearrange(branch_output, "b d ... -> b ... d")
        output = einsum(branch_output, beta, "b ... f1 d, b ... f1 s f2 -> b ... f2 s d")
        output = rearrange(output, "b ... s d -> (b s) ... d")
        output = self.merge_fracs(output)
        if self.channel_first:
            output = rearrange(output, "b ... d -> b d ...")
        residuals = self.depth_residual_fn(output, residuals)
        return self.dropout(residuals)

    def forward(self, residuals, *branch_args, **branch_kwargs):
        branch_input, residuals, residual_kwargs = self.width_connection(residuals)

        def add_residual_fn(branch_out):
            if not self.add_branch_out_to_residual:
                return branch_out
            # In this runner, branches always return a Tensor.
            return self.depth_connection(branch_out, residuals, **residual_kwargs)

        if not exists(self.branch):
            return branch_input, add_residual_fn

        branch_output = self.branch(branch_input, *branch_args, **branch_kwargs)
        return add_residual_fn(branch_output)


def get_all_permutations(n: int) -> torch.Tensor:
    import itertools

    perms = list(itertools.permutations(range(n)))
    index = torch.tensor(perms, dtype=torch.long)
    eye = torch.eye(n, dtype=torch.float32)
    return eye[index]  # (n!, n, n)


_perm_mats: dict[tuple[int, str], torch.Tensor] = {}


class MHCLite(nn.Module):
    def __init__(
        self,
        num_residual_streams: int,
        *,
        dim: int,
        branch: nn.Module | None = None,
        layer_index: int | None = None,
        channel_first: bool = False,
        dropout: float = 0.0,
        residual_transform: nn.Module | None = None,
        add_branch_out_to_residual: bool = True,
        num_input_views: int = 1,
        depth_residual_fn=add,
        num_fracs: int = 1,
    ):
        super().__init__()
        self.branch = branch
        assert num_fracs >= 1
        self.num_fracs = num_fracs
        self.has_fracs = num_fracs > 1

        self.split_fracs = Rearrange("b ... (f d) -> b ... f d", f=num_fracs)
        self.merge_fracs = Rearrange("b ... f d -> b ... (f d)")
        assert divisible_by(dim, num_fracs)
        dim_eff = dim // num_fracs

        assert num_residual_streams > 0
        self.num_residual_streams = num_residual_streams
        init_residual_index = default(layer_index, random.randrange(num_residual_streams)) % num_residual_streams

        num_residual_streams_fracs = num_residual_streams * num_fracs
        num_input_views_fracs = num_input_views * num_fracs
        self.num_input_views = num_input_views

        self.norm = RMSNorm(dim_eff * num_residual_streams_fracs)

        if (num_residual_streams, "cpu") not in _perm_mats:
            _perm_mats[(num_residual_streams, "cpu")] = get_all_permutations(num_residual_streams).to("cpu")
        perms_cpu = _perm_mats[(num_residual_streams, "cpu")]

        init_alpha0 = torch.ones((num_residual_streams_fracs, num_input_views_fracs)) * -1
        init_alpha0[init_residual_index, :] = 1.0
        init_alpha1 = torch.ones(len(perms_cpu) * num_fracs) * -8
        init_alpha1[0] = 0.0

        self.static_alpha = nn.Parameter(torch.cat([init_alpha0.reshape(-1), init_alpha1], dim=-1))

        self.dynamic_alpha_fn = nn.Parameter(
            torch.zeros(dim_eff * num_residual_streams, num_fracs * (len(perms_cpu) + num_residual_streams * num_input_views))
        )
        self.pre_branch_scale = nn.Parameter(torch.ones(1) * 1e-2)
        self.residual_scale = nn.Parameter(torch.ones(1) * 1e-2)

        self.add_branch_out_to_residual = add_branch_out_to_residual
        if add_branch_out_to_residual:
            beta_init = torch.ones(num_residual_streams_fracs) * -1.0
            beta_init[init_residual_index] = 1.0
            self.static_beta = nn.Parameter(beta_init)
            self.dynamic_beta_fn = nn.Parameter(torch.zeros(dim_eff * num_residual_streams, num_fracs * num_residual_streams))
            self.h_post_scale = nn.Parameter(torch.ones(()) * 1e-2)

        self.dropout = nn.Dropout(dropout)
        self.channel_first = channel_first
        self.residual_transform = default(residual_transform, nn.Identity())
        self.depth_residual_fn = depth_residual_fn

    def width_connection(self, residuals):
        streams = self.num_residual_streams

        if self.channel_first:
            residuals = rearrange(residuals, "b d ... -> b ... d")

        residuals = self.split_fracs(residuals)
        residuals = rearrange(residuals, "(b s) ... d -> b ... s d", s=streams)

        normed = rearrange(residuals, "b ... s d -> b ... (s d)", s=streams)
        normed = self.norm(normed)

        wc_weight = normed @ self.dynamic_alpha_fn
        psize = self.num_input_views * streams
        dynamic_pre, dynamic_residual = wc_weight[..., :psize], wc_weight[..., psize:]
        static_pre, static_residual = self.static_alpha[:psize], self.static_alpha[psize:]

        dev = str(wc_weight.device)
        if (streams, dev) not in _perm_mats:
            _perm_mats[(streams, dev)] = get_all_permutations(streams).to(wc_weight.device)
        perms = _perm_mats[(streams, dev)]

        res_coeff = self.residual_scale * dynamic_residual + static_residual
        res_coeff = torch.softmax(res_coeff, dim=-1)
        alpha_residual = einsum(res_coeff, perms, "... r, r i j-> ... i j")
        alpha_residual = self.split_fracs(alpha_residual)

        alpha_pre = self.pre_branch_scale * dynamic_pre + static_pre
        alpha_pre = rearrange(alpha_pre, "... (f s v) -> ... s f v", v=self.num_input_views, f=self.num_fracs)
        alpha_pre = alpha_pre.sigmoid()

        alpha = torch.cat((alpha_pre, alpha_residual), dim=-1)

        beta = None
        if self.add_branch_out_to_residual:
            dc_weight = normed @ self.dynamic_beta_fn
            dc_weight = rearrange(dc_weight, "... (s f) -> ... s f", s=streams)
            dynamic_beta = dc_weight * self.h_post_scale
            static_beta = rearrange(self.static_beta, "... (s f) -> ... s f", s=streams)
            beta = dynamic_beta + static_beta
            beta = beta.sigmoid() * 2

        mix_h = einsum(alpha, residuals, "... f1 s f2 t, ... f1 s d -> ... f2 t d")

        if self.num_input_views == 1:
            branch_input, residuals = mix_h[..., 0, :], mix_h[..., 1:, :]
        else:
            branch_input, residuals = mix_h[..., : self.num_input_views, :], mix_h[..., self.num_input_views :, :]
            branch_input = rearrange(branch_input, "b ... v d -> v b ... d")

        if self.channel_first:
            branch_input = rearrange(branch_input, "b ... d -> b d ...")

        branch_input = self.merge_fracs(branch_input)

        residuals = rearrange(residuals, "b ... s d -> (b s) ... d")
        if self.channel_first:
            residuals = rearrange(residuals, "b ... d -> b d ...")
        residuals = self.merge_fracs(residuals)
        return branch_input, residuals, dict(beta=beta)

    def depth_connection(self, branch_output, residuals, *, beta):
        assert self.add_branch_out_to_residual
        branch_output = self.split_fracs(branch_output)
        if self.channel_first:
            branch_output = rearrange(branch_output, "b d ... -> b ... d")
        output = einsum(branch_output, beta, "b ... f1 d, b ... f1 s f2 -> b ... f2 s d")
        output = rearrange(output, "b ... s d -> (b s) ... d")
        output = self.merge_fracs(output)
        if self.channel_first:
            output = rearrange(output, "b ... d -> b d ...")
        residuals = self.depth_residual_fn(output, residuals)
        return self.dropout(residuals)

    def forward(self, residuals, *branch_args, **branch_kwargs):
        branch_input, residuals, residual_kwargs = self.width_connection(residuals)

        def add_residual_fn(branch_out):
            if not self.add_branch_out_to_residual:
                return branch_out
            # In this runner, branches always return a Tensor.
            return self.depth_connection(branch_out, residuals, **residual_kwargs)

        if not exists(self.branch):
            return branch_input, add_residual_fn

        branch_output = self.branch(branch_input, *branch_args, **branch_kwargs)
        return add_residual_fn(branch_output)


class ExpandStreams(nn.Module):
    def __init__(self, num_streams: int):
        super().__init__()
        self.num_streams = int(num_streams)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_streams == 1:
            return x
        return x.repeat_interleave(self.num_streams, dim=0)


class ReduceStreams(nn.Module):
    def __init__(self, num_streams: int):
        super().__init__()
        self.num_streams = int(num_streams)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_streams == 1:
            return x
        return reduce(x, "(b s) ... d -> b ... d", "sum", s=self.num_streams)


def hyper_conn_init_func(hyper_conn_type: str, hyper_conn_n: int) -> tuple[Any, nn.Module | None, nn.Module | None]:
    # Match mhc-lite semantics:
    # - hyper_conn_type="none" still wraps blocks with Residual (disable=True)
    # - expand/reduce are identity when disabled or num_streams==1
    hyper_conn_type = "none" if hyper_conn_type == "residual" else hyper_conn_type

    if hyper_conn_type == "none":
        init = partial(Residual, hyper_conn_n, num_fracs=1)
        return init, nn.Identity(), nn.Identity()

    disable = (hyper_conn_n == 1)
    if hyper_conn_type == "hc":
        init = partial(HyperConnections if not disable else Residual, hyper_conn_n, num_fracs=1)
        if disable:
            return init, nn.Identity(), nn.Identity()
        return init, ExpandStreams(hyper_conn_n), ReduceStreams(hyper_conn_n)
    if hyper_conn_type == "mhc":
        init = partial(
            ManifoldConstrainedHyperConnections if not disable else Residual,
            hyper_conn_n,
            num_fracs=1,
            sinkhorn_iters=20,
        )
        if disable:
            return init, nn.Identity(), nn.Identity()
        return init, ExpandStreams(hyper_conn_n), ReduceStreams(hyper_conn_n)
    if hyper_conn_type == "mhc_lite":
        init = partial(MHCLite if not disable else Residual, hyper_conn_n, num_fracs=1)
        if disable:
            return init, nn.Identity(), nn.Identity()
        return init, ExpandStreams(hyper_conn_n), ReduceStreams(hyper_conn_n)
    raise ValueError(f"unknown hyper_conn_type={hyper_conn_type!r}")


# -----------------------------------------------------------------------------
# GPT model (NanoGPT-style, minimal)

class LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: "GPTConfig"):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config: "GPTConfig"):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config: "GPTConfig", init_hc):
        super().__init__()
        self.branch_attn = nn.Sequential(LayerNorm(config.n_embd, bias=config.bias), CausalSelfAttention(config))
        self.branch_mlp = nn.Sequential(LayerNorm(config.n_embd, bias=config.bias), MLP(config))

        self.hc_attn = None
        self.hc_mlp = None
        if init_hc is not None:
            self.hc_attn = init_hc(dim=config.n_embd, branch=self.branch_attn)
            self.hc_mlp = init_hc(dim=config.n_embd, branch=self.branch_mlp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.hc_attn is None:
            x = x + self.branch_attn(x)
            x = x + self.branch_mlp(x)
            return x
        x = self.hc_attn(x)
        x = self.hc_mlp(x)
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False
    hyper_conn_n: int = 1
    hyper_conn_type: str = "none"


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        init_hc, expand_stream, reduce_stream = hyper_conn_init_func(config.hyper_conn_type, config.hyper_conn_n)
        self.expand_stream = expand_stream
        self.reduce_stream = reduce_stream

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([Block(config, init_hc) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight  # weight tying

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.wpe.weight.numel()
        return int(n_params)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)
        if self.expand_stream is not None:
            x = self.expand_stream(x)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        if self.reduce_stream is not None:
            x = self.reduce_stream(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    def configure_optimizers(self, weight_decay: float, learning_rate: float, betas: tuple[float, float], device_type: str):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = {"fused": True} if use_fused else {}
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)


# -----------------------------------------------------------------------------
# Training / eval

@dataclass
class ScaleCfg:
    n_layer: int
    n_head: int
    n_embd: int
    learning_rate: float
    min_lr: float
    warmup_iters: int


SCALES: dict[str, ScaleCfg] = {
    "small": ScaleCfg(n_layer=6, n_head=8, n_embd=512, learning_rate=1e-3, min_lr=1e-4, warmup_iters=200),
    "medium": ScaleCfg(n_layer=12, n_head=12, n_embd=768, learning_rate=6e-4, min_lr=6e-5, warmup_iters=200),
    "large": ScaleCfg(n_layer=24, n_head=16, n_embd=1024, learning_rate=3e-4, min_lr=3e-5, warmup_iters=200),
}


METHODS: dict[str, dict[str, Any]] = {
    "residual": {"hyper_conn_type": "none", "hyper_conn_n": 1},
    "hc": {"hyper_conn_type": "hc", "hyper_conn_n": 4},
    "mhc": {"hyper_conn_type": "mhc", "hyper_conn_n": 4},
    "mhc_lite": {"hyper_conn_type": "mhc_lite", "hyper_conn_n": 4},
}


def get_lr(it: int, *, warmup_iters: int, lr_decay_iters: int, learning_rate: float, min_lr: float) -> float:
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0.0 <= decay_ratio <= 1.0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


@torch.no_grad()
def estimate_loss(
    *,
    model: nn.Module,
    ctx,
    eval_iters: int,
    get_batch,
) -> dict[str, float]:
    model.eval()
    out: dict[str, float] = {}

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = float(losses.mean().item())

    model.train()
    return out


def format_table(rows: list[dict[str, Any]], keys: list[str]) -> str:
    # simple markdown table
    cols = keys
    widths = {k: len(k) for k in cols}
    for r in rows:
        for k in cols:
            widths[k] = max(widths[k], len(str(r.get(k, ""))))
    header = "| " + " | ".join(k.ljust(widths[k]) for k in cols) + " |"
    sep = "| " + " | ".join("-" * widths[k] for k in cols) + " |"
    body = []
    for r in rows:
        body.append("| " + " | ".join(str(r.get(k, "")).ljust(widths[k]) for k in cols) + " |")
    return "\n".join([header, sep] + body)


def run_one(
    *,
    ddp: DDPInfo,
    method: str,
    scale: str,
    args: argparse.Namespace,
    train_paths: list[str],
    val_paths: list[str],
    writer: JsonlWriter | None,
) -> dict[str, Any]:
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    if scale not in SCALES:
        raise ValueError(f"unknown scale: {scale}")

    method_cfg = METHODS[method]
    scale_cfg = SCALES[scale]

    device = ddp.device
    device_type = "cuda" if device.type == "cuda" else "cpu"

    # Match mhc-lite: only seed torch, offset by rank in DDP.
    seed_offset = ddp.rank if ddp.ddp else 0
    torch.manual_seed(int(args.seed) + seed_offset)
    # For paper-style comparisons, it's useful for --seed to control HC init too.
    random.seed(int(args.seed) + seed_offset)

    if device_type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats()

    # Precompute shard token prefix sums, treating shards as a concatenated token stream.
    # This is the closest analogue to mhc-lite's single train.bin/val.bin sampling.
    train_token_counts = shard_token_counts(train_paths)
    val_token_counts = shard_token_counts(val_paths)

    train_tok_prefix_ends = torch.tensor(np.cumsum(train_token_counts), dtype=torch.int64)
    val_tok_prefix_ends = torch.tensor(np.cumsum(val_token_counts), dtype=torch.int64)
    train_tok_prefix_starts = torch.zeros_like(train_tok_prefix_ends)
    val_tok_prefix_starts = torch.zeros_like(val_tok_prefix_ends)
    if len(train_tok_prefix_ends) > 1:
        train_tok_prefix_starts[1:] = train_tok_prefix_ends[:-1]
    if len(val_tok_prefix_ends) > 1:
        val_tok_prefix_starts[1:] = val_tok_prefix_ends[:-1]
    train_total_tokens = int(train_tok_prefix_ends[-1].item())
    val_total_tokens = int(val_tok_prefix_ends[-1].item())

    block_size = int(args.block_size)
    batch_size = int(args.batch_size)
    pin = (device_type == "cuda")

    def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
        # Recreate np.memmap every batch (mhc-lite behavior, avoids memmap leak).
        if split == "train":
            paths = train_paths
            tok_prefix_ends = train_tok_prefix_ends
            tok_prefix_starts = train_tok_prefix_starts
            total_tokens = train_total_tokens
        else:
            paths = val_paths
            tok_prefix_ends = val_tok_prefix_ends
            tok_prefix_starts = val_tok_prefix_starts
            total_tokens = val_total_tokens

        # Like mhc-lite: sample start positions uniformly over the full token stream.
        gix = torch.randint(total_tokens - block_size, (batch_size,))

        memmaps: dict[int, np.memmap] = {}
        xs: list[torch.Tensor] = []
        ys: list[torch.Tensor] = []
        for start in gix:
            # We fetch a contiguous window of length (block_size + 1) and then split into x/y.
            need = block_size + 1

            shard_id = int(torch.searchsorted(tok_prefix_ends, start, right=True))
            local = int(start - tok_prefix_starts[shard_id])

            if shard_id not in memmaps:
                memmaps[shard_id] = np.memmap(paths[shard_id], dtype=np.uint16, mode="r")
            data0 = memmaps[shard_id]
            if local + need <= len(data0):
                window = data0[local : local + need]
            else:
                parts: list[np.ndarray] = []
                while need > 0:
                    if shard_id not in memmaps:
                        memmaps[shard_id] = np.memmap(paths[shard_id], dtype=np.uint16, mode="r")
                    data = memmaps[shard_id]
                    take = min(need, len(data) - local)
                    parts.append(data[local : local + take])
                    need -= take
                    shard_id += 1
                    local = 0
                window = np.concatenate(parts, axis=0)
            x_np = window[:-1].astype(np.int64)
            y_np = window[1:].astype(np.int64)
            xs.append(torch.from_numpy(x_np))
            ys.append(torch.from_numpy(y_np))

        x = torch.stack(xs)
        y = torch.stack(ys)
        if pin:
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    # model config
    cfg = GPTConfig(
        block_size=args.block_size,
        vocab_size=args.vocab_size,
        n_layer=scale_cfg.n_layer,
        n_head=scale_cfg.n_head,
        n_embd=scale_cfg.n_embd,
        dropout=args.dropout,
        bias=args.bias,
        hyper_conn_type=method_cfg["hyper_conn_type"],
        hyper_conn_n=method_cfg["hyper_conn_n"],
    )

    model = GPT(cfg).to(device)

    # amp / dtype
    dtype = args.dtype
    ptdtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]

    # mhc-lite: ctx is nullcontext on cpu, else autocast(device_type, dtype)
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # mhc-lite: GradScaler enabled iff dtype == float16
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

    # optimizer / schedule
    lr = float(args.lr) if args.lr is not None else float(scale_cfg.learning_rate)
    min_lr = float(args.min_lr) if args.min_lr is not None else float(scale_cfg.min_lr)
    warmup = int(args.warmup) if args.warmup is not None else int(scale_cfg.warmup_iters)
    lr_decay_iters = int(args.lr_decay_iters) if args.lr_decay_iters is not None else int(args.steps)

    grad_accum_global = int(args.grad_accum)
    if ddp.ddp:
        assert grad_accum_global % ddp.world_size == 0
        grad_accum = grad_accum_global // ddp.world_size
    else:
        grad_accum = grad_accum_global

    model_for_train = model
    if args.compile:
        model_for_train = torch.compile(model_for_train)

    if ddp.ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP

        model_for_train = DDP(model_for_train, device_ids=[ddp.local_rank] if device_type == "cuda" else None)

    optimizer = model.configure_optimizers(args.weight_decay, lr, (args.beta1, args.beta2), device_type)

    tokens_per_iter = grad_accum * ddp.world_size * args.batch_size * args.block_size

    # initial batch
    X, Y = get_batch("train")

    t_start = time.time()
    tps_all: list[float] = []
    last_eval: dict[str, float] | None = None

    if ddp.master:
        print(f"\n[run] method={method} scale={scale} max_iters={args.steps} device={device} world_size={ddp.world_size}")
        print(f"      params={model.get_num_params()/1e6:.2f}M tokens/iter={tokens_per_iter:,} grad_accum_local={grad_accum}")

    max_iters = int(args.steps)
    for it in range(max_iters + 1):
        # eval
        if (args.eval_every > 0) and (it % args.eval_every == 0) and ddp.master:
            losses = estimate_loss(model=model_for_train, ctx=ctx, eval_iters=int(args.eval_iters), get_batch=get_batch)
            last_eval = losses
            if writer is not None:
                writer.write(
                    {
                        "type": "eval",
                        "method": method,
                        "scale": scale,
                        "iter": it,
                        "train_loss": losses["train"],
                        "val_loss": losses["val"],
                        "time_sec": time.time() - t_start,
                    }
                )
            print(f"[eval] it={it:6d} train_loss={losses['train']:.4f} val_loss={losses['val']:.4f}")

        # lr update
        cur_lr = get_lr(it, warmup_iters=warmup, lr_decay_iters=lr_decay_iters, learning_rate=lr, min_lr=min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        # train step
        step_t0 = time.time()
        for micro in range(grad_accum):
            if ddp.ddp:
                model_for_train.require_backward_grad_sync = (micro == grad_accum - 1)
            with ctx:
                _, loss = model_for_train(X, Y)
                loss = loss / grad_accum

            # prefetch next batch
            X, Y = get_batch("train")

            scaler.scale(loss).backward()

        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model_for_train.parameters(), args.grad_clip)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        dt = time.time() - step_t0
        tps = tokens_per_iter / dt
        tps_all.append(float(tps))

        if ddp.master and (args.log_every > 0) and (it % args.log_every == 0):
            mem_gb = 0.0
            if device_type == "cuda":
                mem_gb = float(torch.cuda.max_memory_allocated() / (1024**3))
            if writer is not None:
                writer.write(
                    {
                        "type": "train",
                        "method": method,
                        "scale": scale,
                        "iter": it,
                        "lr": cur_lr,
                        "tps": float(tps),
                        "mem_gb": mem_gb,
                        "time_sec": time.time() - t_start,
                    }
                )
            print(f"[train] it={it:6d} lr={cur_lr:.2e} tps={tps:,.0f} dt_ms={dt*1000:.2f}")

    total_time = time.time() - t_start
    mem_gb = 0.0
    if device_type == "cuda":
        mem_gb = float(torch.cuda.max_memory_allocated() / (1024**3))

    final_losses = last_eval or {"train": float("nan"), "val": float("nan")}

    summary = {
        "method": method,
        "scale": scale,
        "max_iters": max_iters,
        "steps_run": max_iters + 1,
        "final_train_loss": final_losses["train"],
        "final_val_loss": final_losses["val"],
        "avg_tps": float(np.mean(tps_all)) if tps_all else 0.0,
        "peak_mem_gb": mem_gb,
        "total_time_sec": float(total_time),
        "params": int(model.get_num_params()),
        "world_size": int(ddp.world_size),
        "batch_size": int(args.batch_size),
        "grad_accum_global": int(args.grad_accum),
        "block_size": int(args.block_size),
        "seed": int(args.seed),
    }

    # cleanup
    del model_for_train, model, optimizer, scaler
    gc.collect()
    if device_type == "cuda":
        torch.cuda.empty_cache()
    ddp_barrier(ddp)

    if ddp.master:
        print(
            f"[done] method={method} scale={scale} "
            f"val_loss={summary['final_val_loss']:.4f} avg_tps={summary['avg_tps']:,.0f} "
            f"mem_gb={summary['peak_mem_gb']:.2f} time_sec={summary['total_time_sec']:.1f}"
        )

    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    data_root_default = os.environ.get("DATA_PATH") or os.environ.get("DATA_ROOT") or "."
    p.add_argument("--data_root", type=str, default=data_root_default, help="root containing data/fineweb10B/")
    # Optional: auto-download the dataset snapshot from ModelScope (repo_type=dataset)
    p.add_argument("--ms_dataset_id", type=str, default="", help="ModelScope dataset id (owner/name). If set, overrides --data_root.")
    p.add_argument("--ms_revision", type=str, default="", help="Optional ModelScope revision/tag/commit")
    p.add_argument("--ms_cache_dir", type=str, default="", help="Optional ModelScope cache_dir")
    p.add_argument("--train_glob", type=str, default="data/fineweb10B/fineweb_train_*.bin")
    p.add_argument("--val_glob", type=str, default="data/fineweb10B/fineweb_val_*.bin")
    p.add_argument("--auto_glob", type=int, default=1, help="auto-detect shard layout if globs match nothing (1/0)")

    p.add_argument("--methods", type=str, default="residual,hc,mhc,mhc_lite")
    p.add_argument("--scales", type=str, default="small,medium,large")

    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--eval_iters", type=int, default=200)
    p.add_argument("--log_every", type=int, default=10)

    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out_dir", type=str, default=os.path.join("runs", now_ts()))

    # model/training
    p.add_argument("--block_size", type=int, default=1024)
    p.add_argument("--vocab_size", type=int, default=50304)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--bias", action="store_true", help="use bias in LayerNorm/Linear (default: False)")

    p.add_argument("--batch_size", type=int, default=16, help="micro-batch per process")
    p.add_argument("--grad_accum", type=int, default=8, help="global grad accumulation steps (DDP divides per-rank)")

    p.add_argument(
        "--dtype",
        type=str,
        default=None,
        choices=["float16", "bfloat16", "float32"],
        help="float16/bfloat16/float32. Default matches mhc-lite: bfloat16 if CUDA bf16 is supported, else float16.",
    )
    p.add_argument("--compile", type=int, default=0)

    # optimizer / schedule (None => scale default)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--min_lr", type=float, default=None)
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument("--lr_decay_iters", type=int, default=None)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--ddp_backend", type=str, default=None, help="override DDP backend (nccl/gloo)")

    args = p.parse_args()
    args.compile = bool(int(args.compile))

    ddp = ddp_init(args.ddp_backend)
    if args.dtype is None:
        if ddp.device.type == "cpu":
            args.dtype = "float32"
        else:
            args.dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    if ddp.master:
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"out_dir={args.out_dir}")

    if args.ms_dataset_id:
        try:
            from modelscope.hub.snapshot_download import snapshot_download
        except Exception as e:
            raise RuntimeError(
                "--ms_dataset_id was set but `modelscope` is not available. "
                "Install it (pip install modelscope) or omit --ms_dataset_id and use --data_root."
            ) from e

        kwargs: dict[str, Any] = {"repo_type": "dataset"}
        if args.ms_revision:
            kwargs["revision"] = args.ms_revision
        if args.ms_cache_dir:
            kwargs["cache_dir"] = os.path.expanduser(args.ms_cache_dir)
        ms_path = snapshot_download(args.ms_dataset_id, **kwargs)
        args.data_root = ms_path
        if ddp.master:
            print(f"ModelScope snapshot_download: ms_dataset_id={args.ms_dataset_id!r} -> data_root={args.data_root!r}")

    if bool(int(args.auto_glob)):
        train_paths, val_paths, args.train_glob, args.val_glob = resolve_shards_auto(args.data_root, args.train_glob, args.val_glob)
    else:
        train_paths = resolve_shards(args.data_root, args.train_glob)
        val_paths = resolve_shards(args.data_root, args.val_glob)
    if ddp.master:
        print(f"train_shards={len(train_paths)} val_shards={len(val_paths)}")
        print(f"train_glob={args.train_glob!r}")
        print(f"val_glob={args.val_glob!r}")
    if not train_paths or not val_paths:
        raise FileNotFoundError(
            f"no shards matched under data_root={args.data_root!r}: train_glob={args.train_glob!r} val_glob={args.val_glob!r}"
        )

    methods = parse_csv(args.methods)
    scales = parse_csv(args.scales)

    writer = JsonlWriter(os.path.join(args.out_dir, "results.jsonl")) if ddp.master else None
    summaries: list[dict[str, Any]] = []

    # record environment
    if ddp.master:
        env = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(ddp.device),
            "world_size": ddp.world_size,
            "args": vars(args),
        }
        with open(os.path.join(args.out_dir, "env.json"), "w", encoding="utf-8") as f:
            json.dump(env, f, indent=2, ensure_ascii=True)

    try:
        for scale in scales:
            for method in methods:
                ddp_barrier(ddp)
                summary = run_one(
                    ddp=ddp,
                    method=method,
                    scale=scale,
                    args=args,
                    train_paths=train_paths,
                    val_paths=val_paths,
                    writer=writer,
                )
                if ddp.master:
                    summaries.append(summary)
    finally:
        if writer is not None:
            writer.close()

    if ddp.master:
        # summary.json
        with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2, ensure_ascii=True)

        # pretty table
        rows = []
        for s in summaries:
            rows.append(
                {
                    "scale": s["scale"],
                    "method": s["method"],
                    "val_loss": f"{s['final_val_loss']:.4f}",
                    "train_loss": f"{s['final_train_loss']:.4f}",
                    "avg_tps": f"{s['avg_tps']:,.0f}",
                    "mem_gb": f"{s['peak_mem_gb']:.2f}",
                    "time_s": f"{s['total_time_sec']:.1f}",
                    "params_M": f"{s['params']/1e6:.2f}",
                }
            )
        print("\n" + format_table(rows, ["scale", "method", "val_loss", "train_loss", "avg_tps", "mem_gb", "time_s", "params_M"]))

    ddp_destroy(ddp)


if __name__ == "__main__":
    main()
