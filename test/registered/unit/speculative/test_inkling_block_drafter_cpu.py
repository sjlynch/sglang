"""CPU checks for the research block-drafter serving path (Infera task 5k0kt).

``inkling_block_drafter.py`` (SGLANG_INKLING2_MTP_BLOCK_ARCH) loads standalone (torch
only), as does ``draft_filter.py``. The tests check:

1. Exactness of the block proposal: one block forward (base logits for every depth)
   plus the Markov walk, verified by chain speculative sampling, outputs exactly the
   target's distribution (chi-square on the first three committed tokens of >= 4e4
   independent streams), alone, with a QISO-x stopping-time prefix, with the filtered
   draft (q' and p'), and all three together. Two broken controls must be detected:
   a width chosen by peeking at a draft outside the verified prefix, and a verify that
   reads a q other than the one the draft was drawn from.
2. Pinned identity: an empty flag is no arch; C1's merge passes every tensor through
   untouched when off; C2 at its zero init and a lifted control are exactly the init
   head; the C2 serving module equals the trainer's DepthExperts bit for bit
   (``drafter.py`` @ Infera 2bba1e77, copied below), including the final-norm hook.
3. The loader: the trainer's block_arch for every T8 arm (k1, k2, c1, c2) validates,
   unservable or inconsistent heads are refused, ``check_head`` catches a missing or
   reshaped arm tensor, and ``prepare_draft_dir`` writes the flag, the receipt and a
   lifted control that passes the same check.
"""

import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[4]


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bd = _load("inkling_block_drafter", "python/sglang/srt/speculative/inkling_block_drafter.py")
draft_filter = _load(
    "dspark_draft_filter", "python/sglang/srt/speculative/dspark_components/draft_filter.py"
)

STOCK_DRAFT = {"num_hidden_layers": 3, "sliding_window_size": 1024, "block_size": 8}


def trainer_arch(**parts):
    """block_arch exactly as drafter.py checkpoint_metadata writes it (2bba1e77)."""
    arch = dict(n_layers=3, local_layers=[0, 2], window=1024, init_layers=3, block_conv=0,
                conv_group=16, tap_fc=False, selector_rank=0, selector_topk=0)
    arch.update(parts)
    return arch


T8_ARMS = {
    "k1": trainer_arch(block_conv=2),
    "k2": trainer_arch(block_conv=2, tap_fc=True),  # T8 k2 = --block-conv 2 --tap-fc
    "c1": trainer_arch(ffn_extra=4608),
    "c2": trainer_arch(depth_experts=512),
}


def chi2_sf(x: float, dof: int) -> float:
    """Upper tail of the chi-square distribution (regularised gamma Q)."""
    a, z = dof / 2.0, x / 2.0
    if z <= 0:
        return 1.0
    if z < a + 1:  # series for P, then Q = 1 - P
        term = total = 1.0 / a
        n = a
        while abs(term) > 1e-15 * abs(total):
            n += 1
            term *= z / n
            total += term
        return max(0.0, 1.0 - total * math.exp(-z + a * math.log(z) - math.lgamma(a)))
    # continued fraction for Q
    b, c, d = z + 1 - a, 1e300, 1 / (z + 1 - a)
    h = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return h * math.exp(-z + a * math.log(z) - math.lgamma(a))


# ---------------------------------------------------------------------------
# 1. Exactness of block proposal + chain verify (QISO-x prefix, filtered draft)
# ---------------------------------------------------------------------------

V, GAMMA, TOKENS = 4, 7, 3


class ToyWorld:
    """Target: a first-order chain p(x_t | x_{t-1}) = P[x_{t-1}] over V tokens.
    Block drafter: ONE forward from the anchor gives base logits for all 7 depths
    (B[anchor, j]), and the Markov walk adds a bias from the previous draft
    (M[x_{j-1}]): q_j = softmax(B[a, j] + M[x_{j-1}]), the DSPARK proposal shape."""

    def __init__(self, seed=5, top_k=None):
        g = torch.Generator().manual_seed(seed)
        self.P = torch.softmax(torch.randn(V, V, generator=g, dtype=torch.float64) * 1.2, -1)
        self.B = torch.randn(V, GAMMA, V, generator=g, dtype=torch.float64)
        self.M = torch.randn(V, V, generator=g, dtype=torch.float64) * 2.0
        self.top_k = top_k

    def target_rows(self, prefix_tokens):
        p = self.P[prefix_tokens]
        if self.top_k is None:
            return p
        return self.filter_probs(p)

    def filter_probs(self, p):
        """The request's top-k applied to the target row (what the verifier samples)."""
        keep = draft_filter.reference_keep_mask
        out = torch.zeros_like(p)
        flat = p.reshape(-1, V)
        for i, row in enumerate(flat):
            mask = keep(row.log(), 1.0, self.top_k, 1.0)
            kept = row * mask
            out.view(-1, V)[i] = kept / kept.sum()
        return out

    def exact_joint(self):
        """Exact distribution of the first TOKENS tokens after anchor 0."""
        joint = torch.zeros((V,) * TOKENS, dtype=torch.float64)
        P = self.P if self.top_k is None else self.filter_probs(self.P)
        for idx in torch.cartesian_prod(*[torch.arange(V)] * TOKENS):
            prob, prev = 1.0, 0
            for t in idx.tolist():
                prob *= float(P[prev, t])
                prev = t
            joint[tuple(idx.tolist())] = prob
        return joint


def run_streams(world, n, seed, *, width_rule=None, filtered=False, wrong_q=False):
    """n independent streams from anchor 0; rounds until each has TOKENS tokens.
    width_rule(drafts, q) -> widths [bs] in {4,5,6,8} (QISO-x); filtered: top-k on q
    (the filtered draft) and on p; wrong_q: verify reads the UNfiltered q although the
    drafts were drawn from the filtered q' (a broken control)."""
    g = torch.Generator().manual_seed(seed)
    out = torch.full((n, TOKENS), -1, dtype=torch.long)
    have = torch.zeros(n, dtype=torch.long)
    anchor = torch.zeros(n, dtype=torch.long)
    keep = None
    if filtered:
        def keep(j, logits):
            threshold = draft_filter.keep_thresholds(
                logits.float(), torch.ones(logits.shape[0]),
                draft_filter.DraftFilter(torch.full((logits.shape[0],), world.top_k),
                                         torch.ones(logits.shape[0]), 8),
            )
            return logits >= threshold.double().view(-1, 1)
    while (have < TOKENS).any():
        rows = torch.nonzero(have < TOKENS).view(-1)
        a = anchor[rows]
        base = world.B[a]  # one block forward: every depth's base logits at once
        drafts, q = bd.markov_block_reference(base, lambda j, prev: world.M[prev], a, g, keep=keep)
        if wrong_q:
            # recompute the unfiltered q along the SAME drafts
            logits = base + world.M[torch.cat((a.view(-1, 1), drafts[:, :-1]), 1)]
            q = torch.softmax(logits, -1)
        prefix = torch.cat((a.view(-1, 1), drafts), 1)  # p row j follows anchor + j drafts
        p = world.target_rows(prefix)
        widths = None if width_rule is None else width_rule(drafts, q)
        committed, accepted = bd.chain_verify_reference(p, q, drafts, g, widths)
        for local, row in enumerate(rows.tolist()):
            for token in committed[local].tolist():
                if token < 0 or have[row] >= TOKENS:
                    break
                out[row, have[row]] = token
                have[row] += 1
            last = committed[local][committed[local] >= 0][-1]
            anchor[row] = last
    return out


def chi2_against(joint, samples):
    counts = torch.zeros_like(joint)
    for row in samples.tolist():
        counts[tuple(row)] += 1
    n = samples.shape[0]
    expected = joint * n
    mask = expected > 5
    stat = float((((counts - expected) ** 2) / expected.clamp(min=1e-12))[mask].sum())
    lumped_o = float(counts[~mask].sum())
    lumped_e = float(expected[~mask].sum())
    cells = int(mask.sum())
    if lumped_e > 0:
        stat += (lumped_o - lumped_e) ** 2 / lumped_e
        cells += 1
    return stat, cells - 1


def qiso_stopping_rule(drafts, q, threshold=0.45):
    """A QISO-x-like stopping time: W = the first width in (4, 5, 6) whose LAST
    verified draft has q(x) < threshold, else 8. Reads only drafts inside the prefix."""
    rows = torch.arange(drafts.shape[0])
    widths = torch.full((drafts.shape[0],), 8, dtype=torch.long)
    decided = torch.zeros(drafts.shape[0], dtype=torch.bool)
    for width in (4, 5, 6):
        depth = width - 2  # index of the last verified draft
        low = q[rows, depth, drafts[:, depth]] < threshold
        take = low & ~decided
        widths[take] = width
        decided |= take
    return widths


def peeking_rule(drafts, q):
    """Broken: verify only the first draft (width 2) when the SECOND draft, outside
    that prefix, is token 0; else all seven. Not a stopping time: the choice reads a
    draft the verify never sees, which the Markov walk ties to the first one."""
    return torch.where(drafts[:, 1] == 0, torch.full_like(drafts[:, 0], 2), torch.full_like(drafts[:, 0], 8))


class BlockProposalExactness(unittest.TestCase):
    N = int(os.environ.get("BLOCK_DRAFTER_CHI2_N", "40000"))

    def check(self, world, label, seed, **kw):
        samples = run_streams(world, self.N, seed, **kw)
        stat, dof = chi2_against(world.exact_joint(), samples)
        p_value = chi2_sf(stat, dof)
        print(f"[chi2] {label}: chi2={stat:.1f} dof={dof} p={p_value:.3g} n={self.N}", flush=True)
        return p_value

    def test_block_proposal_is_exact(self):
        self.assertGreater(self.check(ToyWorld(), "block w8", 11), 1e-4)

    def test_qiso_prefix_is_exact(self):
        self.assertGreater(self.check(ToyWorld(), "block + QISO-x stopping prefix", 12,
                                      width_rule=qiso_stopping_rule), 1e-4)

    def test_filtered_draft_is_exact(self):
        world = ToyWorld(top_k=2)
        self.assertGreater(self.check(world, "block + filtered draft (top-2)", 13, filtered=True), 1e-4)

    def test_all_composed_is_exact(self):
        world = ToyWorld(top_k=3)
        self.assertGreater(self.check(world, "block + QISO-x + filtered (top-3)", 14,
                                      filtered=True, width_rule=qiso_stopping_rule), 1e-4)

    def test_peeking_width_is_detected(self):
        self.assertLess(self.check(ToyWorld(), "BROKEN: width peeks outside the prefix", 15,
                                   width_rule=peeking_rule), 1e-6)

    def test_wrong_q_is_detected(self):
        world = ToyWorld(top_k=2)
        self.assertLess(self.check(world, "BROKEN: verify reads unfiltered q", 16,
                                   filtered=True, wrong_q=True), 1e-6)

    def test_closed_form_acceptance(self):
        """E[accepted at depth 1] = sum min(p, q) (the served-path identity)."""
        world = ToyWorld()
        g = torch.Generator().manual_seed(3)
        n = 20000
        a = torch.zeros(n, dtype=torch.long)
        drafts, q = bd.markov_block_reference(world.B[a], lambda j, prev: world.M[prev], a, g)
        p = world.target_rows(torch.cat((a.view(-1, 1), drafts), 1))
        _, accepted = bd.chain_verify_reference(p, q, drafts, g)
        expected = float(torch.minimum(p[0, 0], q[0, 0]).sum())
        observed = float((accepted >= 1).double().mean())
        self.assertLess(abs(observed - expected), 4 * math.sqrt(expected * (1 - expected) / n))


# ---------------------------------------------------------------------------
# 2. Pinned identity
# ---------------------------------------------------------------------------


def trainer_rms(x, w, eps=1e-6):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(x.dtype)


class TrainerDepthExperts(torch.nn.Module):
    """drafter.py DepthExperts @ Infera 2bba1e77 (verbatim math, [M, B, H] layout)."""

    def __init__(self, rank, hidden, depths=7):
        super().__init__()
        self.rank, self.depths = rank, depths
        self.row_embed = torch.nn.Parameter(torch.zeros(depths, hidden))
        self.norm = torch.nn.Parameter(torch.ones(hidden))
        self.gate = torch.nn.Parameter(torch.zeros(depths, rank, hidden))
        self.up = torch.nn.Parameter(torch.zeros(depths, rank, hidden))
        self.down = torch.nn.Parameter(torch.zeros(depths, hidden, rank))
        active = [math.ceil(rank * (j + 1) / depths) for j in range(depths)]
        self.register_buffer("mask", (torch.arange(rank)[None] < torch.tensor(active)[:, None]).float(),
                             persistent=False)

    def embed(self, x):
        rows = x.shape[1] - 1
        return torch.cat((x[:, :1], x[:, 1:] + self.row_embed[:rows].to(x.dtype)), dim=1)

    def forward(self, residual):
        rows = residual[:, 1:]
        D = rows.shape[1]
        h = trainer_rms(rows, self.norm)
        a = torch.einsum("mdh,drh->mdr", h, self.gate[:D].to(h.dtype))
        b = torch.einsum("mdh,drh->mdr", h, self.up[:D].to(h.dtype))
        code = F.silu(a) * b * self.mask[:D].to(h.dtype)
        out = torch.einsum("mdr,dhr->mdh", code, self.down[:D].to(code.dtype))
        return torch.cat((residual[:, :1], rows + out), dim=1)


class ToyNorm(torch.nn.Module):
    """SGLang RMSNorm's contract: norm(x) or norm(x, residual) -> (rms(x + r), x + r)."""

    def __init__(self, hidden):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(hidden).to(torch.bfloat16))

    def forward(self, x, residual=None):
        if residual is None:
            return trainer_rms(x, self.weight)
        merged = x + residual
        return trainer_rms(merged, self.weight), merged


def random_depth_experts(rank, hidden, seed, dtype=torch.bfloat16):
    g = torch.Generator().manual_seed(seed)
    tensors = {
        "row_embed": torch.randn(7, hidden, generator=g) * 0.1,
        "norm": 1 + torch.randn(hidden, generator=g) * 0.1,
        "gate": torch.randn(7, rank, hidden, generator=g) / math.sqrt(hidden),
        "up": torch.randn(7, rank, hidden, generator=g) / math.sqrt(hidden),
        "down": torch.randn(7, hidden, rank, generator=g) / math.sqrt(rank),
    }
    return {k: v.to(dtype) for k, v in tensors.items()}


class PinnedIdentity(unittest.TestCase):
    def test_empty_flag_is_no_arch(self):
        self.assertIsNone(bd.parse_block_arch("", STOCK_DRAFT))
        self.assertIsNone(bd.parse_block_arch(None, STOCK_DRAFT))

    def test_ffn_merge_off_is_passthrough(self):
        tensors = [("layers.0.mlp.gate_proj.weight", torch.randn(4, 2)), ("fuse_weight", torch.randn(4, 4))]
        out = list(bd.merge_ffn_extra(iter(tensors), 0))
        self.assertEqual([n for n, _ in out], [n for n, _ in tensors])
        self.assertTrue(all(a is b for (_, a), (_, b) in zip(out, tensors)))

    def test_c1_merge_equals_stock_plus_extra(self):
        g = torch.Generator().manual_seed(0)
        hidden, inter, extra = 32, 48, 16
        names = {}
        tensors = []
        for i in range(3):
            for kind, shape, eshape in (("gate", (inter, hidden), (extra, hidden)),
                                        ("up", (inter, hidden), (extra, hidden)),
                                        ("down", (hidden, inter), (hidden, extra))):
                base = torch.randn(*shape, generator=g, dtype=torch.float64)
                add = torch.randn(*eshape, generator=g, dtype=torch.float64)
                names[(i, kind)] = (base, add)
                tensors.append((f"layers.{i}.mlp_extra.{kind}_proj.weight", add))  # extra first
                tensors.append((f"layers.{i}.mlp.{kind}_proj.weight", base))
        tensors.append(("mask_embed", torch.zeros(hidden, dtype=torch.float64)))
        merged = dict(bd.merge_ffn_extra(iter(tensors), extra))
        self.assertEqual(len(merged), 3 * 3 + 1)
        x = torch.randn(5, hidden, generator=g, dtype=torch.float64)
        for i in range(3):
            gate, up, down = (names[(i, k)] for k in ("gate", "up", "down"))
            stock = F.silu(x @ gate[0].T) * (x @ up[0].T) @ down[0].T
            added = F.silu(x @ gate[1].T) * (x @ up[1].T) @ down[1].T
            served = (F.silu(x @ merged[f"layers.{i}.mlp.gate_proj.weight"].T)
                      * (x @ merged[f"layers.{i}.mlp.up_proj.weight"].T)) @ merged[f"layers.{i}.mlp.down_proj.weight"].T
            torch.testing.assert_close(served, stock + added, rtol=0, atol=1e-12)

    def test_c1_merge_refuses_missing_piece(self):
        tensors = [("layers.0.mlp.gate_proj.weight", torch.zeros(4, 2)),
                   ("layers.0.mlp_extra.gate_proj.weight", torch.zeros(2, 2))]
        with self.assertRaisesRegex(ValueError, "needs stock and extra"):
            list(bd.merge_ffn_extra(iter(tensors), 2))

    def test_c2_zero_init_is_identity(self):
        experts = bd.DepthExperts(512, 64)
        g = torch.Generator().manual_seed(1)
        experts.gate.data = (torch.randn(7, 512, 64, generator=g) * 0.1).to(torch.bfloat16)
        experts.up.data = (torch.randn(7, 512, 64, generator=g) * 0.1).to(torch.bfloat16)
        x = torch.randn(3 * 8, 64, generator=g).to(torch.bfloat16)
        self.assertTrue(torch.equal(experts.embed(x), x))
        self.assertTrue(torch.equal(experts(x), x))

    def test_c2_equals_trainer(self):
        rank, hidden = 128, 64
        values = random_depth_experts(rank, hidden, seed=2)
        serving = bd.DepthExperts(rank, hidden)
        trainer = TrainerDepthExperts(rank, hidden).to(torch.bfloat16)
        with torch.no_grad():
            for name, value in values.items():
                getattr(serving, name).copy_(value)
                getattr(trainer, name).copy_(value)
        g = torch.Generator().manual_seed(3)
        x = torch.randn(5, 8, hidden, generator=g).to(torch.bfloat16)
        self.assertTrue(torch.equal(serving.embed(x.reshape(-1, hidden)).view_as(x), trainer.embed(x)))
        self.assertTrue(torch.equal(serving(x.reshape(-1, hidden)).view_as(x), trainer(x)))
        # A width-4 view: the same experts on depths 1..3.
        serving.block_size = 4
        x4 = x[:, :4].contiguous()
        self.assertTrue(torch.equal(serving(x4.reshape(-1, hidden)).view_as(x4), trainer(x4)))

    def test_c2_final_norm_hook(self):
        """norm(x, residual) with the hook == trainer: rms(experts(x + residual))."""
        rank, hidden = 128, 64
        values = random_depth_experts(rank, hidden, seed=4)
        experts = bd.DepthExperts(rank, hidden)
        trainer = TrainerDepthExperts(rank, hidden).to(torch.bfloat16)
        with torch.no_grad():
            for name, value in values.items():
                getattr(experts, name).copy_(value)
                getattr(trainer, name).copy_(value)
        norm = ToyNorm(hidden)
        norm.register_forward_pre_hook(experts.final_norm_pre_hook, with_kwargs=True)
        g = torch.Generator().manual_seed(5)
        x = torch.randn(2 * 8, hidden, generator=g).to(torch.bfloat16)
        r = torch.randn(2 * 8, hidden, generator=g).to(torch.bfloat16)
        out, _ = norm(x, r)
        expected = trainer_rms(trainer((x + r).view(2, 8, hidden)), norm.weight).view_as(x)
        self.assertTrue(torch.equal(out, expected))
        out_kw, _ = norm(x, residual=r)
        self.assertTrue(torch.equal(out_kw, expected))

    def test_lifted_tensors_are_identity(self):
        for arm, arch in T8_ARMS.items():
            extra = bd.identity_tensors(arch, hidden=64, n_features=4)
            for name, value in extra.items():
                if name.endswith("base_kernel"):
                    self.assertTrue(torch.equal(value[:, 0], torch.ones_like(value[:, 0])), name)
                    self.assertTrue(torch.equal(value[:, 1:], torch.zeros_like(value[:, 1:])), name)
                elif name.endswith(("kernel_projection.weight", "down_proj.weight")) or name in (
                    "tap_fc.weight", "depth_experts.row_embed", "depth_experts.down"):
                    self.assertEqual(int(value.count_nonzero()), 0, f"{arm} {name}")


# ---------------------------------------------------------------------------
# 3. Arch validation, loader, draft-dir preparer
# ---------------------------------------------------------------------------

HIDDEN, FEATURES, INTER = 32, 4, 48


def toy_stock_head(seed=0):
    g = torch.Generator().manual_seed(seed)
    t = {"mask_embed": torch.randn(HIDDEN, generator=g), "fuse_weight": torch.randn(4, FEATURES, generator=g),
         "norm.weight": torch.ones(HIDDEN)}
    for i in range(3):
        t[f"layers.{i}.self_attn.q_proj.weight"] = torch.randn(HIDDEN, HIDDEN, generator=g)
        t[f"layers.{i}.mlp.gate_proj.weight"] = torch.randn(INTER, HIDDEN, generator=g)
        t[f"layers.{i}.mlp.up_proj.weight"] = torch.randn(INTER, HIDDEN, generator=g)
        t[f"layers.{i}.mlp.down_proj.weight"] = torch.randn(HIDDEN, INTER, generator=g)
    return {k: v.to(torch.bfloat16) for k, v in t.items()}


def save(path, tensors, arch=None):
    from safetensors.torch import save_file

    save_file(tensors, str(path), metadata=None if arch is None else {bd.BLOCK_ARCH_KEY: json.dumps(arch)})


class LoaderTests(unittest.TestCase):
    def test_t8_arms_validate(self):
        for arm, arch in T8_ARMS.items():
            parsed = bd.parse_block_arch(json.dumps(arch), STOCK_DRAFT)
            self.assertEqual(bd.arm_name(parsed), arm)

    def test_refusals(self):
        bad = {
            "K4": trainer_arch(selector_rank=256),
            "unknown key": dict(trainer_arch(), pool=4),
            "ffn width": trainer_arch(ffn_extra=1000),
            "expert rank": trainer_arch(depth_experts=300),
            "conv + experts": trainer_arch(block_conv=2, depth_experts=512),
            "window": trainer_arch(window=512),
            "fewer layers": trainer_arch(n_layers=2),
            "conv group": trainer_arch(block_conv=2, conv_group=8),
        }
        for label, arch in bad.items():
            with self.assertRaises(ValueError, msg=label):
                bd.parse_block_arch(json.dumps(arch), STOCK_DRAFT)
        with self.assertRaises(ValueError):
            bd.parse_block_arch("{not json", STOCK_DRAFT)
        with self.assertRaisesRegex(ValueError, "wider"):
            bd.parse_block_arch(json.dumps(T8_ARMS["c2"]), dict(STOCK_DRAFT, block_size=9))

    def test_added_parameters_match_the_trainer_receipts(self):
        # BLOCK-DRAFTER.md: K1 8.02M, K2 +21.2M (cumulative 29.2M)
        self.assertEqual(bd.added_parameters(T8_ARMS["k1"]), 8017920)
        self.assertEqual(bd.added_parameters(T8_ARMS["k2"]) - bd.added_parameters(T8_ARMS["k1"]), 2304 * 9216)
        self.assertEqual(bd.added_parameters(T8_ARMS["c1"]), 3 * 3 * 2304 * 4608)
        self.assertEqual(bd.added_parameters(T8_ARMS["c2"]), 7 * 2304 + 2304 + 3 * 7 * 512 * 2304)

    def test_check_head_and_prepare(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            stock_dir = tmp / "stock"
            stock_dir.mkdir()
            stock = toy_stock_head()
            save(stock_dir / "mtp.safetensors", stock)
            (stock_dir / "config.json").write_text("{}")
            for arm, arch in T8_ARMS.items():
                head = lift_toy(stock, arch)
                path = tmp / f"{arm}.safetensors"
                save(path, head, arch)
                receipt = bd.check_head(path, stock_path=stock_dir / "mtp.safetensors")
                self.assertEqual(receipt["arm"], arm)
                self.assertEqual(json.loads(receipt["env"][bd.BLOCK_ARCH_ENV]), arch)
                out = tmp / f"draft-{arm}"
                control = tmp / "control.safetensors"
                save(control, toy_stock_head(seed=7))
                receipt = bd.prepare_draft_dir(path, stock_dir, out, lift_from=control)
                self.assertTrue((out / "config.json").exists())
                self.assertEqual(receipt["sha256"], bd.sha256_file(out / "mtp.safetensors"))
                env = (out / "block-arch.env").read_text()
                self.assertIn(bd.BLOCK_ARCH_ENV, env)
                self.assertEqual(bd.head_block_arch(out / "control-lifted.safetensors"), arch)
                with self.assertRaises(ValueError):  # not empty
                    bd.prepare_draft_dir(path, stock_dir, out)
            # a missing arm tensor, a reshaped one, and arm tensors without metadata
            head = lift_toy(stock, T8_ARMS["c1"])
            broken = dict(head)
            del broken["layers.1.mlp_extra.up_proj.weight"]
            save(tmp / "missing.safetensors", broken, T8_ARMS["c1"])
            with self.assertRaisesRegex(ValueError, "needs"):
                bd.check_head(tmp / "missing.safetensors")
            broken = dict(head)
            broken["layers.0.mlp_extra.down_proj.weight"] = torch.zeros(HIDDEN, 7, dtype=torch.bfloat16)
            save(tmp / "shape.safetensors", broken, T8_ARMS["c1"])
            with self.assertRaisesRegex(ValueError, "expected"):
                bd.check_head(tmp / "shape.safetensors")
            save(tmp / "nometa.safetensors", head)
            with self.assertRaisesRegex(ValueError, "stock tensors differ"):
                bd.check_head(tmp / "nometa.safetensors", stock_path=stock_dir / "mtp.safetensors")
            save(tmp / "plain.safetensors", toy_stock_head(seed=3))  # a stock-format head
            with self.assertRaisesRegex(ValueError, "no block_arch"):
                bd.prepare_draft_dir(tmp / "plain.safetensors", stock_dir, tmp / "x")


def lift_toy(stock, arch):
    return bd.lift_head(stock, arch)


if __name__ == "__main__":
    unittest.main(verbosity=2)
