import torch
import torch.nn as nn

from backward_map import dewarp

DEFAULT_WEIGHTS = {
    "Grid": 1.0,       # L1 on the backward grid (main term)
    "Recon": 0.2,      # L1 between dewarped clean page and flat page
    "Edge": 0.2,       # L1 between dewarped warped-edges and flat edges (text-line alignment)
    "Curvature": 1.0,  # L1 between second-order grid differences of prediction and GT
}


def _second_diff(g: torch.Tensor):
    dxx = g[..., :, 2:] - 2 * g[..., :, 1:-1] + g[..., :, :-2]
    dyy = g[..., 2:, :] - 2 * g[..., 1:-1, :] + g[..., :-2, :]
    return dxx, dyy


def _gated_l1(pred: torch.Tensor, target: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Per-sample mean L1, averaged over samples with gate == 1."""
    per_sample = (pred - target).abs().flatten(1).mean(1)
    return (per_sample * gate).sum() / gate.sum().clamp(min=1.0)


class BackwardMapLoss(nn.Module):
    """
    Curvature matches the GT grid's second derivatives instead of penalizing curvature itself,
    so it suppresses jitter without pulling real paper curvature towards a flat sheet.
    """
    def __init__(self, weights: dict = None):
        super().__init__()
        self.weights = dict(DEFAULT_WEIGHTS if weights is None else weights)

    def forward(self, pred_bm: torch.Tensor, batch: dict, recon_scale: float = 1.0):
        pred_bm = pred_bm.float()
        gt_bm = batch["bm"].float()

        terms = {"Grid": (pred_bm - gt_bm).abs().mean()}

        pxx, pyy = _second_diff(pred_bm)
        gxx, gyy = _second_diff(gt_bm)
        terms["Curvature"] = (pxx - gxx).abs().mean() + (pyy - gyy).abs().mean()

        size = tuple(batch["flat"].shape[-2:])
        if self.weights.get("Recon", 0) > 0:
            recon = dewarp(batch["warped"].float(), pred_bm, size)
            terms["Recon"] = _gated_l1(recon, batch["flat"].float(), batch["has_flat"])
        if self.weights.get("Edge", 0) > 0:
            recon_edges = dewarp(batch["warped_edges"].float(), pred_bm, size)
            terms["Edge"] = _gated_l1(recon_edges, batch["flat_edges"].float(), batch["has_edges"])

        scale = {"Recon": recon_scale, "Edge": recon_scale}
        total = sum(self.weights[k] * scale.get(k, 1.0) * v for k, v in terms.items())
        return total, {k: v.item() for k, v in terms.items()}
