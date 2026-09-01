"""Construct the checkpoint-compatible SgAMC implementation."""

from .legacy import SgAMC as LegacySgAMC
from .original import SgAMC as OriginalSgAMC


_LAYOUT = "original"


def get_sgamc_layout():
    return _LAYOUT


def set_sgamc_layout(layout):
    global _LAYOUT
    if layout not in {"original", "legacy"}:
        raise ValueError("SgAMC layout must be 'original' or 'legacy'.")
    _LAYOUT = layout


def infer_sgamc_layout(state_dict):
    keys = state_dict.keys()
    if any(".sgAMC.branch_queries" in key or ".sg_amc.branch_queries" in key for key in keys):
        return "legacy"
    return "original"


def configure_from_checkpoint(args, checkpoint, restore_architecture=True):
    """Restore architecture switches saved with a DLSA checkpoint."""
    state_dict = checkpoint.get("model", checkpoint)
    layout = infer_sgamc_layout(state_dict)
    set_sgamc_layout(layout)
    args.sgamc_layout = layout

    saved = checkpoint.get("args")
    if restore_architecture and saved is not None:
        for name in (
            "use_text_decoder",
            "text_decoder_variant",
            "text_decoder_stages",
            "td_res_scale_init",
            "td_gate_bias_init",
            "td_alpha_bias_init",
            "td_alpha_weight_std",
            "td_uncertainty_gate",
            "alti_ablation",
            "use_decoder_sgamc",
        ):
            if hasattr(saved, name):
                setattr(args, name, getattr(saved, name))
    return layout


def SgAMC(*args, **kwargs):
    """Return SgAMC without adding a wrapper level to the state dict."""
    layout = kwargs.pop("layout", None) or _LAYOUT
    if layout == "original":
        kwargs.pop("kernel_ablation", None)
        kwargs.pop("rf_variant", None)
        return OriginalSgAMC(*args, **kwargs)
    if layout == "legacy":
        kwargs.pop("rf_variant", None)
        return LegacySgAMC(*args, **kwargs)
    raise ValueError("Unsupported SgAMC layout: {}".format(layout))
