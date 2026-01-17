import torch
import torch.nn as nn
from ultralytics.models.yolo.segment import SegmentationTrainer
from contrastive_loss import SupContrastLoss
import types

# GLOBAL BUFFER for features
CAPTURED_FEATURES = {}

def global_feature_hook(module, input, output):
    """Top-level function for hooks to remain picklable."""
    CAPTURED_FEATURES[id(module)] = output[0] if isinstance(output, tuple) else output

def log_debug(msg):
    with open("debug_log.txt", "a") as f:
        f.write(msg + "\n")
    print(msg)

def compute_contrastive_loss(batch, model, contrast_fn):
    """Computes contrastive loss using captured features and model's projection heads."""
    if not CAPTURED_FEATURES or 'masks' not in batch:
        return torch.tensor(0.0, device=next(model.parameters()).device)

    try:
        masks = batch['masks']
        device = masks.device
        total_c_loss = torch.tensor(0.0, device=device)
        heads = model.contrastive_heads
        
        for feat_id, feat in CAPTURED_FEATURES.items():
            feat_key = str(feat_id)
            if feat_key not in heads:
                in_ch = feat.shape[1]
                heads[feat_key] = nn.Sequential(
                    nn.Conv2d(in_ch, in_ch, kernel_size=1),
                    nn.ReLU(),
                    nn.Conv2d(in_ch, 128, kernel_size=1)
                ).to(device)
                log_debug(f"DEBUG: Initialized head for {feat_key}")

            proj = heads[feat_key](feat)
            h, w = proj.shape[2:]
            m = torch.nn.functional.interpolate(masks.unsqueeze(1).float(), size=(h, w), mode='nearest').squeeze(1)
            
            # Boundary Aware Sampling
            eroded = 1.0 - torch.nn.functional.max_pool2d(1.0 - m.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
            dilated = torch.nn.functional.max_pool2d(m.unsqueeze(1), kernel_size=5, stride=1, padding=2).squeeze(1)
            boundary = m - eroded
            hard_neg = dilated - m

            sampled_emb = []
            sampled_lbl = []
            n_target = 64
            n_half = 32
            
            for i in range(proj.shape[0]):
                p_idx = torch.where(m[i] > 0.5)
                b_idx = torch.where(boundary[i] > 0.5)
                if len(p_idx[0]) > 0:
                    c = min(len(b_idx[0]), n_half)
                    idx1 = torch.randperm(len(b_idx[0]))[:c]
                    idx2 = torch.randperm(len(p_idx[0]))[:n_target - c]
                    sampled_emb.append(proj[i, :, b_idx[0][idx1], b_idx[1][idx1]].T)
                    sampled_emb.append(proj[i, :, p_idx[0][idx2], p_idx[1][idx2]].T)
                    sampled_lbl.append(torch.ones(len(idx1) + len(idx2), device=device))
                
                n_idx = torch.where(m[i] <= 0.5)
                hn_idx = torch.where(hard_neg[i] > 0.5)
                if len(n_idx[0]) > 0:
                    c = min(len(hn_idx[0]), n_half)
                    idx1 = torch.randperm(len(hn_idx[0]))[:c]
                    idx2 = torch.randperm(len(n_idx[0]))[:n_target - c]
                    sampled_emb.append(proj[i, :, hn_idx[0][idx1], hn_idx[1][idx1]].T)
                    sampled_emb.append(proj[i, :, n_idx[0][idx2], n_idx[1][idx2]].T)
                    sampled_lbl.append(torch.zeros(len(idx1) + len(idx2), device=device))

            if sampled_emb:
                emb = torch.cat(sampled_emb, dim=0)
                lbl = torch.cat(sampled_lbl, dim=0)
                emb = torch.nn.functional.normalize(emb, dim=1, eps=1e-6).unsqueeze(1)
                c_loss = contrast_fn(emb, labels=lbl)
                if not torch.isnan(c_loss) and not torch.isinf(c_loss):
                    total_c_loss += c_loss
                else:
                    log_debug(f"DEBUG: NaN/Inf detected in layer loss")

        return total_c_loss / max(len(heads), 1)
    except Exception as e:
        log_debug(f"ERROR in compute_contrastive_loss: {e}")
        return torch.tensor(0.0, device=next(model.parameters()).device)

def patched_model_forward(self, x, *args, **kwargs):
    """
    Patched forward method for the model to inject contrastive loss.
    `self` is the model instance.
    """
    if isinstance(x, dict) and self.training:
        # 1. Clear features before forward
        CAPTURED_FEATURES.clear()
        
        # 2. Original forward (handles batch and returns (loss, loss_items))
        loss, loss_items = self._orig_forward(x, *args, **kwargs)
        
        # 3. Compute Contrastive Loss
        c_loss_val = compute_contrastive_loss(x, self, self.contrast_fn)
        
        # Apply weighting
        w = getattr(self, 'contrast_weight', 0.05)
        c_loss = c_loss_val * w
        
        # Defensive check
        if torch.isnan(c_loss) or torch.isinf(c_loss) or torch.isnan(loss).any():
            if torch.isnan(loss).any():
                 if torch.rand(1).item() < 0.05:
                    log_debug("DEBUG: Standard loss is NaN, skipping contrastive addition")
            else:
                log_debug("DEBUG: Combined Contrastive Loss is NaN/Inf, skipping addition")
            return loss, loss_items

        c_loss = torch.clamp(c_loss, max=1.0)
        
        if torch.rand(1).item() < 0.05:
            log_debug(f"BATCH: Contrastive Loss: {c_loss.item():.4f} (Original loss: {loss.mean().item():.4f})")
            
        return loss + c_loss, loss_items
    else:
        return self._orig_forward(x, *args, **kwargs)

class ContrastiveTrainer(SegmentationTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        log_debug("DEBUG: get_model() called")
        model = super().get_model(cfg, weights, verbose)
        
        # Initialize Contrastive Components on the Model
        if not hasattr(model, 'contrastive_heads'):
            model.contrastive_heads = nn.ModuleDict()
            model.contrast_fn = SupContrastLoss(temperature=0.2)
            model.contrast_weight = 0.05
            
            # Register Hooks
            for idx in [4, 6, 9]:
                if idx < len(model.model):
                    layer = model.model[idx]
                    layer.register_forward_hook(global_feature_hook)
            
            # Monkeypatch Forward
            model._orig_forward = model.forward
            model.forward = types.MethodType(patched_model_forward, model)
            log_debug("DEBUG: Model forward patched successfully")
            
        return model
