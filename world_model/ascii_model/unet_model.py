"""U-Net ASCII frame predictor — multi-resolution glyph prediction.

Adapts DIAMOND U-Net for discrete glyph classification (not diffusion).
  - Skip connections, 3 levels (40x80 → 20x40 → 10x20)
  - Self-attention at 10x20, AdaGroupNorm audio conditioning
  - Cross-entropy on vocab_size logits, ~500K-1M params
"""
from __future__ import annotations
import argparse, os
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from world_model.ascii_model.model import VOCAB_SIZE

# --- Building blocks (from diamond/unet.py, no sigma embedding) -----------

class AdaGroupNorm(nn.Module):
    def __init__(self, ch: int, groups: int, cond_dim: int):
        super().__init__()
        g = min(groups, ch)
        while ch % g != 0: g -= 1
        self.norm = nn.GroupNorm(g, ch, affine=False)
        self.proj = nn.Linear(cond_dim, ch * 2)
    def forward(self, x, cond):
        x = self.norm(x)
        s = self.proj(cond)[:, :, None, None]
        scale, shift = s.chunk(2, dim=1)
        return x * (1 + scale) + shift

class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int):
        super().__init__()
        self.norm1 = AdaGroupNorm(in_ch, 8, cond_dim)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = AdaGroupNorm(out_ch, 8, cond_dim)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x, cond):
        h = self.conv1(F.silu(self.norm1(x, cond)))
        h = self.conv2(F.silu(self.norm2(h, cond)))
        return h + self.skip(x)

class SelfAttention(nn.Module):
    def __init__(self, ch: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(min(8, ch), ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x)).reshape(B, 3, self.heads, C // self.heads, H * W)
        q, k, v = [t.permute(0, 1, 3, 2) for t in (qkv[:, 0], qkv[:, 1], qkv[:, 2])]
        a = F.scaled_dot_product_attention(q, k, v).permute(0, 1, 3, 2).reshape(B, C, H, W)
        return x + self.proj(a)

class Downsample(nn.Module):
    def __init__(self, ch): super().__init__(); self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
    def forward(self, x): return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, ch): super().__init__(); self.conv = nn.Conv2d(ch, ch, 3, padding=1)
    def forward(self, x): return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))

# --- U-Net for ASCII frame prediction -------------------------------------

CHANNELS = [64, 128, 128]  # 3 levels: 40x80 → 20x40 → 10x20
NUM_RES = 2
COND_DIM = 128
CTX_FRAMES = 4
EMBED_DIM = 16
ATTN_LVL = 2  # self-attention at level 2 (10x20 resolution)

class AsciiUNet(nn.Module):
    """U-Net glyph predictor: 4 prev frames + audio → next frame logits.

    Input:  prev_frames (B,4,40,80) long, audio_context (B,16) float
    Output: (B, VOCAB_SIZE, 40, 80) logits
    """
    def __init__(self, embed_dim=EMBED_DIM, cond_dim=COND_DIM):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, embed_dim)
        self.cond_mlp = nn.Sequential(
            nn.Linear(16, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.input_conv = nn.Conv2d(CTX_FRAMES * embed_dim, CHANNELS[0], 3, padding=1)

        # Encoder
        self.enc_blocks, self.enc_attns, self.downs = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        ch = CHANNELS[0]
        self._skip_ch = [ch]
        for lvl, ch_out in enumerate(CHANNELS):
            blks, atts = nn.ModuleList(), nn.ModuleList()
            for _ in range(NUM_RES):
                blks.append(ResBlock(ch, ch_out, cond_dim))
                atts.append(SelfAttention(ch_out) if lvl == ATTN_LVL else nn.Identity())
                ch = ch_out; self._skip_ch.append(ch)
            self.enc_blocks.append(blks); self.enc_attns.append(atts)
            if lvl < len(CHANNELS) - 1:
                self.downs.append(Downsample(ch)); self._skip_ch.append(ch)
            else:
                self.downs.append(None)

        # Middle
        self.mid1 = ResBlock(ch, ch, cond_dim)
        self.mid_attn = SelfAttention(ch)
        self.mid2 = ResBlock(ch, ch, cond_dim)

        # Decoder
        self.dec_blocks, self.dec_attns, self.ups = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for lvl in reversed(range(len(CHANNELS))):
            ch_out = CHANNELS[lvl]
            blks, atts = nn.ModuleList(), nn.ModuleList()
            for _ in range(NUM_RES + 1):
                skip_ch = self._skip_ch.pop()
                blks.append(ResBlock(ch + skip_ch, ch_out, cond_dim))
                atts.append(SelfAttention(ch_out) if lvl == ATTN_LVL else nn.Identity())
                ch = ch_out
            self.dec_blocks.append(blks); self.dec_attns.append(atts)
            self.ups.append(Upsample(ch) if lvl > 0 else None)

        # Output
        g = min(8, ch)
        while ch % g != 0: g -= 1
        self.out_norm = nn.GroupNorm(g, ch)
        self.out_conv = nn.Conv2d(ch, VOCAB_SIZE, 1)

    def forward(self, prev_frames, audio_context):
        B, N, H, W = prev_frames.shape
        cond = self.cond_mlp(audio_context)
        x = self.embed(prev_frames).permute(0, 1, 4, 2, 3).reshape(B, -1, H, W)
        x = self.input_conv(x)

        # Encoder
        skips = [x]
        for lvl in range(len(CHANNELS)):
            for blk, att in zip(self.enc_blocks[lvl], self.enc_attns[lvl]):
                x = att(blk(x, cond)); skips.append(x)
            if self.downs[lvl] is not None:
                x = self.downs[lvl](x); skips.append(x)

        x = self.mid1(x, cond); x = self.mid_attn(x); x = self.mid2(x, cond)

        # Decoder
        for lvl, (blks, atts) in enumerate(zip(self.dec_blocks, self.dec_attns)):
            for blk, att in zip(blks, atts):
                x = att(blk(torch.cat([x, skips.pop()], dim=1), cond))
            if self.ups[lvl] is not None:
                x = self.ups[lvl](x)

        return self.out_conv(F.silu(self.out_norm(x)))

# --- Training script -------------------------------------------------------

if __name__ == "__main__":
    pa = argparse.ArgumentParser(description="Train AsciiUNet")
    pa.add_argument("--data", default="data/ascii_training.npz")
    pa.add_argument("--epochs", type=int, default=20)
    pa.add_argument("--batch-size", type=int, default=32)
    pa.add_argument("--checkpoint", default="checkpoints/ascii_unet.pt")
    pa.add_argument("--device", default="cpu")
    args = pa.parse_args()

    d = np.load(args.data)
    frames = torch.from_numpy(d["frames"]).long()   # (N, 40, 80)
    audio = torch.from_numpy(d["audio"]).float()     # (N, 16)

    # Build sequences: 4 context → 1 target
    ctx = torch.stack([frames[i-4:i] for i in range(4, len(frames))])
    tgt = frames[4:]
    aud = audio[4:]
    print(f"Dataset: {len(ctx)} samples from {len(frames)} frames")

    device = torch.device(args.device)
    model = AsciiUNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"AsciiUNet: {n_params:,} parameters")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    bs = args.batch_size

    step = 0
    for epoch in range(args.epochs):
        perm = torch.randperm(len(ctx))
        e_loss, e_correct, e_total = 0.0, 0, 0
        for i in range(0, len(ctx), bs):
            idx = perm[i:i+bs]
            xb, ab, yb = ctx[idx].to(device), aud[idx].to(device), tgt[idx].to(device)
            logits = model(xb, ab)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()

            preds = logits.argmax(dim=1)
            correct = (preds == yb).sum().item()
            total = yb.numel()
            e_loss += loss.item() * len(idx)
            e_correct += correct; e_total += total
            step += 1
            if step % 50 == 0:
                print(f"  step {step:>5d}  loss={loss.item():.4f}  acc={correct/total*100:.1f}%")

        print(f"Epoch {epoch+1}/{args.epochs}  loss={e_loss/len(ctx):.4f}  acc={e_correct/e_total*100:.1f}%")

    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "vocab_size": VOCAB_SIZE}, args.checkpoint)
    print(f"Saved checkpoint -> {args.checkpoint}")
