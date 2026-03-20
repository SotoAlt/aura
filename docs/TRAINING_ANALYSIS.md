# Training Analysis: Why the Model Can't Learn Forward Motion

## How Training Currently Works

### Data Pipeline
1. Video frames extracted at 10fps → 2795 frames at 128x128
2. Audio context vectors extracted per frame (16 floats)
3. Optical flow computed between consecutive frames → 3-float action vectors
4. Sliding window (101 frames, stride 20) → 135 base episodes
5. 5 augmentation passes per window → **675 total episodes**
6. Each episode: sequential frames from the video in order, with their real temporal relationships preserved

### Training Loop
1. `NPZDataset` loads episodes and samples **random contiguous subsequences** (seq_length=50 frames)
2. The RSSM `observe` function processes each subsequence:
   - For each timestep t: encode frame[t] → embed, then update RSSM state using (action[t], embed[t], context[t])
   - The **posterior** uses the real encoded frame (teacher forcing)
   - The **prior** predicts from just (previous_state + action + context)
3. Loss = MSE(decoded_frames, real_frames) + KL(posterior, prior)

### What the Model Learns During Training
- **Encoder**: compress 128x128 frame → 512-dim embedding ✅
- **Decoder**: reconstruct frame from RSSM features ✅ (but blurry due to MSE)
- **RSSM posterior**: given real frame + action + audio → good latent state ✅
- **RSSM prior**: predict next state from previous state + action + audio → ❓

### The Key Problem: Teacher Forcing vs Free Imagination

During **training**, the RSSM always sees the real next frame (teacher forcing via `observe`). The prior learns to be close to the posterior (minimized by KL loss), but it never has to actually generate a coherent sequence on its own.

During **eval/imagination**, we call `imagine` which uses only the prior — no real frames. The prior has to chain: state₀ → state₁ → state₂ → ... Each step compounds errors. After a few steps, the state collapses to a fixed point (the "mean" of all training data) because:

1. **The prior was never trained to be self-consistent over long horizons** — it only learned to match the posterior at each single step
2. **MSE loss on the decoder averages everything** — when the prior is uncertain, MSE produces the average of all plausible frames = blur
3. **The KL loss is dominated by free_nats=1.0** — we're barely using the stochastic latent, so the model doesn't learn diverse predictions

### Why Flow Actions Don't Fix It

Flow actions give the RSSM information about how much motion happened between frames. But during imagination, the RSSM still can't generate coherent multi-step trajectories because:
- The prior was trained with teacher forcing (always corrected by real observations)
- Without observations, the prior quickly converges to the data mean
- Action conditioning is a small signal compared to the observation correction

This is a **fundamental limitation of the RSSM architecture with MSE loss**, not a data quality issue. DreamerV3 works in RL because the agent's policy compensates — here we have no agent, just free dreaming.

## What The Video Data Actually Contains

The MATSYA video has:
- ✅ Forward camera motion through organic 3D environments
- ✅ Rich color palette (pinks, purples, greens, blues)
- ✅ Temporal coherence — consecutive frames are highly correlated
- ✅ The data is stored in correct temporal order in episodes

The model IS seeing sequential frames during training. The problem is not the data — it's that the architecture can't reproduce temporal coherence during free imagination.

## What Would Fix It

### Option A: Diffusion Decoder (DIAMOND approach)
Instead of MSE-decoded frames, use a conditional diffusion model that generates the next frame given:
- Previous frame(s) as conditioning
- RSSM latent state (audio + dynamics)

This naturally produces sharp frames and temporal coherence because diffusion models commit to specific details rather than averaging.

### Option B: Autoregressive Frame Prediction
Condition the decoder not just on the RSSM state, but also on the **previous decoded frame**. This creates a feedback loop that maintains visual coherence.

### Option C: Observation-Guided Imagination
During eval, feed back the decoded frame as a pseudo-observation to keep the RSSM grounded. This is a hack but might work for the demo.

## Recommendation

**Option C is the quickest test** — feed decoded frames back into the encoder during imagination to see if self-consistency improves. If it helps, we know the dynamics are there but the open-loop imagination is the bottleneck. Then Option A (DIAMOND) is the proper fix.
