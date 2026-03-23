#!/bin/bash
# Generate diverse ASCII training data v2
# 200 episodes x 300 steps x varied audio = 60K frames

set -euo pipefail

OUTPUT="data/ascii_training_v2.jsonl"
rm -f "$OUTPUT"

for profile in high low ramp pulse random sweep; do
  echo "Generating $profile episodes..."
  python3 -m world_model.envs.ascii_corridor generate \
    --episodes 33 --steps 300 \
    --audio-profile $profile \
    --output "$OUTPUT" --append
done

echo "Converting to NPZ..."
python3 -c "
import json, numpy as np, torch
from world_model.ascii_model.model import frame_to_indices
frames, audios, episodes = [], [], []
with open('$OUTPUT') as f:
    for i, line in enumerate(f):
        r = json.loads(line)
        idx = frame_to_indices(r['ascii_frame'])
        if isinstance(idx, torch.Tensor): idx = idx.numpy()
        frames.append(idx)
        audios.append(r['audio_context'])
        episodes.append(r.get('episode', 0))
        if (i+1) % 10000 == 0: print(f'  {i+1}...')
np.savez_compressed('data/ascii_training_v2.npz',
    frames=np.array(frames, dtype=np.int16),
    audios=np.array(audios, dtype=np.float32),
    episodes=np.array(episodes, dtype=np.int32))
print(f'Done: {len(frames)} frames')
"
