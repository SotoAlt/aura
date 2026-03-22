#!/bin/bash
# Download MATSYA video via pytubefix + build episodes
# Usage: bash scripts/generate_data.sh
set -e

VIDEO_URL="${VIDEO_URL:-https://www.youtube.com/watch?v=qEJfEyHNZl0}"
OUTPUT="${OUTPUT:-/content/video_data}"
SIZE="${SIZE:-64}"

echo "=== AURA Data Generation ==="
echo "Video: ${VIDEO_URL}"
echo "Output: ${OUTPUT}"
echo "Size: ${SIZE}x${SIZE}"

pip install -q pytubefix librosa

python3 -c "
import subprocess, sys
from pathlib import Path
from pytubefix import YouTube

out = Path('${OUTPUT}/_raw')
out.mkdir(parents=True, exist_ok=True)
vpath = out / 'video.mp4'
apath = out / 'audio.wav'
fdir = out / 'frames'
fdir.mkdir(exist_ok=True)

if not vpath.exists():
    print('Downloading video...')
    yt = YouTube('${VIDEO_URL}')
    stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
    stream.download(output_path=str(out), filename='video.mp4')
    print(f'Downloaded: {vpath} ({vpath.stat().st_size / 1e6:.1f} MB)')
else:
    print(f'Video exists: {vpath}')

if not apath.exists():
    print('Extracting audio...')
    subprocess.run(['ffmpeg','-y','-i',str(vpath),'-ar','22050','-ac','1',str(apath)], check=True)

if not list(fdir.glob('frame_*.png')):
    print('Extracting frames...')
    subprocess.run(['ffmpeg','-y','-i',str(vpath),'-vf','crop=min(iw\,ih):min(iw\,ih),scale=${SIZE}:${SIZE}:flags=lanczos','-r','10',str(fdir/'frame_%05d.png')], check=True)

n = len(list(fdir.glob('frame_*.png')))
print(f'Frames extracted: {n}')
"

echo "=== Building episodes ==="
cd /content/aura
python3 -c "
from pathlib import Path
from world_model.data.video import load_frames, build_episodes
build_episodes(
    Path('${OUTPUT}/_raw/frames'),
    Path('${OUTPUT}/_raw/audio.wav'),
    '${OUTPUT}',
    steps_per_episode=100,
    augmentation_passes=5,
    fps=10.0,
)
"

EPISODE_COUNT=$(ls -1 "${OUTPUT}"/episode_*.npz 2>/dev/null | wc -l)
echo "=== Done: ${EPISODE_COUNT} episodes in ${OUTPUT} ==="
