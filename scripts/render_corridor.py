"""Render MuJoCo corridor frames and save as JSON for HTML viewer."""
import os
os.environ["MUJOCO_GL"] = "egl"

import mujoco
import numpy as np
from PIL import Image
import base64, io, json

CORRIDOR_XML = """
<mujoco model="rail_corridor">
  <visual>
    <map znear="0.01" zfar="100"/>
    <rgba haze="0.35 0.38 0.48 1"/>
    <global offwidth="224" offheight="224"/>
    <quality shadowsize="0"/>
    <headlight ambient="0.4 0.4 0.5" diffuse="0.5 0.5 0.6" specular="0 0 0"/>
  </visual>
  <option timestep="0.02" gravity="0 0 0"/>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.35 0.38 0.5" rgb2="0.2 0.22 0.32" width="32" height="512"/>
    <texture name="ftex" type="2d" builtin="checker" width="128" height="128" rgb1="0.3 0.33 0.42" rgb2="0.45 0.48 0.55"/>
    <texture name="wtex" type="2d" builtin="checker" width="128" height="128" rgb1="0.35 0.38 0.48" rgb2="0.45 0.48 0.56"/>
    <texture name="ctex" type="2d" builtin="checker" width="128" height="128" rgb1="0.32 0.35 0.45" rgb2="0.42 0.45 0.53"/>
    <material name="fmat" texture="ftex" texrepeat="20 4" specular="0" reflectance="0"/>
    <material name="wmat" texture="wtex" texrepeat="20 4" specular="0" reflectance="0"/>
    <material name="cmat" texture="ctex" texrepeat="20 4" specular="0" reflectance="0"/>
    <material name="oR" rgba="0.85 0.45 0.25 1" specular="0"/>
    <material name="oB" rgba="0.3 0.45 0.8 1" specular="0"/>
    <material name="oG" rgba="0.3 0.65 0.35 1" specular="0"/>
    <material name="neon" rgba="0.3 0.85 0.95 1" emission="0.15"/>
    <material name="neon_r" rgba="0.95 0.3 0.4 1" emission="0.1"/>
    <material name="pmat" rgba="0.2 0.8 0.3 1" specular="0.2" shininess="0.1"/>
  </asset>
  <worldbody>
    <light pos="50 0 2" dir="1 0 0" directional="true" diffuse="1.0 1.0 1.1" specular="0.4 0.4 0.5"/>
    <light pos="50 0 2" dir="-1 0 0" directional="true" diffuse="0.6 0.6 0.7"/>
    <light pos="50 0 2" dir="0 0 -1" directional="true" diffuse="0.5 0.5 0.6"/>
    <light pos="50 0 2" dir="0 0 1" directional="true" diffuse="0.5 0.5 0.6"/>
    <geom name="floor" type="box" pos="50 0 0" size="55 2.5 0.05" material="fmat"/>
    <geom name="ceil" type="box" pos="50 0 4" size="55 2.5 0.05" material="cmat"/>
    <geom name="wl" type="box" pos="50 2.5 2" size="55 0.05 2" material="wmat"/>
    <geom name="wr" type="box" pos="50 -2.5 2" size="55 0.05 2" material="wmat"/>
    <geom type="box" pos="50 2.44 1.0" size="55 0.02 0.02" material="neon"/>
    <geom type="box" pos="50 -2.44 1.0" size="55 0.02 0.02" material="neon"/>
    <geom type="box" pos="50 2.44 3.0" size="55 0.02 0.02" material="neon_r"/>
    <geom type="box" pos="50 -2.44 3.0" size="55 0.02 0.02" material="neon_r"/>
    <body name="cam" pos="5 0 1.8">
      <joint name="rail" type="slide" axis="1 0 0" range="0 90"/>
      <geom type="sphere" size="0.001" mass="1" rgba="0 0 0 0"/>
      <camera name="ego" mode="fixed" pos="0 0 0" xyaxes="0 -1 0 0 0 1" fovy="75"/>
      <!-- Player is child of camera — always in front, not affected by rail movement -->
      <body name="player" pos="2.5 0 -0.4">
        <joint name="px" type="slide" axis="0 1 0" range="-0.8 0.8" damping="5"/>
        <joint name="pz" type="slide" axis="0 0 1" range="-0.4 0.4" damping="5"/>
        <geom name="player_body" type="sphere" size="0.25" material="pmat" mass="0.5"/>
      </body>
    </body>
    <geom type="box" pos="15 1.0 0.8" size="0.3 0.3 0.8" material="oR"/>
    <geom type="cylinder" pos="22 -0.8 1.2" size="0.4 1.2" material="oB"/>
    <geom type="box" pos="30 0.5 0.5" size="0.5 0.5 0.5" material="oG"/>
    <geom type="sphere" pos="38 -1.2 1.5" size="0.5" material="oR"/>
    <geom type="box" pos="45 0.0 0.8" size="0.2 1.8 0.8" material="oB"/>
    <geom type="cylinder" pos="52 1.5 0.6" size="0.3 0.6" material="oG"/>
    <geom type="box" pos="58 -0.5 1.8" size="0.4 0.4 0.4" material="oR"/>
    <geom type="sphere" pos="65 0.8 1.0" size="0.35" material="oB"/>
    <geom type="box" pos="72 -1.0 0.5" size="0.6 0.3 0.5" material="oG"/>
    <geom type="cylinder" pos="80 0.3 1.5" size="0.25 1.5" material="oR"/>
    <geom type="sphere" pos="90 1.3 2.0" size="0.4" material="neon"/>
  </worldbody>
  <actuator>
    <velocity name="rf" joint="rail" kv="10"/>
    <motor name="dx" joint="px" gear="5" ctrllimited="true" ctrlrange="-1 1"/>
    <motor name="dz" joint="pz" gear="5" ctrllimited="true" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""

print("Building MuJoCo corridor...")
model = mujoco.MjModel.from_xml_string(CORRIDOR_XML)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, 224, 224)

frames_b64 = []
for i in range(80):
    data.ctrl[0] = 3.0   # rail forward (camera + player move together)
    data.ctrl[1] = 0.7 * np.sin(i * 0.12)   # player dodge x
    data.ctrl[2] = 0.4 * np.sin(i * 0.08)   # player dodge z
    mujoco.mj_step(model, data)
    renderer.update_scene(data, camera="ego")
    frame = renderer.render()
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="JPEG", quality=85)
    frames_b64.append(base64.b64encode(buf.getvalue()).decode())
    if i % 20 == 0:
        print(f"  Frame {i}/80")

out_path = "/workspace/corridor_frames.json"
with open(out_path, "w") as f:
    json.dump(frames_b64, f)

size_mb = os.path.getsize(out_path) / 1e6
print(f"Saved {len(frames_b64)} frames to {out_path} ({size_mb:.1f} MB)")
