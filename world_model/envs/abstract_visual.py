"""Sci-fi corridor generator for AURA — 128x128 rectangular tunnel
with geometric shapes (circles, hexagons, diamonds, triangles) as
architectural features. Neon-lit, audio-reactive.
"""

import argparse
import numpy as np
from pathlib import Path


def unpack_context(c: np.ndarray) -> dict[str, float]:
    return {
        'bass': float((c[0] + c[1]) / 2), 'mid': float((c[2] + c[3]) / 2),
        'high': float((c[4] + c[5]) / 2), 'onset': float((c[6] + c[7]) / 2),
        'bpm': float((c[8] + c[9]) / 2), 'temperature': float((c[10] + c[11]) / 2),
        'rms': float((c[12] + c[13]) / 2),
    }


class AlienCorridorEnv:
    def __init__(self, size=128):
        self.size = size
        s = size
        self.tunnel_w = 3.0
        self.tunnel_h = 3.0
        self.eye_h = 1.2
        self.focal = 1.2

        px, py = np.meshgrid(np.arange(s, dtype=np.float32), np.arange(s, dtype=np.float32))
        self.ndx = (px - s / 2) / (s / 2)
        self.ndy = (py - s / 2) / (s / 2)
        rdx = self.ndx / self.focal
        rdy = -self.ndy / self.focal
        rdz = np.ones_like(rdx)
        rl = np.sqrt(rdx**2 + rdy**2 + rdz**2)
        self.ray_dx = rdx / rl
        self.ray_dy = rdy / rl
        self.ray_dz = rdz / rl

        self.z_offset = 0.0
        self.flash = 0.0
        self.glow_phase = 0.0
        self.scan_phase = 0.0
        self.lateral = 0.0
        self._rng = np.random.default_rng(42)

    def reset(self, seed=None):
        self._rng = np.random.default_rng(seed)
        self.z_offset = self._rng.random() * 10.0
        self.flash = 0.0
        self.glow_phase = self._rng.random() * np.pi * 2
        self.scan_phase = 0.0
        self.lateral = 0.0
        self.eye_h = 1.2

    def step(self, audio_context):
        af = unpack_context(audio_context)
        self.z_offset += 0.08 + af['bpm'] * 0.2
        # Diagonal sway — camera shifts left/right with bass, bobs up/down with onset
        self.eye_h = 1.2 + np.sin(self.z_offset * 0.3) * af['bass'] * 0.4
        self.tunnel_w = 3.0 * (0.75 + (1 - af['bass']) * 0.5)
        # Lateral shift
        self.lateral = np.sin(self.z_offset * 0.5) * 0.3 * (0.5 + af['bpm'] * 0.5)

    def render(self, audio_context):
        af = unpack_context(audio_context)
        s = self.size
        W = self.tunnel_w
        H = self.tunnel_h
        ey = self.eye_h
        lat = self.lateral  # lateral camera offset

        self.glow_phase += 0.06 + af['bass'] * 0.12
        self.scan_phase += 0.25 + af['mid'] * 0.4
        if af['onset'] > 0.3:
            self.flash = min(1.5, self.flash + af['onset'] * 2.0)
        self.flash *= 0.55

        # --- RAY INTERSECTIONS ---
        with np.errstate(divide='ignore', invalid='ignore'):
            t_f = np.where(self.ray_dy < -1e-6, -ey / self.ray_dy, 1e6)
            t_c = np.where(self.ray_dy > 1e-6, (H - ey) / self.ray_dy, 1e6)
            t_l = np.where(self.ray_dx < -1e-6, -W / self.ray_dx, 1e6)
            t_r = np.where(self.ray_dx > 1e-6, W / self.ray_dx, 1e6)
        t_f = np.where(t_f > 0, t_f, 1e6)
        t_c = np.where(t_c > 0, t_c, 1e6)
        t_l = np.where(t_l > 0, t_l, 1e6)
        t_r = np.where(t_r > 0, t_r, 1e6)
        t_min = np.minimum(np.minimum(t_f, t_c), np.minimum(t_l, t_r))
        t_min = np.clip(t_min, 0, 50)

        surf = np.zeros((s, s), dtype=np.int32)  # 0=floor, 1=ceil, 2=left, 3=right
        surf = np.where(t_min == t_c, 1, surf)
        surf = np.where(t_min == t_l, 2, surf)
        surf = np.where(t_min == t_r, 3, surf)

        hx = self.ray_dx * t_min + lat  # lateral camera sway
        hy = ey + self.ray_dy * t_min
        hz = self.ray_dz * t_min + self.z_offset
        depth = t_min
        df = np.clip(1 - depth / 35, 0, 1)  # depth fade

        is_w = (surf >= 2).astype(np.float32)
        is_f = (surf == 0).astype(np.float32)
        is_c = (surf == 1).astype(np.float32)

        # --- RIBS ---
        rs = max(0.7, 2.2 - af['bass'] * 1.2)
        rz = np.mod(hz, rs) / rs
        rib = np.clip(1 - np.abs(rz - 0.5) * 7, 0, 1)
        rib_g = np.clip(1 - np.abs(rz - 0.5) * 2.5, 0, 1)

        # Sub-ribs
        sn = max(1, 2 + int(af['high'] * 5))
        sz = np.mod(hz, rs / sn) / (rs / sn)
        sub = np.clip(1 - np.abs(sz - 0.5) * 13, 0, 1) * af['high'] * 0.5

        # --- SHAPES ON WALLS ---

        # Circular portholes
        pz = np.mod(hz + rs * 0.5, rs) / rs - 0.5
        py_p = np.mod(hy, 1.5) / 1.5 - 0.5
        circ_d = np.sqrt(pz**2 + py_p**2) * 3.5
        circle_ring = np.clip(1 - np.abs(circ_d - 1) * 5, 0, 1) * is_w * 0.5
        circle_fill = np.clip(1 - circ_d * 1.3, 0, 1) * is_w * 0.2

        # Diamond shapes (between portholes)
        dz = np.mod(hz, rs) / rs - 0.5
        dy_d = np.mod(hy + 0.75, 1.5) / 1.5 - 0.5
        diamond_d = np.abs(dz) + np.abs(dy_d)
        diamond_ring = np.clip(1 - np.abs(diamond_d - 0.3) * 12, 0, 1) * is_w * 0.35

        # Hexagonal panels
        hex_s = 0.8
        hex_z = np.mod(hz, hex_s) / hex_s - 0.5
        hex_y = np.mod(hy, hex_s * 0.866) / (hex_s * 0.866) - 0.5
        row_off = (np.floor(hy / (hex_s * 0.866)) % 2) * 0.5
        hex_zo = np.mod(hex_z + row_off + 0.5, 1) - 0.5
        hex_d = np.maximum(np.abs(hex_zo), np.abs(hex_y) * 1.15)
        hex_border = np.clip(1 - (0.45 - hex_d) * 18, 0, 1) * is_w * 0.2

        # Triangular markers (small warning triangles near floor on walls)
        tri_z = np.mod(hz, rs * 0.5) / (rs * 0.5) - 0.5
        tri_y = (hy - 0.3) * 3  # near floor
        tri_shape = (tri_y > 0) & (tri_y < 1) & (np.abs(tri_z) < tri_y * 0.4)
        tri_ring = np.clip(1 - np.abs(np.abs(tri_z) - tri_y * 0.35) * 15, 0, 1)
        tri_ring *= (tri_y > 0).astype(np.float32) * (tri_y < 1).astype(np.float32)
        triangles = tri_ring * is_w * 0.3

        # --- SHAPES ON FLOOR ---

        # Chevron arrows
        chev_z = np.mod(hz, rs * 0.4) / (rs * 0.4)
        chev_x = np.abs(hx) / W
        chev = np.clip(1 - np.abs(chev_z - chev_x * 0.7 - 0.15) * 14, 0, 1)
        chevrons = chev * is_f * (chev_x < 0.35).astype(np.float32) * 0.3

        # Floor grid
        ts = 1.0
        tx = np.mod(hx + 50, ts) / ts
        tz = np.mod(hz, ts) / ts
        fgrid = ((np.minimum(tx, 1 - tx) < 0.03) | (np.minimum(tz, 1 - tz) < 0.03)).astype(np.float32)
        floor_grid = fgrid * is_f * 0.18

        # Concentric floor rings (from center line)
        fl_rings = np.sin(np.abs(hx) * 5 + self.glow_phase) * is_f
        floor_rings = np.clip(fl_rings, 0, 1) * 0.08 * af['bass']

        # --- SHAPES ON CEILING ---

        # Diamond lattice
        cd_z = np.mod(hz, 1.2) / 1.2 - 0.5
        cd_x = np.mod(hx + 50, 1.2) / 1.2 - 0.5
        ceil_diamond = np.clip(1 - np.abs(np.abs(cd_z) + np.abs(cd_x) - 0.35) * 10, 0, 1)
        ceil_diamonds = ceil_diamond * is_c * 0.25

        # Circular ceiling lights
        cl_z = np.mod(hz + rs * 0.5, rs) / rs - 0.5
        cl_x = np.mod(hx + 50, 2.0) / 2.0 - 0.5
        ceil_light_d = np.sqrt(cl_z**2 + cl_x**2) * 4
        ceil_lights = np.clip(1 - ceil_light_d * 1.5, 0, 1) * is_c * 0.3

        # --- CROSS-BEAMS on walls ---
        ca1 = np.mod(hy * 2 + hz * 1.5, rs) / rs
        ca2 = np.mod(hy * 2 - hz * 1.5 + 10, rs) / rs
        c1 = np.clip(1 - np.abs(ca1 - 0.5) * 14, 0, 1)
        c2 = np.clip(1 - np.abs(ca2 - 0.5) * 14, 0, 1)
        cross = np.maximum(c1, c2) * is_w * (0.3 + af['mid'] * 0.5)

        # Floor diagonals
        fa1 = np.mod(hx * 1.5 + hz * 1.5, rs) / rs
        fa2 = np.mod(hx * 1.5 - hz * 1.5 + 10, rs) / rs
        fc1 = np.clip(1 - np.abs(fa1 - 0.5) * 14, 0, 1)
        fc2 = np.clip(1 - np.abs(fa2 - 0.5) * 14, 0, 1)
        floor_cross = np.maximum(fc1, fc2) * is_f * 0.25

        # Ceiling diagonals
        cca1 = np.mod(hx * 2 + hz * 2, rs) / rs
        cca2 = np.mod(hx * 2 - hz * 2 + 10, rs) / rs
        cc1 = np.clip(1 - np.abs(cca1 - 0.5) * 12, 0, 1)
        cc2 = np.clip(1 - np.abs(cca2 - 0.5) * 12, 0, 1)
        ceil_cross = np.maximum(cc1, cc2) * is_c * 0.25

        # --- PIPES along walls ---
        pipes = np.zeros_like(hy)
        for yp in [0.25, 0.75, 2.25, 2.75]:
            pipes += np.clip(1 - np.abs(hy - yp) * 9, 0, 1)
        pipes = np.clip(pipes, 0, 1) * is_w * 0.25

        # --- EDGE NEON ---
        ey_d = np.minimum(np.abs(hy), np.abs(hy - H))
        ex_d = np.minimum(np.abs(hx + W), np.abs(hx - W))
        w_edge = np.clip(1 - ey_d * 3, 0, 1) * is_w
        f_edge = np.clip(1 - ex_d / W * 2.5, 0, 1) * (is_f + is_c) * 0.35

        # Wall top/bottom neon strips
        wt_neon = (np.abs(hy) < 0.12).astype(np.float32) * is_w * 0.4
        wt_neon += (np.abs(hy - H) < 0.12).astype(np.float32) * is_w * 0.4

        # Scanning line
        sc_pos = np.mod(self.scan_phase, 30)
        sc_line = np.clip(1 - np.abs(hz - self.z_offset - sc_pos) * 3, 0, 1) * af['mid'] * 0.25

        # --- COMBINE NEON ---
        neon = np.clip(rib * 1.3 + w_edge * rib_g * 1.3 + sub * 0.6
                       + f_edge + wt_neon + circle_ring + diamond_ring
                       + chevrons + pipes * 0.4 + ceil_diamonds * 0.4
                       + ceil_lights + triangles + sc_line, 0, 1)

        struct = np.clip(cross + hex_border + floor_grid + floor_rings + circle_fill, 0, 1)

        # --- COLORS ---
        t_col = af['temperature']
        gp = 0.75 + 0.25 * np.sin(self.glow_phase * 2)
        nr, ng, nb = 20 + t_col * 235, 200 - t_col * 100, 255 - t_col * 230
        ar, ag, ab = 180 - t_col * 140, 60 + t_col * 80, 120 + t_col * 60
        br, bg, bb = 16 + t_col * 10, 20 + (1-t_col) * 7, 26 + (1-t_col) * 12
        fr, fg, fb = 12 + t_col * 8, 15 + (1-t_col) * 5, 20 + (1-t_col) * 9

        # --- COMPOSE ---
        r = np.zeros((s, s), dtype=np.float32)
        g = np.zeros((s, s), dtype=np.float32)
        b = np.zeros((s, s), dtype=np.float32)

        # Surfaces
        pid = np.floor(hz / 0.8) + np.floor(hy / 0.7) * 17
        psh = 0.8 + 0.2 * np.sin(pid * 3.7)
        r += is_w * br * psh; g += is_w * bg * psh; b += is_w * bb * psh
        r += is_f * (fr + floor_grid * 40); g += is_f * (fg + floor_grid * 50); b += is_f * (fb + floor_grid * 55)
        r += is_c * br * 0.7; g += is_c * bg * 0.7; b += is_c * bb * 0.8

        # Structure (accent color)
        r += struct * ar * 0.5 * df; g += struct * ag * 0.5 * df; b += struct * ab * 0.5 * df

        # Porthole inner glow
        r += circle_fill * (90 + t_col * 70); g += circle_fill * (50 + (1-t_col) * 40); b += circle_fill * (120 - t_col * 60)

        # Neon (primary)
        np_w = 0.4 + af['rms'] * 0.6
        r += neon * nr * np_w * gp * df; g += neon * ng * np_w * gp * df; b += neon * nb * np_w * gp * df

        # Rib glow bleed
        rb_s = rib_g * (0.12 + af['bass'] * 0.2) * df
        r += rb_s * nr * 0.06; g += rb_s * ng * 0.06; b += rb_s * nb * 0.06

        # Floor reflections
        refl = is_f * neon * 0.2 * df
        r += refl * nr * 0.25; g += refl * ng * 0.25; b += refl * nb * 0.25

        # Ceiling lights glow
        r += ceil_lights * nr * 0.4 * df; g += ceil_lights * ng * 0.4 * df; b += ceil_lights * nb * 0.4 * df

        # Depth fog
        fog = (1 - df) * 0.8
        r = r * (1 - fog) + 2 * fog; g = g * (1 - fog) + 2 * fog; b = b * (1 - fog) + 3 * fog

        # Brightness
        bri = 0.15 + af['rms'] * 0.85
        r *= bri; g *= bri; b *= bri

        # Flash
        if self.flash > 0.02:
            fl = df * self.flash
            r += fl * nr * 0.6; g += fl * ng * 0.6; b += fl * nb * 0.6

        # High boost
        if af['high'] > 0.2:
            hb = neon * af['high'] * 0.2 * df
            r += hb * nr * 0.2; g += hb * ng * 0.2; b += hb * nb * 0.2

        # Vignette
        vd = np.sqrt(self.ndx**2 + self.ndy**2)
        vig = np.clip(1 - 0.4 * vd ** 1.8, 0.3, 1)
        r *= vig; g *= vig; b *= vig

        return np.clip(np.stack([r, g, b], axis=-1), 0, 255).astype(np.uint8)


def _make_audio_context(t, step, profile, rng):
    ctx = np.zeros(16, dtype=np.float32)
    if profile == 'high':
        v = rng.uniform(0.7, 0.95, 7).astype(np.float32); v[3] = 1.0 if rng.random() < 0.2 else 0.0
    elif profile == 'low':
        v = rng.uniform(0.02, 0.15, 7).astype(np.float32)
    elif profile == 'ramp':
        v = np.array([t*0.9, 0.3+t*0.5, 0.2+t*0.3, 1.0 if step%30==0 else 0.0, 0.4+t*0.4, t, 0.2+t*0.7]).astype(np.float32)
    elif profile == 'pulse':
        on = (step//20)%2==0; v = rng.uniform(0.7,0.95,7) if on else rng.uniform(0.05,0.2,7); v=v.astype(np.float32); v[3]=1.0 if on and step%20==0 else 0.0
    elif profile == 'random':
        v = rng.uniform(0,1,7).astype(np.float32); v[3] = 1.0 if rng.random() < 0.15 else 0.0
    elif profile == 'sweep':
        v = np.array([1-t, 0.5, 0.3, 0.0, 0.5, t, 0.5+0.3*np.sin(t*6)]).astype(np.float32)
    else:
        v = np.zeros(7, dtype=np.float32)
    v = np.clip(v, 0, 1); ctx[0:14:2] = v; ctx[1:14:2] = v
    return ctx


def generate_episodes(output_dir, num_episodes=100, steps=200, size=128):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    env = AlienCorridorEnv(size=size)
    profiles = ['high', 'low', 'ramp', 'pulse', 'random', 'sweep']
    for ep in range(num_episodes):
        env.reset(seed=ep*7+42); profile = profiles[ep%len(profiles)]
        images, contexts = [], []; rng = np.random.default_rng(ep)
        for step in range(steps+1):
            ctx = _make_audio_context(step/max(steps,1), step, profile, rng)
            images.append(env.render(ctx)); contexts.append(ctx)
            if step < steps: env.step(ctx)
        np.savez_compressed(out/f'episode_{ep:04d}.npz', image=np.array(images,dtype=np.uint8),
            action=np.zeros((steps,3),dtype=np.float32), context=np.array(contexts,dtype=np.float32),
            reward=np.zeros(steps,dtype=np.float32), is_first=np.concatenate([np.array([1.0]),np.zeros(steps)]).astype(np.float32))
        if (ep+1)%10==0: print(f'Episode {ep+1}/{num_episodes} ({profile})')
    print(f'Done: {num_episodes} episodes in {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--output', default='data/abstract')
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()

    if args.preview:
        from PIL import Image
        Path('/tmp/aura_samples').mkdir(exist_ok=True)
        env = AlienCorridorEnv(size=args.size)
        env.reset(seed=42)
        for name, ctx in {
            '1_silence': np.zeros(16, dtype=np.float32),
            '2_bass': np.array([.95,.95,.3,.3,.1,.1,0,0,.5,.5,.2,.2,.9,.9,0,0], dtype=np.float32),
            '3_mid': np.array([.4,.4,.9,.9,.4,.4,0,0,.5,.5,.4,.4,.7,.7,0,0], dtype=np.float32),
            '4_warm': np.array([.6,.6,.5,.5,.3,.3,0,0,.5,.5,.95,.95,.8,.8,0,0], dtype=np.float32),
            '5_cool': np.array([.3,.3,.3,.3,.1,.1,0,0,.3,.3,.05,.05,.4,.4,0,0], dtype=np.float32),
            '6_onset': np.array([.5,.5,.5,.5,.5,.5,1,1,.6,.6,.5,.5,.8,.8,0,0], dtype=np.float32),
            '7_full': np.array([.9,.9,.8,.8,.7,.7,.3,.3,.8,.8,.6,.6,.95,.95,0,0], dtype=np.float32),
        }.items():
            Image.fromarray(env.render(ctx)).save(f'/tmp/aura_samples/{name}.png')
            print(f'{name}: {env.render(ctx).mean():.0f}')
        # Sweep GIF
        env.reset(seed=42); frames=[]
        for i in range(120):
            t=i/119; on=1.0 if i in [30,60,90] else 0.0
            ctx=np.array([t*.9,t*.9,.2+t*.6,.2+t*.6,.1+t*.5,.1+t*.5,on,on,.3+t*.5,.3+t*.5,t,t,.1+t*.85,.1+t*.85,0,0],dtype=np.float32)
            frames.append(Image.fromarray(env.render(ctx))); env.step(ctx)
        frames[0].save('/tmp/aura_samples/sweep.gif',save_all=True,append_images=frames[1:],duration=50,loop=0)
        # Bass GIF
        env.reset(seed=99); frames2=[]
        for i in range(80):
            b=.5+.5*np.sin(i*.25)
            ctx=np.array([b,b,.5,.5,.3,.3,0,0,.6,.6,.3,.3,.5+b*.4,.5+b*.4,0,0],dtype=np.float32)
            frames2.append(Image.fromarray(env.render(ctx))); env.step(ctx)
        frames2[0].save('/tmp/aura_samples/bass_pulse.gif',save_all=True,append_images=frames2[1:],duration=60,loop=0)
        # Temp GIF
        env.reset(seed=77); frames3=[]
        for i in range(80):
            tp=.5+.5*np.sin(i*.12)
            ctx=np.array([.5,.5,.5,.5,.3,.3,0,0,.5,.5,tp,tp,.7,.7,0,0],dtype=np.float32)
            frames3.append(Image.fromarray(env.render(ctx))); env.step(ctx)
        frames3[0].save('/tmp/aura_samples/temp_shift.gif',save_all=True,append_images=frames3[1:],duration=60,loop=0)
        print('Preview at /tmp/aura_samples/')
        # HTML
        html = """<!DOCTYPE html><html><head><title>AURA v5</title><style>
body{background:#050508;color:#fff;font-family:'Courier New',monospace;padding:20px}
h1{font-weight:200;letter-spacing:.3em;color:#33ffcc}h2{font-weight:200;color:#555;margin-top:30px;font-size:.9em}
img{image-rendering:pixelated;margin:5px;border:1px solid #1a1a2a}.row{display:flex;flex-wrap:wrap;gap:10px;margin-top:10px}
.card{text-align:center}.label{font-size:.65em;color:#444;margin-top:3px}p{color:#333;font-size:.8em}
</style></head><body><h1>AURA — Sci-Fi Corridor v5</h1>
<p>rectangular tunnel • circles • diamonds • hexagons • triangles • chevrons • pipes • cross-beams</p>
<h2>SWEEP (silence→full, cool→warm, 3 onset flashes)</h2><img src="sweep.gif" width="512" height="512">
<h2>BASS PULSE (tunnel breathes)</h2><img src="bass_pulse.gif" width="384" height="384">
<h2>TEMPERATURE (cyan↔orange)</h2><img src="temp_shift.gif" width="384" height="384">
<h2>PROFILES</h2><div class="row">"""
        for n in ['1_silence','2_bass','3_mid','4_warm','5_cool','6_onset','7_full']:
            lbl = n.split('_',1)[1]
            html += f'<div class="card"><img src="{n}.png" width="180" height="180"><div class="label">{lbl}</div></div>'
        html += '</div></body></html>'
        open('/tmp/aura_samples/preview.html','w').write(html)
    else:
        generate_episodes(args.output, args.episodes, args.steps, args.size)
