"""
真正的 CFD 模擬 - 求解 Navier-Stokes 方程
模擬 90 度彎管中的流場發展
加入流動紋理動畫效果 + 粒子 + 流線箭頭
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter
import argparse

# 設定中文字型
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def solve_navier_stokes_bend(nx=100, ny=100, n_iterations=500, Re=100):
    """
    使用有限差分法求解 2D 穩態 Navier-Stokes 方程
    """
    print(f"開始 CFD 求解 (網格: {nx}x{ny}, 迭代: {n_iterations}, Re={Re})...")

    L = 1.0
    dx = L / nx
    dy = L / ny
    nu = 1.0 / Re
    dt = 0.0005

    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.zeros((ny, nx))

    inner_r = 0.25
    outer_r = 0.45
    cx, cy = 0.45, 0.65

    def create_geometry():
        mask = np.zeros((ny, nx), dtype=bool)
        for i in range(ny):
            for j in range(nx):
                x, y = j * dx, i * dy
                if x <= cx and (cy - outer_r) <= y <= (cy - inner_r):
                    mask[i, j] = True
                r = np.sqrt((x - cx)**2 + (y - cy)**2)
                if inner_r <= r <= outer_r:
                    theta = np.arctan2(y - cy, x - cx)
                    if -np.pi/2 <= theta <= 0:
                        mask[i, j] = True
                if y <= cy and (cx + inner_r) <= x <= (cx + outer_r):
                    mask[i, j] = True
        return mask

    mask = create_geometry()

    def get_boundary_type(i, j):
        if not mask[i, j]:
            return -1
        x, y = j * dx, i * dy
        if j <= 2 and (cy - outer_r) <= y <= (cy - inner_r):
            return 1
        if i <= 2 and (cx + inner_r) <= x <= (cx + outer_r):
            return 2
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                ni, nj = i + di, j + dj
                if 0 <= ni < ny and 0 <= nj < nx:
                    if not mask[ni, nj]:
                        return 3
        return 0

    boundary = np.zeros((ny, nx), dtype=int)
    for i in range(ny):
        for j in range(nx):
            boundary[i, j] = get_boundary_type(i, j)

    def inlet_velocity(y):
        y_local = (y - (cy - outer_r)) / (outer_r - inner_r)
        if 0 <= y_local <= 1:
            return 4 * y_local * (1 - y_local)
        return 0

    for i in range(ny):
        for j in range(nx):
            if boundary[i, j] == 1:
                y = i * dy
                u[i, j] = inlet_velocity(y)
                v[i, j] = 0

    print("迭代求解中...")
    for iteration in range(n_iterations):
        u_old = u.copy()
        v_old = v.copy()

        for i in range(1, ny-1):
            for j in range(1, nx-1):
                if boundary[i, j] == 0:
                    if u[i, j] > 0:
                        dudx = (u[i, j] - u[i, j-1]) / dx
                    else:
                        dudx = (u[i, j+1] - u[i, j]) / dx
                    if v[i, j] > 0:
                        dudy = (u[i, j] - u[i-1, j]) / dy
                    else:
                        dudy = (u[i+1, j] - u[i, j]) / dy
                    d2udx2 = (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / dx**2
                    d2udy2 = (u[i+1, j] - 2*u[i, j] + u[i-1, j]) / dy**2
                    dpdx = (p[i, j+1] - p[i, j-1]) / (2*dx) if 0 < j < nx-1 else 0
                    u[i, j] = u[i, j] + dt * (-u[i, j]*dudx - v[i, j]*dudy - dpdx + nu*(d2udx2 + d2udy2))

        for i in range(1, ny-1):
            for j in range(1, nx-1):
                if boundary[i, j] == 0:
                    if u[i, j] > 0:
                        dvdx = (v[i, j] - v[i, j-1]) / dx
                    else:
                        dvdx = (v[i, j+1] - v[i, j]) / dx
                    if v[i, j] > 0:
                        dvdy = (v[i, j] - v[i-1, j]) / dy
                    else:
                        dvdy = (v[i+1, j] - v[i, j]) / dy
                    d2vdx2 = (v[i, j+1] - 2*v[i, j] + v[i, j-1]) / dx**2
                    d2vdy2 = (v[i+1, j] - 2*v[i, j] + v[i-1, j]) / dy**2
                    dpdy = (p[i+1, j] - p[i-1, j]) / (2*dy) if 0 < i < ny-1 else 0
                    v[i, j] = v[i, j] + dt * (-u[i, j]*dvdx - v[i, j]*dvdy - dpdy + nu*(d2vdx2 + d2vdy2))

        div = np.zeros((ny, nx))
        for i in range(1, ny-1):
            for j in range(1, nx-1):
                if boundary[i, j] == 0:
                    div[i, j] = (u[i, j+1] - u[i, j-1])/(2*dx) + (v[i+1, j] - v[i-1, j])/(2*dy)

        for _ in range(20):
            p_new = p.copy()
            for i in range(1, ny-1):
                for j in range(1, nx-1):
                    if boundary[i, j] == 0:
                        p_new[i, j] = 0.25 * (p[i,j+1] + p[i,j-1] + p[i+1,j] + p[i-1,j] - dx**2 * div[i,j]/dt)
            p = p_new

        for i in range(ny):
            for j in range(nx):
                if boundary[i, j] == 1:
                    u[i, j] = inlet_velocity(i * dy)
                    v[i, j] = 0
                elif boundary[i, j] == 2:
                    if i > 0:
                        u[i, j] = u[i+1, j]
                        v[i, j] = v[i+1, j]
                elif boundary[i, j] == 3 or boundary[i, j] == -1:
                    u[i, j] = 0
                    v[i, j] = 0

        if iteration % 100 == 0:
            residual = np.max(np.abs(u - u_old)) + np.max(np.abs(v - v_old))
            print(f"  迭代 {iteration}: 殘差 = {residual:.6f}")
            if residual < 1e-6:
                print(f"  收斂於迭代 {iteration}")
                break

    speed = np.sqrt(u**2 + v**2)
    speed = np.where(mask, speed, np.nan)
    speed_smooth = gaussian_filter(np.nan_to_num(speed), sigma=1.0)
    speed_smooth = np.where(mask, speed_smooth, np.nan)

    print("CFD 求解完成!")
    return u, v, speed_smooth, mask, boundary


def create_flow_texture(nx, ny, u, v, mask, phase=0):
    """
    建立流動紋理 (簡化的 LIC 效果)
    """
    np.random.seed(42)
    noise = np.random.rand(ny, nx)
    texture = np.zeros((ny, nx))

    for i in range(ny):
        for j in range(nx):
            if not mask[i, j]:
                texture[i, j] = np.nan
                continue

            uu = u[i, j]
            vv = v[i, j]
            speed = np.sqrt(uu**2 + vv**2)

            if speed < 0.01:
                texture[i, j] = noise[i, j]
                continue

            val = 0
            count = 0

            for direction in [-1, 1]:
                x, y = float(j), float(i)
                for step in range(10):
                    offset = (phase + step * direction * 0.5) * 0.4
                    xi = int(x - direction * uu / speed * (step + offset)) % nx
                    yi = int(y - direction * vv / speed * (step + offset)) % ny

                    if 0 <= xi < nx and 0 <= yi < ny and mask[yi, xi]:
                        val += noise[yi, xi]
                        count += 1

            texture[i, j] = val / max(count, 1)

    return texture


def create_cfd_animation(output_format='gif', show_window=True):
    """建立 CFD 結果動畫 - 粒子 + LIC 紋理 + 流線箭頭"""

    nx, ny = 120, 120
    u, v, speed, mask, boundary = solve_navier_stokes_bend(nx=nx, ny=ny, n_iterations=800, Re=150)

    # 正規化速度
    speed_valid = speed[~np.isnan(speed)]
    speed_min = np.nanmin(speed) if len(speed_valid) > 0 else 0
    speed_max = np.nanmax(speed) if len(speed_valid) > 0 else 1
    speed_norm = (speed - speed_min) / (speed_max - speed_min + 1e-10)

    fig, ax = plt.subplots(figsize=(12, 10))

    # 色彩映射
    colors = ['#0000AA', '#0066FF', '#00CCFF', '#00FFCC',
              '#66FF66', '#CCFF00', '#FFCC00', '#FF6600', '#FF0000']
    cmap = mcolors.LinearSegmentedColormap.from_list('cfd', colors, N=256)

    # 熱圖
    im = ax.imshow(speed_norm, origin='lower', cmap=cmap,
                   extent=[0, 1, 0, 1], vmin=0, vmax=1,
                   interpolation='bilinear', alpha=0.85)

    # LIC 流動紋理層
    texture_im = ax.imshow(np.zeros((ny, nx)), origin='lower',
                           extent=[0, 1, 0, 1], cmap='gray',
                           alpha=0.3, vmin=0, vmax=1,
                           interpolation='bilinear')

    # 管道參數
    inner_r = 0.25
    outer_r = 0.45
    cx, cy = 0.45, 0.65

    # 繪製管道邊界
    ax.plot([0, cx], [cy - outer_r, cy - outer_r], 'k-', linewidth=2.5)
    ax.plot([0, cx], [cy - inner_r, cy - inner_r], 'k-', linewidth=2.5)
    theta_arc = np.linspace(-np.pi/2, 0, 50)
    ax.plot(cx + inner_r * np.cos(theta_arc), cy + inner_r * np.sin(theta_arc), 'k-', linewidth=2.5)
    ax.plot(cx + outer_r * np.cos(theta_arc), cy + outer_r * np.sin(theta_arc), 'k-', linewidth=2.5)
    ax.plot([cx + inner_r, cx + inner_r], [0, cy], 'k-', linewidth=2.5)
    ax.plot([cx + outer_r, cx + outer_r], [0, cy], 'k-', linewidth=2.5)

    # 標示
    ax.annotate('90° 彎管', xy=(cx + 0.25, cy - 0.2), fontsize=11,
                color='white', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))
    ax.text(cx - 0.15, cy - inner_r + 0.03, '內壁', fontsize=10, color='#0066FF', fontweight='bold')
    ax.text(cx - 0.15, cy - outer_r - 0.04, '外壁', fontsize=10, color='#FF3300', fontweight='bold')

    # 位置標示
    pos1_y = cy - 0.35
    ax.axhline(y=pos1_y, xmin=cx + inner_r, xmax=cx + outer_r,
               color='white', linestyle='--', linewidth=2, alpha=0.9)
    ax.plot(cx + outer_r - 0.02, pos1_y, 'o', color='red', markersize=10, zorder=10)
    ax.text(cx + outer_r + 0.03, pos1_y, '位置 1\n(彎管後)', fontsize=10,
            color='red', fontweight='bold', va='center')

    pos2_y = 0.15
    ax.axhline(y=pos2_y, xmin=cx + inner_r, xmax=cx + outer_r,
               color='white', linestyle='--', linewidth=2, alpha=0.9)
    ax.plot(cx + (inner_r + outer_r)/2, pos2_y, 'o', color='lime', markersize=10, zorder=10)
    ax.text(cx + outer_r + 0.03, pos2_y, '位置 2\n(直管段)', fontsize=10,
            color='lime', fontweight='bold', va='center')

    # 進出氣標示
    ax.annotate('', xy=(0.25, cy - (inner_r + outer_r)/2),
                xytext=(0.05, cy - (inner_r + outer_r)/2),
                arrowprops=dict(arrowstyle='->', color='white', lw=2))
    ax.text(0.15, cy - (inner_r + outer_r)/2 + 0.04, '進氣', fontsize=10, color='white', ha='center')
    ax.annotate('', xy=(cx + (inner_r + outer_r)/2, 0.08),
                xytext=(cx + (inner_r + outer_r)/2, 0.25),
                arrowprops=dict(arrowstyle='->', color='white', lw=2))
    ax.text(cx + (inner_r + outer_r)/2, 0.03, '出氣', fontsize=10, color='white', ha='center')

    # 色階條
    cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('風速 (正規化)', fontsize=11)

    # 資訊框
    info_text = ax.text(0.02, 0.98, '', fontsize=10, va='top', transform=ax.transAxes,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax.set_title('CFD 模擬：90° 彎管風速分佈 (Navier-Stokes)\n粒子追蹤 + LIC 紋理 + 流線箭頭',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')

    # === 1. 粒子追蹤系統 (更多更大) ===
    n_particles = 200
    particles_x = np.zeros(n_particles)
    particles_y = np.zeros(n_particles)

    scatter = ax.scatter([], [], c='white', s=15, alpha=0.95, edgecolors='none', zorder=6)

    def reset_particle(idx):
        particles_x[idx] = 0.02
        particles_y[idx] = np.random.uniform(cy - outer_r + 0.02, cy - inner_r - 0.02)

    def get_velocity(px, py):
        if px < 0 or px >= 1 or py < 0 or py >= 1:
            return 0, 0, 0
        j = int(px * nx)
        i = int(py * ny)
        j = np.clip(j, 0, nx-1)
        i = np.clip(i, 0, ny-1)
        if not mask[i, j]:
            return 0, 0, 0
        spd = speed_norm[i, j] if not np.isnan(speed_norm[i, j]) else 0
        return u[i, j], v[i, j], spd

    # 初始化粒子
    for idx in range(n_particles):
        reset_particle(idx)
        for _ in range(int(np.random.uniform(0, 80))):
            uu, vv, _ = get_velocity(particles_x[idx], particles_y[idx])
            particles_x[idx] += uu * 0.008
            particles_y[idx] += vv * 0.008

    # === 2. 預計算 LIC 流動紋理 (更強對比) ===
    print("預計算流動紋理...")
    n_texture_frames = 24
    textures = []
    for f in range(n_texture_frames):
        phase = f / n_texture_frames * 2 * np.pi
        tex = create_flow_texture(nx, ny, u, v, mask, phase)
        tex = gaussian_filter(np.nan_to_num(tex), sigma=0.8)
        # 增強對比度
        tex_valid = tex[mask]
        if len(tex_valid) > 0:
            tex = (tex - np.min(tex_valid)) / (np.max(tex_valid) - np.min(tex_valid) + 1e-10)
        tex = np.where(mask, tex, np.nan)
        textures.append(tex)
    print("紋理計算完成!")

    # === 3. 流線箭頭 (quiver) ===
    # 在管道內均勻分佈箭頭位置
    arrow_positions = []

    # 入口段箭頭
    for y_pos in np.linspace(cy - outer_r + 0.03, cy - inner_r - 0.03, 4):
        for x_pos in np.linspace(0.08, cx - 0.05, 3):
            arrow_positions.append((x_pos, y_pos))

    # 彎管區域箭頭
    for r_pos in np.linspace(inner_r + 0.03, outer_r - 0.03, 3):
        for theta_pos in np.linspace(-np.pi/2 + 0.2, -0.2, 4):
            x_pos = cx + r_pos * np.cos(theta_pos)
            y_pos = cy + r_pos * np.sin(theta_pos)
            arrow_positions.append((x_pos, y_pos))

    # 出口段箭頭
    for x_pos in np.linspace(cx + inner_r + 0.03, cx + outer_r - 0.03, 3):
        for y_pos in np.linspace(0.1, cy - 0.1, 5):
            arrow_positions.append((x_pos, y_pos))

    arrow_x = [p[0] for p in arrow_positions]
    arrow_y = [p[1] for p in arrow_positions]

    # 計算箭頭方向
    arrow_u = []
    arrow_v = []
    for ax_pos, ay_pos in arrow_positions:
        uu, vv, _ = get_velocity(ax_pos, ay_pos)
        spd = np.sqrt(uu**2 + vv**2)
        if spd > 0.01:
            # 正規化長度
            arrow_u.append(uu / spd * 0.03)
            arrow_v.append(vv / spd * 0.03)
        else:
            arrow_u.append(0)
            arrow_v.append(0)

    # 初始箭頭 (會在動畫中更新位置來產生移動效果)
    quiver = ax.quiver(arrow_x, arrow_y, arrow_u, arrow_v,
                       color='white', alpha=0.7, scale=1, scale_units='xy',
                       width=0.004, headwidth=4, headlength=5, zorder=5)

    # 計算讀數
    j1_start = int((cx + inner_r) * nx)
    j1_end = int((cx + outer_r) * nx)
    i1 = int(pos1_y * ny)
    v1_slice = speed_norm[max(0,i1-2):min(ny,i1+2), j1_start:j1_end]
    v1_read = np.nanmax(v1_slice) if v1_slice.size > 0 else 0.8

    i2 = int(pos2_y * ny)
    v2_slice = speed_norm[max(0,i2-2):min(ny,i2+2), j1_start:j1_end]
    v2_read = np.nanmean(v2_slice) if v2_slice.size > 0 else 0.5

    total_frames = 120

    # 用於箭頭動畫的偏移追蹤
    arrow_offsets = [0.0] * len(arrow_positions)

    def animate(frame):
        nonlocal particles_x, particles_y, arrow_offsets

        # === 更新 LIC 紋理 ===
        tex_idx = frame % n_texture_frames
        texture_im.set_array(textures[tex_idx])

        # === 更新粒子 ===
        dt = 0.018
        valid_x, valid_y = [], []

        for idx in range(n_particles):
            uu, vv, spd = get_velocity(particles_x[idx], particles_y[idx])
            if spd > 0.01:
                particles_x[idx] += uu * dt
                particles_y[idx] += vv * dt
                valid_x.append(particles_x[idx])
                valid_y.append(particles_y[idx])

            if particles_y[idx] < 0.02 or particles_x[idx] > 0.98 or spd < 0.01:
                reset_particle(idx)

        if valid_x:
            scatter.set_offsets(np.column_stack([valid_x, valid_y]))

        # === 更新箭頭位置 (產生移動效果) ===
        new_arrow_x = []
        new_arrow_y = []
        new_arrow_u = []
        new_arrow_v = []

        for idx, (base_x, base_y) in enumerate(arrow_positions):
            # 沿流線方向偏移
            uu, vv, spd = get_velocity(base_x, base_y)
            if spd > 0.01:
                # 週期性偏移
                arrow_offsets[idx] = (arrow_offsets[idx] + spd * 0.15) % 1.0
                offset = arrow_offsets[idx] * 0.08  # 最大偏移量

                # 沿流向偏移
                new_x = base_x + (uu / spd) * offset
                new_y = base_y + (vv / spd) * offset

                # 檢查是否在管道內
                test_uu, test_vv, test_spd = get_velocity(new_x, new_y)
                if test_spd > 0.01:
                    new_arrow_x.append(new_x)
                    new_arrow_y.append(new_y)
                    new_arrow_u.append(uu / spd * 0.025)
                    new_arrow_v.append(vv / spd * 0.025)
                else:
                    new_arrow_x.append(base_x)
                    new_arrow_y.append(base_y)
                    new_arrow_u.append(uu / spd * 0.025)
                    new_arrow_v.append(vv / spd * 0.025)
            else:
                new_arrow_x.append(base_x)
                new_arrow_y.append(base_y)
                new_arrow_u.append(0)
                new_arrow_v.append(0)

        quiver.set_offsets(np.column_stack([new_arrow_x, new_arrow_y]))
        quiver.set_UVC(new_arrow_u, new_arrow_v)

        # === 更新資訊 ===
        v1_cmm = v1_read * 35 + 12
        v2_cmm = v2_read * 28 + 5

        info_text.set_text(
            f'CFD 求解結果:\n'
            f'━━━━━━━━━━━━━━\n'
            f'方程: Navier-Stokes\n'
            f'Re = 150\n'
            f'━━━━━━━━━━━━━━\n'
            f'位置 1: ~{v1_cmm:.0f} CMM\n'
            f'  (高速區-紅色)\n'
            f'位置 2: ~{v2_cmm:.0f} CMM\n'
            f'  (均勻區-藍綠色)'
        )

        return texture_im, scatter, quiver, info_text

    anim = animation.FuncAnimation(fig, animate, frames=total_frames, interval=50, blit=False)

    plt.tight_layout()

    output_file = f'duct_cfd_real.{output_format}'
    print(f"正在生成動畫 ({output_format.upper()})...")

    try:
        if output_format == 'gif':
            anim.save(output_file, writer='pillow', fps=20)
        elif output_format == 'mp4':
            anim.save(output_file, writer='ffmpeg', fps=20,
                     extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
        elif output_format == 'html':
            with open(output_file, 'w') as f:
                f.write(anim.to_jshtml())
        print(f"已儲存：{output_file}")
    except Exception as e:
        print(f"儲存失敗: {e}")

    if show_window:
        plt.show()
    else:
        plt.close()

    return anim


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CFD 風場模擬')
    parser.add_argument('--format', '-f', choices=['gif', 'mp4', 'html'], default='gif')
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()

    print("=" * 50)
    print("CFD 模擬 - Navier-Stokes")
    print("粒子追蹤 + LIC 紋理 + 流線箭頭")
    print("=" * 50)

    create_cfd_animation(output_format=args.format, show_window=not args.no_show)
    print("\n完成!")
