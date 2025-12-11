import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import numpy as np
from scipy.ndimage import gaussian_filter
import argparse

# 設定中文字型
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def create_cfd_bend_animation(output_format='mp4', show_window=True):
    """
    建立 90° 彎管 CFD 風場模擬動畫
    使用流線 (streamlines) 展示連續的流動效果
    """

    fig, ax = plt.subplots(figsize=(12, 10))

    # 色彩映射 - 類似 CFD 軟體的漸層
    colors = ['#0000AA', '#0066FF', '#00CCFF', '#00FFCC',
              '#66FF66', '#CCFF00', '#FFCC00', '#FF6600', '#FF0000']
    cmap = mcolors.LinearSegmentedColormap.from_list('cfd', colors, N=256)

    # 網格設定
    nx, ny = 150, 150
    x = np.linspace(0, 150, nx)
    y = np.linspace(0, 150, ny)
    X, Y = np.meshgrid(x, y)

    # 管道參數
    inner_radius = 25
    outer_radius = 45
    duct_width = outer_radius - inner_radius
    cx, cy = 55, 100  # 彎管圓心

    def create_duct_mask():
        """建立管道遮罩"""
        mask = np.zeros((ny, nx), dtype=bool)

        for i in range(ny):
            for j in range(nx):
                # 入口水平段 (左側)
                if j <= cx and (cy - outer_radius) <= i <= (cy - inner_radius):
                    mask[i, j] = True

                # 彎管區域 (90度，第四象限)
                r = np.sqrt((j - cx)**2 + (i - cy)**2)
                if inner_radius <= r <= outer_radius:
                    theta = np.arctan2(i - cy, j - cx)
                    if -np.pi/2 <= theta <= 0:
                        mask[i, j] = True

                # 出口垂直段 (向下)
                if i <= cy and (cx + inner_radius) <= j <= (cx + outer_radius):
                    mask[i, j] = True

        return mask

    duct_mask = create_duct_mask()

    def compute_velocity_field():
        """
        計算穩態速度場 (不隨時間變化的基礎場)
        """
        U = np.zeros((ny, nx))  # x 方向速度
        V = np.zeros((ny, nx))  # y 方向速度
        speed = np.zeros((ny, nx))

        for i in range(ny):
            for j in range(nx):
                if not duct_mask[i, j]:
                    continue

                # 入口水平段
                if j <= cx and (cy - outer_radius) <= i <= (cy - inner_radius):
                    # 拋物線速度分佈
                    y_norm = (i - (cy - outer_radius)) / duct_width
                    vel = 0.4 + 0.6 * 4 * y_norm * (1 - y_norm)
                    U[i, j] = vel
                    V[i, j] = 0
                    speed[i, j] = vel

                # 彎管區域
                r = np.sqrt((j - cx)**2 + (i - cy)**2)
                if inner_radius <= r <= outer_radius:
                    theta = np.arctan2(i - cy, j - cx)
                    if -np.pi/2 <= theta <= 0:
                        # 徑向位置 (0=內側, 1=外側)
                        r_norm = (r - inner_radius) / duct_width

                        # 離心力效應：外側速度高
                        vel = 0.3 + 0.8 * r_norm**0.7

                        # 切線方向 (順時針)
                        U[i, j] = vel * np.sin(-theta)
                        V[i, j] = -vel * np.cos(-theta)
                        speed[i, j] = vel

                # 出口垂直段
                if i <= cy and (cx + inner_radius) <= j <= (cx + outer_radius):
                    x_norm = (j - (cx + inner_radius)) / duct_width
                    dist_from_bend = cy - i

                    if dist_from_bend < 25:
                        # 彎管後：速度偏向外壁 (x 較大)
                        recovery = dist_from_bend / 25
                        skew = 0.3 + 0.8 * x_norm**0.6
                        parabolic = 0.4 + 0.6 * 4 * x_norm * (1 - x_norm)
                        vel = skew * (1 - recovery) + parabolic * recovery
                    else:
                        # 已恢復
                        vel = 0.4 + 0.6 * 4 * x_norm * (1 - x_norm)

                    U[i, j] = 0
                    V[i, j] = -vel  # 向下
                    speed[i, j] = vel

        # 平滑處理
        speed = gaussian_filter(speed, sigma=1.5)
        speed = np.where(duct_mask, speed, np.nan)

        return U, V, speed

    U, V, speed = compute_velocity_field()

    # 繪製熱圖
    im = ax.imshow(speed, origin='lower', cmap=cmap,
                   extent=[0, 150, 0, 150], vmin=0.25, vmax=1.05,
                   interpolation='bilinear')

    # 繪製管道邊界
    # 入口段
    ax.plot([0, cx], [cy - outer_radius, cy - outer_radius], 'k-', linewidth=2.5)
    ax.plot([0, cx], [cy - inner_radius, cy - inner_radius], 'k-', linewidth=2.5)

    # 彎管圓弧
    theta_arc = np.linspace(-np.pi/2, 0, 50)
    ax.plot(cx + inner_radius * np.cos(theta_arc),
            cy + inner_radius * np.sin(theta_arc), 'k-', linewidth=2.5)
    ax.plot(cx + outer_radius * np.cos(theta_arc),
            cy + outer_radius * np.sin(theta_arc), 'k-', linewidth=2.5)

    # 出口段
    ax.plot([cx + inner_radius, cx + inner_radius], [0, cy], 'k-', linewidth=2.5)
    ax.plot([cx + outer_radius, cx + outer_radius], [0, cy], 'k-', linewidth=2.5)

    # === 流動粒子系統 ===
    n_particles = 80

    # 初始化粒子位置 (沿著入口分佈)
    particles_x = np.zeros(n_particles)
    particles_y = np.random.uniform(cy - outer_radius + 2, cy - inner_radius - 2, n_particles)
    particles_age = np.random.uniform(0, 1, n_particles)  # 用於錯開粒子

    scatter = ax.scatter([], [], c=[], cmap=cmap, s=25, alpha=0.8,
                        vmin=0.25, vmax=1.05, edgecolors='none')

    # 標示區域
    ax.annotate('90° 彎管\nTurn Bend', xy=(cx + 30, cy - 30), fontsize=11,
                color='white', fontweight='bold', ha='center',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))

    ax.text(cx - 30, cy - inner_radius + 5, '內壁', fontsize=10, color='#0066FF', fontweight='bold')
    ax.text(cx - 30, cy - outer_radius - 8, '外壁', fontsize=10, color='#FF3300', fontweight='bold')

    # 位置標示
    pos1_y = cy - 50
    ax.axhline(y=pos1_y, xmin=(cx + inner_radius)/150, xmax=(cx + outer_radius)/150,
               color='white', linestyle='--', linewidth=2, alpha=0.9)
    ax.plot(cx + outer_radius - 3, pos1_y, 'o', color='red', markersize=10, zorder=10)
    ax.text(cx + outer_radius + 5, pos1_y, '位置 1\n(彎管後)', fontsize=10,
            color='red', fontweight='bold', va='center')

    pos2_y = 25
    ax.axhline(y=pos2_y, xmin=(cx + inner_radius)/150, xmax=(cx + outer_radius)/150,
               color='white', linestyle='--', linewidth=2, alpha=0.9)
    ax.plot(cx + (inner_radius + outer_radius)/2, pos2_y, 'o', color='lime', markersize=10, zorder=10)
    ax.text(cx + outer_radius + 5, pos2_y, '位置 2\n(直管段)', fontsize=10,
            color='lime', fontweight='bold', va='center')

    # 流動方向
    ax.annotate('', xy=(35, cy - duct_width/2 - inner_radius),
                xytext=(10, cy - duct_width/2 - inner_radius),
                arrowprops=dict(arrowstyle='->', color='white', lw=2))
    ax.text(22, cy - duct_width/2 - inner_radius + 6, '進氣', fontsize=10, color='white', ha='center')

    ax.annotate('', xy=(cx + duct_width/2 + inner_radius, 15),
                xytext=(cx + duct_width/2 + inner_radius, 40),
                arrowprops=dict(arrowstyle='->', color='white', lw=2))
    ax.text(cx + duct_width/2 + inner_radius, 8, '出氣', fontsize=10, color='white', ha='center')

    # 色階條
    cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('風速 (相對值)', fontsize=11)

    # 資訊框
    info_box = ax.text(5, 145, '', fontsize=10, va='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax.set_title('CFD 模擬：90° 彎管風速分佈\n離心力使氣流偏向外壁', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 150)
    ax.set_aspect('equal')
    ax.axis('off')

    # 粒子追蹤函數
    def get_velocity_at(px, py):
        """取得指定位置的速度"""
        if px < 0 or px >= 150 or py < 0 or py >= 150:
            return 0, 0, 0

        j = int(px * nx / 150)
        i = int(py * ny / 150)
        j = np.clip(j, 0, nx-1)
        i = np.clip(i, 0, ny-1)

        if not duct_mask[i, j]:
            return 0, 0, 0

        return U[i, j], V[i, j], speed[i, j]

    def reset_particle(idx):
        """重置粒子到入口"""
        particles_x[idx] = 2
        particles_y[idx] = np.random.uniform(cy - outer_radius + 3, cy - inner_radius - 3)

    total_frames = 120

    def animate(frame):
        nonlocal particles_x, particles_y

        # 更新粒子位置
        dt = 2.5  # 時間步長
        colors_list = []
        valid_x = []
        valid_y = []

        for idx in range(n_particles):
            u, v, spd = get_velocity_at(particles_x[idx], particles_y[idx])

            if spd > 0:
                # 根據速度移動粒子
                particles_x[idx] += u * dt
                particles_y[idx] += v * dt
                valid_x.append(particles_x[idx])
                valid_y.append(particles_y[idx])
                colors_list.append(spd)

            # 粒子離開管道則重置
            if particles_y[idx] < 5 or particles_x[idx] > 145 or spd == 0:
                reset_particle(idx)

        # 更新粒子顯示
        if valid_x:
            scatter.set_offsets(np.column_stack([valid_x, valid_y]))
            scatter.set_array(np.array(colors_list))

        # 更新資訊
        v1_region = speed[int((pos1_y/150)*ny)-2:int((pos1_y/150)*ny)+2,
                         int((cx+inner_radius)/150*nx):int((cx+outer_radius)/150*nx)]
        v1_max = np.nanmax(v1_region) if v1_region.size > 0 else 0.9

        v2_region = speed[int((pos2_y/150)*ny)-2:int((pos2_y/150)*ny)+2,
                         int((cx+inner_radius)/150*nx):int((cx+outer_radius)/150*nx)]
        v2_avg = np.nanmean(v2_region) if v2_region.size > 0 else 0.65

        v1_cmm = v1_max * 42
        v2_cmm = v2_avg * 33

        info_box.set_text(
            f'Velocity Heatmap Key:\n'
            f'━━━━━━━━━━━━━━\n'
            f'紅色 = 高速 (外壁)\n'
            f'藍色 = 低速 (內壁)\n'
            f'━━━━━━━━━━━━━━\n'
            f'位置 1: ~{v1_cmm:.0f} CMM (偏高)\n'
            f'位置 2: ~{v2_cmm:.0f} CMM (正常)'
        )

        return scatter, info_box

    # 初始化粒子
    for idx in range(n_particles):
        particles_x[idx] = np.random.uniform(0, 40)
        reset_particle(idx)
        # 錯開初始位置
        for _ in range(int(np.random.uniform(0, 30))):
            u, v, _ = get_velocity_at(particles_x[idx], particles_y[idx])
            particles_x[idx] += u * 2.5
            particles_y[idx] += v * 2.5

    anim = animation.FuncAnimation(fig, animate, frames=total_frames,
                                   interval=50, blit=False)

    plt.tight_layout()

    output_file = f'duct_cfd_bend.{output_format}'
    print(f"正在生成 CFD 動畫 ({output_format.upper()})...")

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
    parser = argparse.ArgumentParser(description='CFD 風場模擬動畫')
    parser.add_argument('--format', '-f', choices=['gif', 'mp4', 'html'],
                        default='mp4', help='輸出格式 (預設: mp4)')
    parser.add_argument('--no-show', action='store_true', help='不顯示視窗')

    args = parser.parse_args()

    print("=" * 50)
    print("CFD 風場模擬動畫")
    print(f"輸出格式: {args.format.upper()}")
    print("=" * 50)

    create_cfd_bend_animation(output_format=args.format, show_window=not args.no_show)

    print("\n完成！")
