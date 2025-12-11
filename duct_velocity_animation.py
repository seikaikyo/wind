import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import argparse

# 設定中文字型
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def create_velocity_animation(output_format='gif', show_window=True):
    """
    建立風速剖面動畫

    參數:
        output_format: 'gif', 'mp4', 'webm', 或 'html'
        show_window: 是否顯示 matplotlib 視窗
    """
    # 模擬風管寬度 (0 到 1，代表從內壁到外壁)
    x = np.linspace(0, 1, 100)

    # 1. 理想/充分發展的流場 (位置 2) - 拋物線分佈
    velocity_developed = 45 * (1 - (2 * (x - 0.5))**2)

    # 2. 彎管後的偏差流場 (位置 1) - 偏斜分佈
    velocity_skewed_raw = 100 * x**4 * (1-x) * 5
    velocity_skewed = velocity_skewed_raw / np.mean(velocity_skewed_raw) * 30

    # 建立圖表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # === 左圖：風管截面動畫 ===
    duct_height = 0.6
    ax1.set_xlim(-0.3, 1.2)
    ax1.set_ylim(-0.1, 0.85)
    ax1.set_aspect('equal')
    ax1.set_title('風管截面 - 風速向量示意', fontsize=14)
    ax1.set_xlabel('流動方向 →', fontsize=12)

    # 繪製風管壁
    ax1.plot([0, 1], [0, 0], 'k-', linewidth=3)
    ax1.plot([0, 1], [duct_height, duct_height], 'k-', linewidth=3)
    ax1.text(0.5, -0.05, '內壁', ha='center', fontsize=10)
    ax1.text(0.5, duct_height + 0.02, '外壁', ha='center', fontsize=10)

    # 初始化箭頭列表
    n_arrows = 12
    arrow_y = np.linspace(0.05, duct_height - 0.05, n_arrows)
    arrows = []

    # === 右圖：速度剖面圖 ===
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 60)
    ax2.set_title('風速剖面分佈', fontsize=14)
    ax2.set_xlabel('風管截面位置 (0=內側, 1=外側)', fontsize=12)
    ax2.set_ylabel('局部風速 (換算流率 CMM)', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)

    # 繪製參考線
    ax2.plot(x, velocity_developed, 'g--', linewidth=2, alpha=0.3, label='目標：均勻分佈')
    ax2.plot(x, velocity_skewed, 'r--', linewidth=2, alpha=0.3, label='起始：偏斜分佈')

    # 當前分佈線
    line, = ax2.plot([], [], 'b-', linewidth=3, label='當前分佈')

    # 探針位置標記
    probe_pos = 0.8
    probe_point, = ax2.plot([], [], 'ro', markersize=12, zorder=5)
    probe_text = ax2.text(0.5, 55, '', fontsize=12, ha='center',
                          bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    # 階段文字
    stage_text = ax1.text(0.5, 0.78, '', fontsize=13, ha='center', fontweight='bold',
                          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))

    ax2.legend(loc='upper left', fontsize=10)

    # 減少幀數加速生成 (60 幀)
    total_frames = 60

    def init():
        line.set_data([], [])
        probe_point.set_data([], [])
        return line, probe_point, probe_text, stage_text

    def animate(frame):
        # 清除之前的箭頭
        for arrow in arrows:
            arrow.remove()
        arrows.clear()

        # 計算過渡進度
        if frame < 15:
            progress = 0
            stage = "階段 1：彎管後 - 風場偏斜"
            stage_color = 'lightcoral'
        elif frame < 45:
            progress = (frame - 15) / 30
            stage = f"階段 2：風場發展中... ({progress*100:.0f}%)"
            stage_color = 'lightyellow'
        else:
            progress = 1
            stage = "階段 3：直管段 - 風場均勻"
            stage_color = 'lightgreen'

        # 內插當前速度分佈
        current_velocity = velocity_skewed * (1 - progress) + velocity_developed * progress

        # 更新速度剖面線
        line.set_data(x, current_velocity)

        # 更新探針讀數
        probe_reading = np.interp(probe_pos, x, current_velocity)
        probe_point.set_data([probe_pos], [probe_reading])
        probe_text.set_text(f'探針讀數 (x=0.8): {probe_reading:.1f} CMM')

        # 更新階段文字
        stage_text.set_text(stage)
        stage_text.set_bbox(dict(boxstyle='round', facecolor=stage_color, alpha=0.9))

        # 繪製風速向量箭頭
        for i, y_pos in enumerate(arrow_y):
            idx = int((y_pos / duct_height) * (len(x) - 1))
            v = current_velocity[idx] / 45

            # 箭頭顏色
            color = plt.cm.RdYlGn(0.3 + 0.7 * (1 - abs(v - 1)))

            arrow = ax1.arrow(0.15, y_pos, v * 0.5, 0,
                             head_width=0.025, head_length=0.025,
                             fc=color, ec='black', linewidth=0.5)
            arrows.append(arrow)

        return line, probe_point, probe_text, stage_text

    # 建立動畫
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=total_frames, interval=80, blit=False)

    plt.tight_layout()

    # 根據格式儲存
    output_file = f'duct_velocity_animation.{output_format}'
    print(f"正在生成 {output_format.upper()} 動畫...")

    try:
        if output_format == 'gif':
            anim.save(output_file, writer='pillow', fps=12)
        elif output_format == 'mp4':
            anim.save(output_file, writer='ffmpeg', fps=15,
                     extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
        elif output_format == 'webm':
            anim.save(output_file, writer='ffmpeg', fps=15,
                     extra_args=['-vcodec', 'libvpx-vp9'])
        elif output_format == 'html':
            # 輸出為 HTML5 (內嵌 JavaScript)
            html_content = anim.to_jshtml()
            with open(output_file, 'w') as f:
                f.write(html_content)
        print(f"已儲存：{output_file}")
    except Exception as e:
        print(f"儲存 {output_format} 失敗: {e}")
        if output_format != 'gif':
            print("嘗試改用 GIF 格式...")
            try:
                anim.save('duct_velocity_animation.gif', writer='pillow', fps=12)
                print("已儲存：duct_velocity_animation.gif")
            except Exception as e2:
                print(f"GIF 也失敗: {e2}")

    if show_window:
        plt.show()
    else:
        plt.close()

    return anim


def create_particle_animation(output_format='gif', show_window=True):
    """建立粒子流動動畫"""

    n_particles = 40
    x = np.linspace(0, 1, 100)

    velocity_developed = 45 * (1 - (2 * (x - 0.5))**2)
    velocity_skewed_raw = 100 * x**4 * (1-x) * 5
    velocity_skewed = velocity_skewed_raw / np.mean(velocity_skewed_raw) * 30

    fig, ax = plt.subplots(figsize=(12, 5))

    duct_length = 10
    duct_height = 1

    ax.set_xlim(0, duct_length)
    ax.set_ylim(-0.3, duct_height + 0.3)
    ax.set_title('風管內粒子流動模擬\n(觀察風場從偏斜到均勻的發展過程)', fontsize=14)
    ax.set_xlabel('風管長度方向', fontsize=12)
    ax.set_ylabel('風管高度', fontsize=12)

    # 繪製風管壁
    ax.fill_between([0, duct_length], [-0.1, -0.1], [0, 0], color='gray', alpha=0.3)
    ax.fill_between([0, duct_length], [duct_height, duct_height], [duct_height+0.1, duct_height+0.1], color='gray', alpha=0.3)
    ax.plot([0, duct_length], [0, 0], 'k-', linewidth=3)
    ax.plot([0, duct_length], [duct_height, duct_height], 'k-', linewidth=3)

    # 標示區域
    ax.axvline(x=3, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=7, color='gray', linestyle='--', alpha=0.5)
    ax.text(1.5, duct_height + 0.18, '彎管後\n(偏斜)', ha='center', fontsize=10, color='red')
    ax.text(5, duct_height + 0.18, '過渡區域', ha='center', fontsize=10, color='orange')
    ax.text(8.5, duct_height + 0.18, '發展區域\n(均勻)', ha='center', fontsize=10, color='green')

    # 初始化粒子
    particles_x = np.random.uniform(0, duct_length, n_particles)
    particles_y = np.random.uniform(0.08, duct_height - 0.08, n_particles)
    scatter = ax.scatter(particles_x, particles_y, c='blue', s=40, alpha=0.8)

    # 圖例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=plt.cm.coolwarm(0.9), label='高速'),
        Patch(facecolor=plt.cm.coolwarm(0.1), label='低速')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    def get_velocity_at_position(px, py):
        y_norm = np.clip(py / duct_height, 0, 1)

        if px < 3:
            mix = 0
        elif px < 7:
            mix = (px - 3) / 4
        else:
            mix = 1

        idx = int(y_norm * 99)
        v_skewed = velocity_skewed[idx]
        v_developed = velocity_developed[idx]

        return v_skewed * (1 - mix) + v_developed * mix

    def animate(frame):
        nonlocal particles_x, particles_y

        for i in range(n_particles):
            v = get_velocity_at_position(particles_x[i], particles_y[i])
            particles_x[i] += v * 0.025

            if particles_x[i] > duct_length:
                particles_x[i] = 0
                particles_y[i] = np.random.uniform(0.08, duct_height - 0.08)

        colors = []
        for i in range(n_particles):
            v = get_velocity_at_position(particles_x[i], particles_y[i])
            v_norm = np.clip((v - 10) / 40, 0, 1)
            colors.append(plt.cm.coolwarm(v_norm))

        scatter.set_offsets(np.column_stack([particles_x, particles_y]))
        scatter.set_facecolors(colors)

        return scatter,

    # 減少幀數
    anim = animation.FuncAnimation(fig, animate, frames=100, interval=60, blit=False)

    plt.tight_layout()

    output_file = f'duct_particle_animation.{output_format}'
    print(f"正在生成粒子動畫 ({output_format.upper()})...")

    try:
        if output_format == 'gif':
            anim.save(output_file, writer='pillow', fps=15)
        elif output_format == 'mp4':
            anim.save(output_file, writer='ffmpeg', fps=20)
        elif output_format == 'html':
            html_content = anim.to_jshtml()
            with open(output_file, 'w') as f:
                f.write(html_content)
        print(f"已儲存：{output_file}")
    except Exception as e:
        print(f"儲存失敗: {e}")

    if show_window:
        plt.show()
    else:
        plt.close()

    return anim


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='風管速度分佈動畫模擬')
    parser.add_argument('--format', '-f', choices=['gif', 'mp4', 'webm', 'html'],
                        default='gif', help='輸出格式 (預設: gif)')
    parser.add_argument('--no-show', action='store_true',
                        help='不顯示視窗，只儲存檔案')
    parser.add_argument('--velocity-only', action='store_true',
                        help='只生成速度剖面動畫')
    parser.add_argument('--particle-only', action='store_true',
                        help='只生成粒子動畫')

    args = parser.parse_args()
    show = not args.no_show

    print("=" * 50)
    print("風管速度分佈動畫模擬")
    print(f"輸出格式: {args.format.upper()}")
    print("=" * 50)

    if not args.particle_only:
        print("\n1. 生成速度剖面動畫...")
        create_velocity_animation(output_format=args.format, show_window=show)

    if not args.velocity_only:
        print("\n2. 生成粒子流動動畫...")
        create_particle_animation(output_format=args.format, show_window=show)

    print("\n動畫生成完成！")
