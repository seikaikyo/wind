"""
將 CFD 計算結果匯出成 JSON，供 HTML 互動式模擬使用
"""

import numpy as np
import json
from duct_cfd_real import solve_navier_stokes_bend

def export_cfd_to_json():
    """執行 CFD 計算並匯出結果"""

    nx, ny = 120, 120
    u, v, speed, mask, boundary = solve_navier_stokes_bend(
        nx=nx, ny=ny, n_iterations=800, Re=150
    )

    # 正規化速度
    speed_valid = speed[~np.isnan(speed)]
    speed_min = np.nanmin(speed) if len(speed_valid) > 0 else 0
    speed_max = np.nanmax(speed) if len(speed_valid) > 0 else 1
    speed_norm = (speed - speed_min) / (speed_max - speed_min + 1e-10)

    # 轉換為 JSON 格式
    # 將 NaN 轉換為 -1 (表示管道外)
    speed_list = []
    u_list = []
    v_list = []

    for i in range(ny):
        row_speed = []
        row_u = []
        row_v = []
        for j in range(nx):
            if mask[i, j]:
                row_speed.append(round(float(speed_norm[i, j]), 3))
                row_u.append(round(float(u[i, j]), 4))
                row_v.append(round(float(v[i, j]), 4))
            else:
                row_speed.append(-1)
                row_u.append(0)
                row_v.append(0)
        speed_list.append(row_speed)
        u_list.append(row_u)
        v_list.append(row_v)

    data = {
        'nx': nx,
        'ny': ny,
        'speed': speed_list,
        'u': u_list,
        'v': v_list,
        'geometry': {
            'inner_r': 0.25,
            'outer_r': 0.45,
            'cx': 0.45,
            'cy': 0.65
        }
    }

    # 儲存為 JSON
    with open('cfd_data.json', 'w') as f:
        json.dump(data, f)

    print(f"已匯出 CFD 資料: cfd_data.json")
    print(f"  網格大小: {nx} x {ny}")
    print(f"  檔案大小: {len(json.dumps(data)) / 1024:.1f} KB")

    return data

if __name__ == "__main__":
    export_cfd_to_json()
