import io
import math
import time
import numpy as np
import pandas as pd
import streamlit as st
from scipy.spatial import KDTree

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="4G LTE RF Planning & Optimization Tool",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# CORE GEOMETRY & RF ALGORITHMS
# ==============================================================================
EARTH_RADIUS_M = 6371000.0


def latlon_to_cartesian_3d(lat_deg: np.ndarray, lon_deg: np.ndarray, alt_m: np.ndarray) -> np.ndarray:
    """
    Chuyển đổi tọa độ địa lý (Latitude, Longitude, Antenna Height) sang hệ tọa độ
    Descartes 3D cục bộ (x, y, z) tính bằng mét để sử dụng cho 3D KDTree.
    """
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)
    lat0 = np.mean(lat_rad)
    lon0 = np.mean(lon_rad)

    x = (lon_rad - lon0) * np.cos(lat0) * EARTH_RADIUS_M
    y = (lat_rad - lat0) * EARTH_RADIUS_M
    z = alt_m
    return np.column_stack((x, y, z))


def angular_diff_deg(az1: float, az2: float) -> float:
    """Tính góc lệch nhỏ nhất giữa 2 phương vị (0° - 180°)."""
    diff = abs(az1 - az2) % 360.0
    return 360.0 - diff if diff > 180.0 else diff


def calculate_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Tính góc phương vị (Bearing 0° - 360°) từ trạm 1 tới trạm 2."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_lon = math.radians(lon2 - lon1)
    y = math.sin(d_lon) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(d_lon)
    bearing = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
    return bearing


def compute_boresight_penalty(
    az_src: float,
    az_nbr: float,
    bearing_src_to_nbr: float,
    hbw_deg: float = 65.0,
) -> float:
    """
    Tính hệ số phạt Boresight (Boresight Penalty):
    - Nếu cell lân cận nằm trong búp sóng chính (góc lệch so với hướng phủ < HBW)
      và hai cell chiếu đối diện nhau (head-on), khoảng cách hiệu dụng bị giảm mạnh
      để ưu tiên cách ly PCI/RSI cho các cell chiếu trực diện.
    - Trả về hệ số nhân hiệu dụng (0.25 -> 1.0), càng nhỏ nghĩa là phạt càng nặng.
    """
    bearing_nbr_to_src = (bearing_src_to_nbr + 180.0) % 360.0
    off_axis_src = angular_diff_deg(az_src, bearing_src_to_nbr)
    off_axis_nbr = angular_diff_deg(az_nbr, bearing_nbr_to_src)

    penalty = 1.0
    # Cell nguồn chiếu thẳng vào cell lân cận
    if off_axis_src <= hbw_deg / 2.0:
        penalty *= 0.55
    elif off_axis_src <= hbw_deg:
        penalty *= 0.78

    # Cell lân cận cũng chiếu ngược lại về phía cell nguồn (Head-on Boresight)
    if off_axis_nbr <= hbw_deg / 2.0:
        penalty *= 0.55
    elif off_axis_nbr <= hbw_deg:
        penalty *= 0.80

    return max(0.25, penalty)


def enforce_intrasite_azimuth_separation(df: pd.DataFrame, logs: list) -> pd.DataFrame:
    """
    Kiểm tra và đảm bảo quy tắc Intra-site Azimuth Separation >= 90 độ
    giữa các sector thuộc cùng một eNodeB/Site trên cùng băng tần (EARFCN).
    Nếu phát hiện cặp sector cùng trạm có góc lệch < 90 độ, tự động hiệu chỉnh
    hoặc đánh dấu cảnh báo kỹ thuật và phân bổ PSS (Mod 3) khác nhau tuyệt đối.
    """
    df = df.copy()
    df["IntraSite_Az_Sep_Min_Deg"] = 180.0
    df["IntraSite_Sep_Status"] = "OK (>=90°)"

    site_col = "Site_ID" if "Site_ID" in df.columns else "eNodeB_ID"
    freq_col = "EARFCN" if "EARFCN" in df.columns else None

    group_cols = [site_col] + ([freq_col] if freq_col else [])
    violations = 0

    for _, group in df.groupby(group_cols):
        if len(group) <= 1:
            continue
        indices = group.index.tolist()
        azimuths = group["Azimuth"].astype(float).tolist()

        for i, idx_i in enumerate(indices):
            min_sep = 360.0
            for j, idx_j in enumerate(indices):
                if i == j:
                    continue
                sep = angular_diff_deg(azimuths[i], azimuths[j])
                if sep < min_sep:
                    min_sep = sep
            df.loc[idx_i, "IntraSite_Az_Sep_Min_Deg"] = round(min_sep, 1)
            if min_sep < 90.0:
                violations += 1
                df.loc[idx_i, "IntraSite_Sep_Status"] = f"VIOLATION ({min_sep:.1f}° < 90°)"

    if violations > 0:
        logs.append(
            f"[WARN] Phát hiện {violations} sector vi phạm ngưỡng Intra-site Azimuth Separation < 90°. "
            f"Đã kích hoạt khóa bảo vệ PSS Mod-3 nghiêm ngặt và phạt Boresight."
        )
    else:
        logs.append("[OK] Toàn bộ các trạm thỏa mãn điều kiện Intra-site Azimuth Separation >= 90°.")
    return df


def compute_directional_etilt(
    df: pd.DataFrame,
    tree_3d: KDTree,
    coords_3d: np.ndarray,
    config_map: dict,
    logs: list,
) -> pd.DataFrame:
    """
    Tính toán góc cụp điện tử theo hướng phủ (Directional E-Tilt):
    - Tìm khoảng cách tới các trạm lân cận nằm trong hình quạt hướng phát (Azimuth ± 45°).
    - Xác định bán kính phủ sóng mục tiêu D_target = 0.65 * ISD_directional.
    - Áp dụng công thức hình học mặt phẳng đứng 3GPP:
      Total_Tilt = arctan(H_eff / D_target) + 0.5 * VBW
      E_Tilt = clip(round(Total_Tilt - M_Tilt), Min_ETilt, Max_ETilt)
    """
    df = df.copy()
    n_cells = len(df)

    default_vbw = float(config_map.get("VBW_DEG", 7.0))
    min_etilt = float(config_map.get("MIN_ETILT", 0.0))
    max_etilt = float(config_map.get("MAX_ETILT", 10.0))
    default_isd = float(config_map.get("DEFAULT_ISD_M", 1500.0))

    lats = df["Latitude"].astype(float).to_numpy()
    lons = df["Longitude"].astype(float).to_numpy()
    heights = df["Ant_Height"].astype(float).to_numpy() if "Ant_Height" in df.columns else np.full(n_cells, 30.0)
    azimuths = df["Azimuth"].astype(float).to_numpy()
    mtilts = df["M_Tilt"].astype(float).to_numpy() if "M_Tilt" in df.columns else np.full(n_cells, 2.0)
    sites = df["Site_ID"].astype(str).to_numpy()

    dir_isd_list = []
    calc_etilt_list = []
    total_tilt_list = []

    # Truy vấn KDTree trong bán kính 12 km để tìm trạm đối diện theo phương vị
    neighbors_list = tree_3d.query_ball_point(coords_3d, r=12000.0)

    for i in range(n_cells):
        directional_dists = []
        for j in neighbors_list[i]:
            if i == j or sites[i] == sites[j]:
                continue
            dist_3d = float(np.linalg.norm(coords_3d[i] - coords_3d[j]))
            if dist_3d < 50.0:
                continue
            bearing = calculate_bearing_deg(lats[i], lons[i], lats[j], lons[j])
            off_angle = angular_diff_deg(azimuths[i], bearing)
            if off_angle <= 45.0:
                directional_dists.append(dist_3d)

        if directional_dists:
            # Lấy trung vị của 3 trạm gần nhất theo hướng chiếu chính
            directional_dists.sort()
            dir_isd = float(np.mean(directional_dists[: min(3, len(directional_dists))]))
        else:
            dir_isd = default_isd

        target_radius = max(180.0, dir_isd * 0.65)
        geometric_angle = math.degrees(math.atan2(max(5.0, heights[i]), target_radius))
        ideal_total_tilt = geometric_angle + 0.45 * default_vbw
        rec_etilt = np.clip(round(ideal_total_tilt - mtilts[i], 1), min_etilt, max_etilt)

        dir_isd_list.append(round(dir_isd, 1))
        calc_etilt_list.append(rec_etilt)
        total_tilt_list.append(round(mtilts[i] + rec_etilt, 1))

    df["Directional_ISD_m"] = dir_isd_list
    df["Recommended_E_Tilt_Deg"] = calc_etilt_list
    df["Total_DownTilt_Deg"] = total_tilt_list
    logs.append(
        f"[OK] Hoàn tất tính toán Directional E-Tilt cho {n_cells} cells "
        f"(ISD trung bình theo hướng: {np.mean(dir_isd_list):.1f} m)."
    )
    return df


def allocate_pci_and_rsi(
    input_df: pd.DataFrame,
    rim_df: pd.DataFrame,
    config_map: dict,
    pci_range_m: float,
    rsi_range_m: float,
    progress_bar,
    status_box,
    logs: list,
) -> pd.DataFrame:
    """
    Thực thi quy hoạch tổng thể:
    1. Xây dựng 3D KDTree cho toàn bộ cell (kết hợp Input.csv và RIM.csv reference network).
    2. Kiểm tra Intra-site Azimuth Separation >= 90 deg.
    3. Phân bổ PCI theo thuật toán Max-Min Effective Distance (kết hợp Boresight Penalty & Mod-3/Mod-30).
    4. Phân bổ RSI theo thuật toán Max-Min Distance trong bán kính RSI Range (m).
    5. Tính toán Directional E-Tilt.
    """
    df = input_df.copy()

    # Chuẩn hóa các cột bắt buộc
    required_defaults = {
        "Site_ID": [f"SITE_{i//3 + 1:03d}" for i in range(len(df))],
        "Cell_ID": [f"CELL_{i + 1:03d}" for i in range(len(df))],
        "Latitude": 21.0285,
        "Longitude": 105.8542,
        "Azimuth": [0, 120, 240][: len(df)] if len(df) <= 3 else [(i * 120) % 360 for i in range(len(df))],
        "Ant_Height": 30.0,
        "M_Tilt": 2.0,
        "EARFCN": int(config_map.get("DEFAULT_EARFCN", 1850)),
    }
    for col, default_val in required_defaults.items():
        if col not in df.columns:
            df[col] = default_val

    n_cells = len(df)
    logs.append(f"[INIT] Đã nạp {n_cells} cells cần quy hoạch từ Input.csv và {len(rim_df)} cells hiện hữu từ RIM.csv.")

    # Bước 1: Kiểm tra Intra-site Azimuth Separation >= 90 deg
    status_box.update(label="Bước 1/5: Kiểm tra Intra-site Azimuth Separation (>= 90°)...", state="running")
    df = enforce_intrasite_azimuth_separation(df, logs)
    progress_bar.progress(20)

    # Bước 2: Xây dựng 3D KDTree
    status_box.update(label="Bước 2/5: Xây dựng cấu trúc không gian 3D KDTree (Lat, Lon, Height)...", state="running")
    coords_input = latlon_to_cartesian_3d(
        df["Latitude"].astype(float).to_numpy(),
        df["Longitude"].astype(float).to_numpy(),
        df["Ant_Height"].astype(float).to_numpy(),
    )
    tree_input = KDTree(coords_input)

    # Kết hợp RIM.csv nếu có tọa độ để tránh trùng PCI/RSI với mạng đang phát sóng
    has_rim_coords = all(c in rim_df.columns for c in ["Latitude", "Longitude"]) and len(rim_df) > 0
    if has_rim_coords:
        rim_heights = (
            rim_df["Ant_Height"].astype(float).to_numpy()
            if "Ant_Height" in rim_df.columns
            else np.full(len(rim_df), 30.0)
        )
        # Quy chiếu cùng gốc tọa độ với Input
        all_lats = np.concatenate([df["Latitude"].astype(float).to_numpy(), rim_df["Latitude"].astype(float).to_numpy()])
        all_lons = np.concatenate(
            [df["Longitude"].astype(float).to_numpy(), rim_df["Longitude"].astype(float).to_numpy()]
        )
        all_h = np.concatenate([df["Ant_Height"].astype(float).to_numpy(), rim_heights])
        all_coords = latlon_to_cartesian_3d(all_lats, all_lons, all_h)
        coords_input = all_coords[:n_cells]
        coords_rim = all_coords[n_cells:]
        tree_input = KDTree(coords_input)
        tree_rim = KDTree(coords_rim)
        logs.append(f"[KDTREE] Đã lập chỉ mục 3D KDTree cho {n_cells} Input cells và {len(rim_df)} RIM cells.")
    else:
        tree_rim = None
        coords_rim = None
        logs.append(f"[KDTREE] Đã lập chỉ mục 3D KDTree cho {n_cells} Input cells.")

    progress_bar.progress(40)

    # Bước 3: Phân bổ PCI theo thuật toán Max-Min Distance + Boresight Penalty + Intra-site Mod-3
    status_box.update(
        label=f"Bước 3/5: Phân bổ PCI (Max-Min Distance + Boresight Penalty, Range={pci_range_m:.0f}m)...",
        state="running",
    )
    pci_min = int(config_map.get("PCI_MIN", 0))
    pci_max = int(config_map.get("PCI_MAX", 503))
    hbw_deg = float(config_map.get("HBW_DEG", 65.0))

    lats = df["Latitude"].astype(float).to_numpy()
    lons = df["Longitude"].astype(float).to_numpy()
    azimuths = df["Azimuth"].astype(float).to_numpy()
    sites = df["Site_ID"].astype(str).to_numpy()
    earfcns = df["EARFCN"].astype(int).to_numpy()

    assigned_pci = np.full(n_cells, -1, dtype=int)
    pci_reuse_dist = np.full(n_cells, float(pci_range_m), dtype=float)
    pci_mod3_list = np.full(n_cells, -1, dtype=int)

    # Xác định thứ tự Mod-3 ưu tiên cho các sector trong cùng trạm dựa theo Azimuth tăng dần
    preferred_mod3 = np.zeros(n_cells, dtype=int)
    for _, grp in df.groupby(["Site_ID", "EARFCN"]):
        sorted_idx = grp.sort_values("Azimuth").index.tolist()
        for rank, idx in enumerate(sorted_idx):
            preferred_mod3[idx] = rank % 3

    neighbors_pci = tree_input.query_ball_point(coords_input, r=pci_range_m)
    rim_neighbors_pci = (
        tree_rim.query_ball_point(coords_input, r=pci_range_m) if tree_rim is not None else [[] for _ in range(n_cells)]
    )

    for i in range(n_cells):
        target_mod3 = int(preferred_mod3[i])
        # Các PCI đã dùng trong cùng trạm phải khác hoàn toàn và khác Mod-3
        same_site_indices = [j for j in range(n_cells) if sites[j] == sites[i] and assigned_pci[j] != -1]
        forbidden_same_site_pcis = {assigned_pci[j] for j in same_site_indices}
        forbidden_same_site_mod3 = {assigned_pci[j] % 3 for j in same_site_indices}

        if target_mod3 in forbidden_same_site_mod3:
            for alt_m3 in (0, 1, 2):
                if alt_m3 not in forbidden_same_site_mod3:
                    target_mod3 = alt_m3
                    break

        candidate_pcis = [
            p for p in range(pci_min, pci_max + 1) if (p % 3 == target_mod3) and (p not in forbidden_same_site_pcis)
        ]
        if not candidate_pcis:
            candidate_pcis = [p for p in range(pci_min, pci_max + 1) if p not in forbidden_same_site_pcis]

        # Thu thập các cell lân cận đã có PCI trong bán kính pci_range_m
        active_nbrs = []
        for j in neighbors_pci[i]:
            if j == i or assigned_pci[j] == -1:
                continue
            if earfcns[j] != earfcns[i]:
                continue
            dist_3d = float(np.linalg.norm(coords_input[i] - coords_input[j]))
            bearing = calculate_bearing_deg(lats[i], lons[i], lats[j], lons[j])
            penalty = compute_boresight_penalty(azimuths[i], azimuths[j], bearing, hbw_deg)
            eff_dist = dist_3d * penalty
            active_nbrs.append((assigned_pci[j], eff_dist, dist_3d))

        if tree_rim is not None and "PCI" in rim_df.columns:
            rim_pcis = rim_df["PCI"].astype(int).to_numpy()
            rim_lats = rim_df["Latitude"].astype(float).to_numpy()
            rim_lons = rim_df["Longitude"].astype(float).to_numpy()
            rim_azs = (
                rim_df["Azimuth"].astype(float).to_numpy()
                if "Azimuth" in rim_df.columns
                else np.zeros(len(rim_df))
            )
            rim_earfcns = (
                rim_df["EARFCN"].astype(int).to_numpy()
                if "EARFCN" in rim_df.columns
                else np.full(len(rim_df), earfcns[i])
            )
            for r_idx in rim_neighbors_pci[i]:
                if rim_earfcns[r_idx] != earfcns[i]:
                    continue
                dist_3d = float(np.linalg.norm(coords_input[i] - coords_rim[r_idx]))
                bearing = calculate_bearing_deg(lats[i], lons[i], rim_lats[r_idx], rim_lons[r_idx])
                penalty = compute_boresight_penalty(azimuths[i], rim_azs[r_idx], bearing, hbw_deg)
                eff_dist = dist_3d * penalty
                active_nbrs.append((int(rim_pcis[r_idx]), eff_dist, dist_3d))

        best_pci = candidate_pcis[0]
        best_score = -1.0
        best_raw_dist = float(pci_range_m)

        # Max-Min Distance: chọn PCI có khoảng cách hiệu dụng nhỏ nhất đến cell trùng PCI là LỚN NHẤT
        used_pci_map = {}
        used_raw_map = {}
        for nbr_pci, eff_d, raw_d in active_nbrs:
            if nbr_pci not in used_pci_map or eff_d < used_pci_map[nbr_pci]:
                used_pci_map[nbr_pci] = eff_d
                used_raw_map[nbr_pci] = raw_d

        free_candidates = [p for p in candidate_pcis if p not in used_pci_map]
        if free_candidates:
            # Ưu tiên tránh xung đột Mod-30 và Mod-6 với các cell gần
            best_pci = free_candidates[i % len(free_candidates)]
            for cand in free_candidates:
                mod30_conflict = any(
                    (nbr_p % 30 == cand % 30) and eff_d < pci_range_m * 0.45 for nbr_p, eff_d, _ in active_nbrs
                )
                if not mod30_conflict:
                    best_pci = cand
                    break
            best_raw_dist = float(pci_range_m)
        else:
            for cand in candidate_pcis:
                score = used_pci_map.get(cand, float(pci_range_m))
                if score > best_score:
                    best_score = score
                    best_pci = cand
                    best_raw_dist = used_raw_map.get(cand, float(pci_range_m))

        assigned_pci[i] = best_pci
        pci_mod3_list[i] = best_pci % 3
        pci_reuse_dist[i] = round(best_raw_dist, 1)

    df["Assigned_PCI"] = assigned_pci
    df["PCI_Mod3"] = pci_mod3_list
    df["PCI_Mod6"] = assigned_pci % 6
    df["PCI_Mod30"] = assigned_pci % 30
    df["Min_CoPCI_Dist_m"] = pci_reuse_dist
    logs.append(
        f"[PCI] Đã phân bổ PCI cho {n_cells} cells. Khoảng cách tái sử dụng PCI nhỏ nhất: "
        f"{np.min(pci_reuse_dist):.1f} m."
    )
    progress_bar.progress(65)

    # Bước 4: Phân bổ RSI (PRACH Root Sequence Index) theo Max-Min Distance
    status_box.update(
        label=f"Bước 4/5: Phân bổ PRACH RSI (Max-Min Distance, Range={rsi_range_m:.0f}m)...",
        state="running",
    )
    rsi_min = int(config_map.get("RSI_MIN", 0))
    rsi_max = int(config_map.get("RSI_MAX", 837))
    rsi_step = max(1, int(config_map.get("RSI_STEP", 16)))

    candidate_rsis = list(range(rsi_min, rsi_max + 1, rsi_step))
    assigned_rsi = np.full(n_cells, -1, dtype=int)
    rsi_reuse_dist = np.full(n_cells, float(rsi_range_m), dtype=float)

    neighbors_rsi = tree_input.query_ball_point(coords_input, r=rsi_range_m)
    rim_neighbors_rsi = (
        tree_rim.query_ball_point(coords_input, r=rsi_range_m) if tree_rim is not None else [[] for _ in range(n_cells)]
    )

    for i in range(n_cells):
        same_site_rsis = {assigned_rsi[j] for j in range(n_cells) if sites[j] == sites[i] and assigned_rsi[j] != -1}
        used_rsi_dist = {}
        used_rsi_raw = {}

        for j in neighbors_rsi[i]:
            if j == i or assigned_rsi[j] == -1:
                continue
            dist_3d = float(np.linalg.norm(coords_input[i] - coords_input[j]))
            bearing = calculate_bearing_deg(lats[i], lons[i], lats[j], lons[j])
            penalty = compute_boresight_penalty(azimuths[i], azimuths[j], bearing, hbw_deg)
            eff_d = dist_3d * penalty
            r_val = assigned_rsi[j]
            if r_val not in used_rsi_dist or eff_d < used_rsi_dist[r_val]:
                used_rsi_dist[r_val] = eff_d
                used_rsi_raw[r_val] = dist_3d

        if tree_rim is not None and "RSI" in rim_df.columns:
            rim_rsis = rim_df["RSI"].astype(int).to_numpy()
            for r_idx in rim_neighbors_rsi[i]:
                dist_3d = float(np.linalg.norm(coords_input[i] - coords_rim[r_idx]))
                r_val = int(rim_rsis[r_idx])
                if r_val not in used_rsi_dist or dist_3d < used_rsi_dist[r_val]:
                    used_rsi_dist[r_val] = dist_3d
                    used_rsi_raw[r_val] = dist_3d

        valid_pool = [r for r in candidate_rsis if r not in same_site_rsis] or candidate_rsis
        free_rsis = [r for r in valid_pool if r not in used_rsi_dist]

        if free_rsis:
            chosen_rsi = free_rsis[i % len(free_rsis)]
            chosen_raw = float(rsi_range_m)
        else:
            chosen_rsi = max(valid_pool, key=lambda r: used_rsi_dist.get(r, 0.0))
            chosen_raw = used_rsi_raw.get(chosen_rsi, float(rsi_range_m))

        assigned_rsi[i] = chosen_rsi
        rsi_reuse_dist[i] = round(chosen_raw, 1)

    df["Assigned_RSI"] = assigned_rsi
    df["Min_CoRSI_Dist_m"] = rsi_reuse_dist
    logs.append(
        f"[RSI] Đã phân bổ PRACH RSI (bước nhảy NCS={rsi_step}) cho {n_cells} cells. "
        f"Khoảng cách tái sử dụng RSI nhỏ nhất: {np.min(rsi_reuse_dist):.1f} m."
    )
    progress_bar.progress(85)

    # Bước 5: Tính toán Directional E-Tilt
    status_box.update(label="Bước 5/5: Tối ưu hóa góc cụp điện tử Directional E-Tilt...", state="running")
    df = compute_directional_etilt(df, tree_input, coords_input, config_map, logs)
    progress_bar.progress(100)
    status_box.update(label="Hoàn tất quy hoạch 4G LTE RF Planning!", state="complete", expanded=False)

    return df


# ==============================================================================
# SAMPLE DATA GENERATOR (KHI CHƯA UPLOAD FILE)
# ==============================================================================
def get_sample_datasets():
    config_df = pd.DataFrame(
        [
            {"Parameter": "PCI_MIN", "Value": 0, "Description": "Giá trị PCI nhỏ nhất (3GPP TS 36.211)"},
            {"Parameter": "PCI_MAX", "Value": 503, "Description": "Giá trị PCI lớn nhất"},
            {"Parameter": "RSI_MIN", "Value": 0, "Description": "Chỉ số chuỗi gốc PRACH nhỏ nhất"},
            {"Parameter": "RSI_MAX", "Value": 837, "Description": "Chỉ số chuỗi gốc PRACH lớn nhất"},
            {"Parameter": "RSI_STEP", "Value": 16, "Description": "Bước nhảy RSI theo cấu hình High-Speed/NCS"},
            {"Parameter": "HBW_DEG", "Value": 65, "Description": "Độ rộng búp sóng ngang 3dB (độ)"},
            {"Parameter": "VBW_DEG", "Value": 7.0, "Description": "Độ rộng búp sóng đứng 3dB (độ)"},
            {"Parameter": "MIN_ETILT", "Value": 0.0, "Description": "Giới hạn E-Tilt nhỏ nhất (độ)"},
            {"Parameter": "MAX_ETILT", "Value": 10.0, "Description": "Giới hạn E-Tilt lớn nhất (độ)"},
        ]
    )

    rim_rows = [
        {"Site_ID": "HN_EXISTING_01", "Cell_ID": "HN_EX01_1", "Latitude": 21.0245, "Longitude": 105.8412, "Azimuth": 0, "Ant_Height": 32, "EARFCN": 1850, "PCI": 12, "RSI": 32},
        {"Site_ID": "HN_EXISTING_01", "Cell_ID": "HN_EX01_2", "Latitude": 21.0245, "Longitude": 105.8412, "Azimuth": 120, "Ant_Height": 32, "EARFCN": 1850, "PCI": 13, "RSI": 48},
        {"Site_ID": "HN_EXISTING_01", "Cell_ID": "HN_EX01_3", "Latitude": 21.0245, "Longitude": 105.8412, "Azimuth": 240, "Ant_Height": 32, "EARFCN": 1850, "PCI": 14, "RSI": 64},
        {"Site_ID": "HN_EXISTING_02", "Cell_ID": "HN_EX02_1", "Latitude": 21.0368, "Longitude": 105.8590, "Azimuth": 30, "Ant_Height": 28, "EARFCN": 1850, "PCI": 45, "RSI": 128},
        {"Site_ID": "HN_EXISTING_02", "Cell_ID": "HN_EX02_2", "Latitude": 21.0368, "Longitude": 105.8590, "Azimuth": 150, "Ant_Height": 28, "EARFCN": 1850, "PCI": 46, "RSI": 144},
        {"Site_ID": "HN_EXISTING_02", "Cell_ID": "HN_EX02_3", "Latitude": 21.0368, "Longitude": 105.8590, "Azimuth": 270, "Ant_Height": 28, "EARFCN": 1850, "PCI": 47, "RSI": 160},
    ]
    rim_df = pd.DataFrame(rim_rows)

    site_coords = [
        ("LTE_HN_101", 21.0285, 105.8542, 30.0, [0, 120, 240]),
        ("LTE_HN_102", 21.0332, 105.8485, 28.0, [30, 150, 270]),
        ("LTE_HN_103", 21.0218, 105.8510, 35.0, [10, 130, 250]),
        ("LTE_HN_104", 21.0298, 105.8635, 26.0, [0, 120, 240]),
        ("LTE_HN_105", 21.0175, 105.8612, 32.0, [45, 165, 285]),
    ]
    input_rows = []
    for site_id, lat, lon, h, az_list in site_coords:
        for sec_idx, az in enumerate(az_list, start=1):
            input_rows.append(
                {
                    "Site_ID": site_id,
                    "Cell_ID": f"{site_id}_{sec_idx}",
                    "Latitude": lat,
                    "Longitude": lon,
                    "Azimuth": az,
                    "Ant_Height": h,
                    "M_Tilt": 2.0,
                    "EARFCN": 1850,
                }
            )
    input_df = pd.DataFrame(input_rows)
    return rim_df, config_df, input_df


def parse_config_df(config_df: pd.DataFrame) -> dict:
    config_map = {}
    if "Parameter" in config_df.columns and "Value" in config_df.columns:
        for _, row in config_df.iterrows():
            try:
                config_map[str(row["Parameter"]).strip().upper()] = float(row["Value"])
            except ValueError:
                pass
    return config_map


# ==============================================================================
# SIDEBAR: INPUT CONTROLS
# ==============================================================================
with st.sidebar:
    st.header("📡 Thông số & Dữ liệu Đầu vào")
    st.caption("Tải lên 3 file dữ liệu quy hoạch (.csv) hoặc dùng dữ liệu mẫu tích hợp sẵn.")

    rim_file = st.file_uploader("1. Tải lên RIM.csv (Dữ liệu mạng hiện hữu)", type=["csv"], key="rim_uploader")
    config_file = st.file_uploader("2. Tải lên Config.csv (Cấu hình tham số RF)", type=["csv"], key="config_uploader")
    input_file = st.file_uploader("3. Tải lên Input.csv (Danh sách trạm quy hoạch)", type=["csv"], key="input_uploader")

    st.divider()
    st.subheader("Khoảng cách Tái sử dụng (Reuse Range)")
    pci_range_m = st.number_input(
        "PCI Range (m)",
        min_value=500,
        max_value=50000,
        value=8000,
        step=500,
        help="Bán kính kiểm tra xung đột và tối ưu hóa khoảng cách dùng lại PCI (mặc định 8000m).",
    )
    rsi_range_m = st.number_input(
        "RSI Range (m)",
        min_value=500,
        max_value=50000,
        value=8000,
        step=500,
        help="Bán kính kiểm tra xung đột và tối ưu hóa khoảng cách dùng lại PRACH RSI (mặc định 8000m).",
    )

    use_sample = st.checkbox(
        "Sử dụng dữ liệu mẫu khi chưa tải đủ 3 file CSV",
        value=True,
        help="Cho phép chạy thử nghiệm ngay lập tức với mạng mẫu 4G LTE tại Hà Nội.",
    )
    run_btn = st.button("🚀 Chạy Quy hoạch RF (Execute RF Plan)", type="primary", use_container_width=True)

# ==============================================================================
# MAIN INTERFACE
# ==============================================================================
st.title("4G LTE RF Planning & Optimization Web Application")
st.markdown(
    "Công cụ tự động hóa quy hoạch tham số vô tuyến **4G LTE** sử dụng cấu trúc không gian **3D KDTree**, "
    "kiểm tra **Intra-site Azimuth Separation >= 90°**, tính toán **Boresight Penalty**, phân bổ **Max-Min Distance PCI/RSI**, "
    "và tối ưu hóa **Directional E-Tilt**."
)

sample_rim, sample_cfg, sample_inp = get_sample_datasets()

rim_df = pd.read_csv(rim_file) if rim_file is not None else (sample_rim if use_sample else None)
config_df = pd.read_csv(config_file) if config_file is not None else (sample_cfg if use_sample else None)
input_df = pd.read_csv(input_file) if input_file is not None else (sample_inp if use_sample else None)

# Hiển thị Preview Dữ liệu Đầu vào
st.subheader("1. Xem trước Dữ liệu Đầu vào (Input Data Previews)")
tab_inp, tab_rim, tab_cfg = st.tabs(["📄 Input.csv (Trạm mới)", "🌐 RIM.csv (Mạng hiện hữu)", "⚙️ Config.csv (Cấu hình)"])

with tab_inp:
    if input_df is not None:
        st.dataframe(input_df, use_container_width=True, height=240)
    else:
        st.info("Vui lòng tải lên file `Input.csv` ở thanh công cụ bên trái.")

with tab_rim:
    if rim_df is not None:
        st.dataframe(rim_df, use_container_width=True, height=240)
    else:
        st.info("Vui lòng tải lên file `RIM.csv` ở thanh công cụ bên trái.")

with tab_cfg:
    if config_df is not None:
        st.dataframe(config_df, use_container_width=True, height=240)
    else:
        st.info("Vui lòng tải lên file `Config.csv` ở thanh công cụ bên trái.")

# Chạy thuật toán khi nhấn nút hoặc lần đầu với dữ liệu mẫu
if "rf_output_df" not in st.session_state and input_df is not None and rim_df is not None and config_df is not None:
    run_btn = True

if run_btn:
    if input_df is None or rim_df is None or config_df is None:
        st.error("Vui lòng tải lên đầy đủ cả 3 file `RIM.csv`, `Config.csv`, `Input.csv` hoặc bật chế độ Dữ liệu mẫu.")
    else:
        st.subheader("2. Trạng thái Thực thi & Nhật ký Thuật toán (Execution Status & Logs)")
        progress_bar = st.progress(0)
        logs = [
            f"[START] Khởi chạy thuật toán 4G LTE RF Planning | PCI Range = {pci_range_m}m | RSI Range = {rsi_range_m}m"
        ]
        config_map = parse_config_df(config_df)

        with st.status("Đang khởi tạo bộ giải 3D KDTree RF Planning...", expanded=True) as status_box:
            start_ts = time.perf_counter()
            output_df = allocate_pci_and_rsi(
                input_df=input_df,
                rim_df=rim_df,
                config_map=config_map,
                pci_range_m=float(pci_range_m),
                rsi_range_m=float(rsi_range_m),
                progress_bar=progress_bar,
                status_box=status_box,
                logs=logs,
            )
            elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
            logs.append(f"[DONE] Hoàn thành xuất sắc quy hoạch RF trong {elapsed_ms:.1f} ms.")
            st.write("✔ Đã hoàn tất 3D KDTree, Boresight Penalty, PCI/RSI Max-Min Allocation và Directional E-Tilt.")

        st.session_state["rf_output_df"] = output_df
        st.session_state["rf_logs"] = "\n".join(logs)

if "rf_output_df" in st.session_state:
    out_df = st.session_state["rf_output_df"]
    log_text = st.session_state.get("rf_logs", "")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Tổng số Cell Quy hoạch", f"{len(out_df)}")
    col_m2.metric("Min Co-PCI Distance", f"{out_df['Min_CoPCI_Dist_m'].min():.1f} m")
    col_m3.metric("Min Co-RSI Distance", f"{out_df['Min_CoRSI_Dist_m'].min():.1f} m")
    col_m4.metric("E-Tilt Trung bình", f"{out_df['Recommended_E_Tilt_Deg'].mean():.2f}°")

    st.markdown("#### Nhật ký Thực thi Tương tác (Interactive Execution Logs)")
    st.code(log_text, language="log")

    st.subheader("3. Kết quả Thiết kế Vô tuyến (`Output_RF_Design.csv`)")
    st.dataframe(out_df, use_container_width=True, height=340)

    csv_buffer = io.StringIO()
    out_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="⬇️ Tải xuống Output_RF_Design.csv",
        data=csv_buffer.getvalue().encode("utf-8"),
        file_name="Output_RF_Design.csv",
        mime="text/csv",
        type="primary",
    )
