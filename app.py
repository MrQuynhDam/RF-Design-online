import io
import time
import math
import numpy as np
import pandas as pd
import streamlit as st
from scipy.spatial import KDTree

# ==========================================
# 1. CẤU HÌNH TRANG (ULTRA-COMPACT)
# ==========================================
st.set_page_config(
    page_title="LTE RF Design Tool",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS ép toàn bộ giao diện siêu nhỏ gọn
st.markdown(
    "",
    unsafe_allow_html=True
)

# Header siêu nhỏ
st.markdown("#### 📡 LTE RF NETWORK DESIGN AUTOMATION TOOL")
st.caption("Ericsson RAN Systems • Automatic Allocation for TAC, PCI, RSI, Azimuth, M-Tilt & Directional E-Tilt")

# ==========================================
# 2. RF CORE CALCULATIONS & UTILS
# ==========================================

def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return c * 6367000.0

def latlon_to_cartesian(lat, lon):
    R = 6371000.0
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    x = R * np.cos(lat_rad) * np.cos(lon_rad)
    y = R * np.cos(lat_rad) * np.sin(lon_rad)
    z = R * np.sin(lat_rad)
    return np.column_stack((x, y, z))

def calculate_optimum_azimuth(site_lat, site_lon, neighbor_lats, neighbor_lons, neighbor_azimuths, sector_idx, assigned_site_azimuths=[]):
    default_azimuths = [0, 120, 240]
    best_azimuth = default_azimuths[sector_idx]
    max_score = -1e9

    if sector_idx == 0:
        candidates = list(range(300, 360, 5)) + list(range(0, 65, 5))
    elif sector_idx == 1:
        candidates = list(range(60, 185, 5))
    else:
        candidates = list(range(180, 305, 5))

    valid_candidates = []
    for az in candidates:
        valid = True
        for prev_az in assigned_site_azimuths:
            diff = np.abs((az - prev_az + 180) % 360 - 180)
            if diff < 90:
                valid = False
                break
        if valid:
            valid_candidates.append(az)

    if len(valid_candidates) == 0:
        valid_candidates = candidates

    if len(neighbor_lats) == 0:
        return min(valid_candidates, key=lambda x: np.abs((x - best_azimuth + 180) % 360 - 180))

    dlat = np.radians(neighbor_lats - site_lat)
    dlon = np.radians(neighbor_lons - site_lon)
    y = np.sin(dlon) * np.cos(np.radians(neighbor_lats))
    x = np.cos(np.radians(site_lat)) * np.sin(np.radians(neighbor_lats)) - \
        np.sin(np.radians(site_lat)) * np.cos(np.radians(neighbor_lats)) * np.cos(dlon)
    bearings_to_neighbors = (np.degrees(np.arctan2(y, x)) + 360) % 360

    for az in valid_candidates:
        angle_diff1 = np.abs((az - bearings_to_neighbors + 180) % 360 - 180)
        neighbor_boresight = (bearings_to_neighbors + 180) % 360
        angle_diff2 = np.abs((neighbor_azimuths - neighbor_boresight + 180) % 360 - 180)
        
        penalty = np.sum(np.exp(-((angle_diff1**2 + angle_diff2**2) / (2 * 30**2))))
        score = -penalty

        if score > max_score:
            max_score = score
            best_azimuth = az

    return best_azimuth

def get_directional_nearest_distance(site_lat, site_lon, cell_azimuth, neighbor_lats, neighbor_lons, default_dist=1500.0):
    if len(neighbor_lats) == 0:
        return default_dist

    dlat = np.radians(neighbor_lats - site_lat)
    dlon = np.radians(neighbor_lons - site_lon)
    y = np.sin(dlon) * np.cos(np.radians(neighbor_lats))
    x = np.cos(np.radians(site_lat)) * np.sin(np.radians(neighbor_lats)) - \
        np.sin(np.radians(site_lat)) * np.cos(np.radians(neighbor_lats)) * np.cos(dlon)
    bearings = (np.degrees(np.arctan2(y, x)) + 360) % 360

    angle_diffs = np.abs((bearings - cell_azimuth + 180) % 360 - 180)
    in_cone_mask = angle_diffs <= 45

    if not np.any(in_cone_mask):
        return default_dist

    dists = haversine_np(site_lon, site_lat, neighbor_lons[in_cone_mask], neighbor_lats[in_cone_mask])
    return max(np.min(dists), 100.0)

# ==========================================
# 3. GIAO DIỆN HÀNG NGANG TỐI ƯU (1 SINGLE ROW)
# ==========================================

# Gộp toàn bộ File Uploaders + Parameters + Execute Button vào chung 1 hàng (6 Cột)
c1, c2, c3, c4, c5, c6 = st.columns([1.2, 1.2, 1.2, 1, 1, 1.2])

with c1:
    rim_file = st.file_uploader("1. RIM.csv", type=["csv"], key="rim")
with c2:
    config_file = st.file_uploader("2. Config.csv", type=["csv"], key="config")
with c3:
    input_file = st.file_uploader("3. Input.csv", type=["csv"], key="input")
with c4:
    pci_min_dist = st.number_input("PCI Range (m)", min_value=1000, value=8000, step=500)
with c5:
    rsi_min_dist = st.number_input("RSI Range (m)", min_value=1000, value=8000, step=500)
with c6:
    execute_btn = st.button("🚀 Run Design", type="primary", use_container_width=True)

# ==========================================
# 4. PROCESSING LOGIC & DASHBOARD
# ==========================================

if execute_btn:
    if not rim_file or not config_file or not input_file:
        st.error("⚠️ Vui lòng nạp đủ 3 file CSV đầu vào!")
    else:
        start_time = time.time()
        logs = []
        
        def add_log(msg):
            timestamp = time.strftime("[%H:%M:%S] ")
            logs.append(timestamp + msg)

        status_box = st.status("⚙️ Đang thực thi quy hoạch...", expanded=True)
        progress_bar = st.progress(0)

        try:
            status_box.write("Đang đọc file dữ liệu...")
            df_rim = pd.read_csv(rim_file)
            df_config = pd.read_csv(config_file)
            df_input = pd.read_csv(input_file)

            df_rim.columns = df_rim.columns.str.strip()
            df_config.columns = df_config.columns.str.strip()
            df_input.columns = df_input.columns.str.strip()

            add_log(f"Đọc dữ liệu thành công: RIM ({len(df_rim)} dòng), Config ({len(df_config)} dòng), Input ({len(df_input)} dòng).")
            progress_bar.progress(10)

            df_existing = pd.merge(df_rim, df_config[['Cellname', 'TAC', 'PCI', 'RSI']], on='Cellname', how='inner')
            add_log(f"Tổng hợp thành công {len(df_existing)} cell hiện hữu.")
            progress_bar.progress(20)

            existing_coords_cart = latlon_to_cartesian(df_existing['Lat'].values, df_existing['Lon'].values)
            kdtree_existing = KDTree(existing_coords_cart)

            assigned_pci_list = np.column_stack((existing_coords_cart, df_existing['PCI'].values))
            assigned_rsi_list = np.column_stack((existing_coords_cart, df_existing['RSI'].values))

            pci_groups = [list(range(i, i+3)) for i in range(0, 448, 3)]
            rsi_groups = [[r, (r+6)%643, (r+12)%643] for r in range(0, 643-12, 6)]

            unique_sites = df_input['Sitename'].unique()
            total_sites = len(unique_sites)
            add_log(f"Bắt đầu quy hoạch cho {total_sites} site mới...")

            output_rows = []

            for idx, site_name in enumerate(unique_sites):
                status_box.write(f"Đang tính toán site {idx+1}/{total_sites}: {site_name}")
                site_cells = df_input[df_input['Sitename'] == site_name].copy()
                site_lat = site_cells['Lat'].iloc[0]
                site_lon = site_cells['Lon'].iloc[0]
                site_cart = latlon_to_cartesian(site_lat, site_lon)[0]

                _, nearest_idx = kdtree_existing.query(site_cart)
                assigned_tac = df_existing.iloc[nearest_idx]['TAC']

                nearest_site_dist = haversine_np(
                    site_lon, site_lat, 
                    df_existing.iloc[nearest_idx]['Lon'], df_existing.iloc[nearest_idx]['Lat']
                )
                nearest_site_dist = max(nearest_site_dist, 100.0)

                neighbor_indices = kdtree_existing.query_ball_point(site_cart, r=5000)
                if len(neighbor_indices) > 0:
                    n_lats = df_existing.iloc[neighbor_indices]['Lat'].values
                    n_lons = df_existing.iloc[neighbor_indices]['Lon'].values
                    n_azs = df_existing.iloc[neighbor_indices]['Azimuth'].values
                else:
                    n_lats, n_lons, n_azs = np.array([]), np.array([]), np.array([])

                selected_pci_group = None
                max_min_pci_dist = -1
                best_fallback_pci_group = pci_groups[0]

                for group in pci_groups:
                    min_dist_for_this_group = 1e9
                    conflict = False
                    for pci_val in group:
                        matched_pcis = assigned_pci_list[assigned_pci_list[:, 3] == pci_val]
                        if len(matched_pcis) > 0:
                            dists = haversine_np(
                                site_lon, site_lat, 
                                np.degrees(np.arctan2(matched_pcis[:,1], matched_pcis[:,0])), 
                                np.degrees(np.arcsin(matched_pcis[:,2]/6371000.0))
                            )
                            current_min_d = np.min(dists)
                            if current_min_d < min_dist_for_this_group:
                                min_dist_for_this_group = current_min_d
                            if current_min_d < pci_min_dist:
                                conflict = True
                        else:
                            min_dist_for_this_group = 1e9

                    if min_dist_for_this_group > max_min_pci_dist:
                        max_min_pci_dist = min_dist_for_this_group
                        best_fallback_pci_group = group

                    if not conflict:
                        selected_pci_group = group
                        break

                if selected_pci_group is None:
                    selected_pci_group = best_fallback_pci_group
                    add_log(f"[WARNING] Site {site_name}: Hết PCI đạt chuẩn {pci_min_dist}m! Đã chọn nhóm tốt nhất d_min = {int(max_min_pci_dist)}m")

                selected_rsi_group = None
                max_min_rsi_dist = -1
                best_fallback_rsi_group = rsi_groups[0]

                for group in rsi_groups:
                    min_dist_for_this_group = 1e9
                    conflict = False
                    for rsi_val in group:
                        matched_rsis = assigned_rsi_list[assigned_rsi_list[:, 3] == rsi_val]
                        if len(matched_rsis) > 0:
                            dists = haversine_np(
                                site_lon, site_lat, 
                                np.degrees(np.arctan2(matched_rsis[:,1], matched_rsis[:,0])), 
                                np.degrees(np.arcsin(matched_rsis[:,2]/6371000.0))
                            )
                            current_min_d = np.min(dists)
                            if current_min_d < min_dist_for_this_group:
                                min_dist_for_this_group = current_min_d
                            if current_min_d < rsi_min_dist:
                                conflict = True
                        else:
                            min_dist_for_this_group = 1e9

                    if min_dist_for_this_group > max_min_rsi_dist:
                        max_min_rsi_dist = min_dist_for_this_group
                        best_fallback_rsi_group = group

                    if not conflict:
                        selected_rsi_group = group
                        break

                if selected_rsi_group is None:
                    selected_rsi_group = best_fallback_rsi_group
                    add_log(f"[WARNING] Site {site_name}: Hết RSI đạt chuẩn {rsi_min_dist}m! Đã chọn nhóm tốt nhất d_min = {int(max_min_rsi_dist)}m")

                site_assigned_azs = []
                for cell_idx in range(min(3, len(site_cells))):
                    cell_row = site_cells.iloc[cell_idx].to_dict()

                    opt_azimuth = calculate_optimum_azimuth(
                        site_lat, site_lon, n_lats, n_lons, n_azs, 
                        sector_idx=cell_idx,
                        assigned_site_azimuths=site_assigned_azs
                    )
                    site_assigned_azs.append(opt_azimuth)

                    m_tilt = 2.0
                    ant_height = float(cell_row.get('Height', 30.0))

                    cell_directional_dist = get_directional_nearest_distance(
                        site_lat, site_lon, opt_azimuth, n_lats, n_lons, default_dist=nearest_site_dist
                    )
                    
                    d_coverage = (2.0 / 3.0) * cell_directional_dist
                    total_tilt = math.degrees(math.atan(ant_height / d_coverage))
                    e_tilt = max(0, int(round(total_tilt - m_tilt)))

                    cell_row['TAC'] = int(assigned_tac)
                    cell_row['PCI'] = int(selected_pci_group[cell_idx])
                    cell_row['RSI'] = int(selected_rsi_group[cell_idx])
                    cell_row['Azimuth'] = int(opt_azimuth)
                    cell_row['M-Tilt'] = int(m_tilt)
                    cell_row['E-Tilt'] = int(e_tilt)

                    output_rows.append(cell_row)

                    assigned_pci_list = np.vstack([assigned_pci_list, [*site_cart, cell_row['PCI']]])
                    assigned_rsi_list = np.vstack([assigned_rsi_list, [*site_cart, cell_row['RSI']]])

                progress = 20 + int(((idx + 1) / total_sites) * 70)
                progress_bar.progress(progress)

            df_output = pd.DataFrame(output_rows)
            progress_bar.progress(100)
            elapsed_time = round(time.time() - start_time, 2)
            
            add_log("="*50)
            add_log(f"THÀNH CÔNG: Hoàn thành quy hoạch cho {len(output_rows)} cells ({total_sites} sites) trong {elapsed_time}s.")

            status_box.update(label="✅ Hoàn tất quy hoạch!", state="complete", expanded=False)
            st.session_state["output_df"] = df_output
            st.session_state["logs"] = "\n".join(logs)
            st.session_state["exec_time"] = elapsed_time

        except Exception as e:
            status_box.update(label="❌ Lỗi tính toán!", state="error")
            st.error(f"Lỗi: {str(e)}")

# HIỂN THỊ KẾT QUẢ VỚI CHIỀU CAO THU NHỎ
if "output_df" in st.session_state:
    df_out = st.session_state["output_df"]
    exec_t = st.session_state.get("exec_time", 0)
    
    # 4 thẻ chỉ số nhanh gọn
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Site Mới", f"{df_out['Sitename'].nunique()}")
    m2.metric("Tổng Cell", f"{len(df_out)}")
    m3.metric("E-Tilt TB", f"{df_out['E-Tilt'].mean():.1f}°")
    m4.metric("Thời Gian", f"{exec_t}s")

    tab_data, tab_log = st.tabs(["📋 Kết Quả (Output Data)", "📜 Nhật Ký (Logs)"])

    with tab_data:
        st.dataframe(df_out, use_container_width=True, height=180)
        csv_buffer = io.StringIO()
        df_out.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Output_RF_Design.csv",
            data=csv_buffer.getvalue().encode('utf-8-sig'),
            file_name="Output_RF_Design.csv",
            mime="text/csv",
            type="primary"
        )

    with tab_log:
        st.code(st.session_state.get("logs", ""), language="text")
