import io
import os
import time
import math
import requests
import numpy as np
import pandas as pd
import streamlit as st
from scipy.spatial import KDTree

# ==========================================
# 1. CẤU HÌNH TRANG & GIAO DIỆN CHUYÊN NGHIỆP
# ==========================================
st.set_page_config(
    page_title="LTE RF Design Tool",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS cho layout chuyên nghiệp
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    
    .custom-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid #e9ecef;
        margin-bottom: 20px;
    }
    
    div[data-testid="stFileUploaderDropzoneInstructions"] > * {
        display: none !important;
    }
    div[data-testid="stFileUploaderDropzoneInstructions"]::after {
        content: "Kéo thả hoặc chọn file CSV (Max 10MB)";
        font-size: 13px;
        color: #6c757d;
    }
    
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #4d648a;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    div.stButton > button[kind="primary"] {
        background-color: #2563eb;
        border-color: #2563eb;
        font-weight: 600;
        border-radius: 6px;
        height: 46px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📡 LTE RF DESIGN AUTOMATION TOOL")
st.caption("Ericsson RAN Systems • Automatic Allocation for TAC, PCI (Best-Fit Range 0-449), RSI, Azimuth, M-Tilt & Directional E-Tilt")
st.markdown("---")

# ==========================================
# 2. THANH BÊN (SIDEBAR) - CẤU HÌNH THAM SỐ
# ==========================================
with st.sidebar:
    st.header("⚙️ Cấu Hình Tham Số")
    st.markdown("Thiết lập khoảng cách an toàn cho thuật toán phân bổ Best-Fit:")
    
    pci_min_dist = st.number_input("PCI Min Range (m)", min_value=1000, value=8000, step=500, help="Khoảng cách tối thiểu tái sử dụng PCI")
    rsi_min_dist = st.number_input("RSI Min Range (m)", min_value=1000, value=8000, step=500, help="Khoảng cách tối thiểu tái sử dụng RSI")
    
    st.markdown("---")
    st.markdown("##### 🛡️ Ràng buộc Modulo")
    mod3_factor = st.slider("Bảo vệ Mod3 (% PCI Range)", min_value=10, max_value=100, value=40, step=5) / 100.0
    mod6_factor = st.slider("Bảo vệ Mod6 (% PCI Range)", min_value=10, max_value=100, value=25, step=5) / 100.0
    
    st.markdown("---")
    st.caption("Developed for Ericsson RAN RF Planning Automation")

# ==========================================
# 3. TẢI FILE MẪU & INPUT DATA
# ==========================================
GITHUB_USER = "MrQuynhDam"
GITHUB_REPO = "YOUR_REPO_NAME"

@st.cache_data
def get_sample_file_bytes(filename):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return f.read()
            
    possible_names = [filename, filename.replace("_Sample", "S_Sample")]
    for p_name in possible_names:
        if os.path.exists(p_name):
            with open(p_name, "rb") as f:
                return f.read()

    url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/main/{filename}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.content
    except Exception:
        pass

    return None

col_left, col_right = st.columns([1, 2], gap="medium")

with col_left:
    st.markdown('<div class="section-title">📥 1. Download Sample Files</div>', unsafe_allow_html=True)
        
    sample_files = {
        "RIMS_Sample.csv": "File thông tin Trạm RIM hiện hữu",
        "Config_Sample.csv": "File cấu hình Cell hiện hữu (TAC/PCI/RSI)",
        "Input_Sample.csv": "File danh sách Site mới cần quy hoạch"
    }

    for fname, fdesc in sample_files.items():
        file_bytes = get_sample_file_bytes(fname)
        if file_bytes:
            st.download_button(
                label="📄 " + fname,
                data=file_bytes,
                file_name=fname,
                mime="text/csv",
                use_container_width=True,
                help=fdesc
            )
        else:
            st.button(f"❌ Không tìm thấy {fname}", disabled=True, use_container_width=True)

with col_right:
    st.markdown('<div class="section-title">📤 2. Upload input files</div>', unsafe_allow_html=True)
    
    u1, u2, u3 = st.columns(3)
    with u1:
        rim_file = st.file_uploader("1. RIMS.csv", type=["csv"], key="rim")
    with u2:
        config_file = st.file_uploader("2. Config.csv", type=["csv"], key="config")
    with u3:
        input_file = st.file_uploader("3. Input.csv", type=["csv"], key="input")

st.markdown("---")
col_btn, _ = st.columns([1, 2])
with col_btn:
    execute_btn = st.button("🚀 BẮT ĐẦU QUY HOẠCH RF", type="primary", use_container_width=True)

# ==========================================
# 4. RF CORE CALCULATIONS & UTILS
# ==========================================

def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return c * 6371000.0

def latlon_to_cartesian(lat, lon):
    R = 6371000.0
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    x = R * np.cos(lat_rad) * np.cos(lon_rad)
    y = R * np.cos(lat_rad) * np.sin(lon_rad)
    z = R * np.sin(lat_rad)
    return np.column_stack((x, y, z))

def calculate_optimum_azimuth(site_lat, site_lon, neighbor_lats, neighbor_lons, neighbor_azimuths, sector_idx, total_sectors=3, assigned_site_azimuths=[]):
    base_angle = 360.0 / total_sectors
    default_azimuth = int((sector_idx * base_angle) % 360)
    best_azimuth = default_azimuth
    max_score = -1e9

    start_angle = int((default_azimuth - 30) % 360)
    end_angle = int((default_azimuth + 35) % 360)
    
    if start_angle < end_angle:
        candidates = list(range(start_angle, end_angle, 5))
    else:
        candidates = list(range(start_angle, 360, 5)) + list(range(0, end_angle, 5))

    valid_candidates = []
    for az in candidates:
        valid = True
        for prev_az in assigned_site_azimuths:
            diff = np.abs((az - prev_az + 180) % 360 - 180)
            if diff < (360 / total_sectors) * 0.6:
                valid = False
                break
        if valid:
            valid_candidates.append(az)

    if len(valid_candidates) == 0:
        valid_candidates = candidates if len(candidates) > 0 else [default_azimuth]

    if len(neighbor_lats) == 0:
        return min(valid_candidates, key=lambda x: np.abs((x - default_azimuth + 180) % 360 - 180))

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

def check_pci_group_validity(candidate_group, site_lon, site_lat, assigned_pci_list, pci_min_dist, mod3_min_dist, mod6_min_dist):
    if len(assigned_pci_list) == 0:
        return True, 1e9

    min_pci_dist = 1e9
    assigned_lons = np.degrees(np.arctan2(assigned_pci_list[:, 1], assigned_pci_list[:, 0]))
    assigned_lats = np.degrees(np.arcsin(np.clip(assigned_pci_list[:, 2] / 6371000.0, -1.0, 1.0)))
    assigned_pcis = assigned_pci_list[:, 3].astype(int)

    dists = haversine_np(site_lon, site_lat, assigned_lons, assigned_lats)

    for pci_candidate in candidate_group:
        cand_mod3 = pci_candidate % 3
        cand_mod6 = pci_candidate % 6

        same_pci_mask = (assigned_pcis == pci_candidate)
        if np.any(same_pci_mask):
            d = np.min(dists[same_pci_mask])
            if d < min_pci_dist:
                min_pci_dist = d
            if d < pci_min_dist:
                return False, min_pci_dist

        same_mod3_mask = ((assigned_pcis % 3) == cand_mod3)
        if np.any(same_mod3_mask):
            d_mod3 = np.min(dists[same_mod3_mask])
            if d_mod3 < mod3_min_dist:
                return False, min_pci_dist

        same_mod6_mask = ((assigned_pcis % 6) == cand_mod6)
        if np.any(same_mod6_mask):
            d_mod6 = np.min(dists[same_mod6_mask])
            if d_mod6 < mod6_min_dist:
                return False, min_pci_dist

    return True, min_pci_dist

# ==========================================
# 5. XỬ LÝ QUY HOẠCH & XUẤT KẾT QUẢ
# ==========================================

if execute_btn:
    if not rim_file or not config_file or not input_file:
        st.error("⚠️ Vui lòng tải đủ 3 file CSV đầu vào (hoặc chọn dùng file mẫu)!")
    else:
        start_time = time.time()
        logs = []
        
        def add_log(msg):
            timestamp = time.strftime("[%H:%M:%S] ")
            logs.append(timestamp + msg)

        status_box = st.status("⚙️ Đang tiến hành phân bổ tham số RF...", expanded=True)
        progress_bar = st.progress(0)

        try:
            status_box.write("Đang tải dữ liệu...")
            df_rim = pd.read_csv(rim_file)
            df_config = pd.read_csv(config_file)
            df_input = pd.read_csv(input_file)

            df_rim.columns = df_rim.columns.str.strip()
            df_config.columns = df_config.columns.str.strip()
            df_input.columns = df_input.columns.str.strip()

            add_log(f"Đọc thành công: RIM ({len(df_rim)} dòng), Config ({len(df_config)} dòng), Input ({len(df_input)} dòng).")
            progress_bar.progress(10)

            df_existing = pd.merge(df_rim, df_config[['Cellname', 'TAC', 'PCI', 'RSI']], on='Cellname', how='inner')
            add_log(f"Tổng hợp {len(df_existing)} cell mạng hiện hữu.")
            progress_bar.progress(20)

            existing_coords_cart = latlon_to_cartesian(df_existing['Lat'].values, df_existing['Lon'].values)
            kdtree_existing = KDTree(existing_coords_cart)

            assigned_pci_list = np.column_stack((existing_coords_cart, df_existing['PCI'].values))
            assigned_rsi_list = np.column_stack((existing_coords_cart, df_existing['RSI'].values))

            # Giới hạn dải PCI từ 0 đến 449 (450 giá trị -> 150 nhóm 3)
            pci_groups = [list(range(i, i+3)) for i in range(0, 450, 3)]
            rsi_groups = [[r, (r+6)%643, (r+12)%643] for r in range(0, 643-12, 6)]

            unique_sites = df_input['Sitename'].unique()
            total_sites = len(unique_sites)
            add_log(f"Bắt đầu quy hoạch cho {total_sites} site mới (PCI Best-Fit trong dải 0-449)...")

            output_rows = []

            for idx, site_name in enumerate(unique_sites):
                status_box.write(f"Đang xử lý site [{idx+1}/{total_sites}]: {site_name}")
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

                # ========================================================
                # PHÂN BỔ PCI - STRATEGY: BEST-FIT (RANGE 0 - 449)
                # ========================================================
                selected_pci_group = None
                max_valid_dist = -1

                best_fallback_pci_group = pci_groups[0]
                max_fallback_dist = -1

                mod3_dist_req = min(3000.0, pci_min_dist * mod3_factor)
                mod6_dist_req = min(2000.0, pci_min_dist * mod6_factor)

                for group in pci_groups:
                    is_valid, min_d = check_pci_group_validity(
                        group, site_lon, site_lat, assigned_pci_list, 
                        pci_min_dist=pci_min_dist, 
                        mod3_min_dist=mod3_dist_req, 
                        mod6_min_dist=mod6_dist_req
                    )

                    if is_valid:
                        if min_d > max_valid_dist:
                            max_valid_dist = min_d
                            selected_pci_group = group
                    else:
                        if min_d > max_fallback_dist:
                            max_fallback_dist = min_d
                            best_fallback_pci_group = group

                if selected_pci_group is None:
                    selected_pci_group = best_fallback_pci_group
                    add_log(f"[CẢNH BÁO] Site {site_name}: Chọn nhóm PCI dự phòng tốt nhất (d_min = {int(max_fallback_dist)}m)")

                # ========================================================
                # PHÂN BỔ RSI - STRATEGY: BEST-FIT
                # ========================================================
                selected_rsi_group = None
                max_valid_rsi_dist = -1

                best_fallback_rsi_group = rsi_groups[0]
                max_fallback_rsi_dist = -1

                for group in rsi_groups:
                    min_dist_for_this_group = 1e9
                    conflict = False

                    for rsi_val in group:
                        matched_rsis = assigned_rsi_list[assigned_rsi_list[:, 3] == rsi_val]
                        if len(matched_rsis) > 0:
                            dists = haversine_np(
                                site_lon, site_lat, 
                                np.degrees(np.arctan2(matched_rsis[:,1], matched_rsis[:,0])), 
                                np.degrees(np.arcsin(np.clip(matched_rsis[:,2]/6371000.0, -1.0, 1.0)))
                            )
                            current_min_d = np.min(dists)
                            if current_min_d < min_dist_for_this_group:
                                min_dist_for_this_group = current_min_d
                            if current_min_d < rsi_min_dist:
                                conflict = True
                        else:
                            min_dist_for_this_group = 1e9

                    if not conflict:
                        if min_dist_for_this_group > max_valid_rsi_dist:
                            max_valid_rsi_dist = min_dist_for_this_group
                            selected_rsi_group = group
                    else:
                        if min_dist_for_this_group > max_fallback_rsi_dist:
                            max_fallback_rsi_dist = min_dist_for_this_group
                            best_fallback_rsi_group = group

                if selected_rsi_group is None:
                    selected_rsi_group = best_fallback_rsi_group

                # Tính Góc Azimuth & Tilt
                site_assigned_azs = []
                num_cells = len(site_cells)

                for cell_idx in range(num_cells):
                    cell_row = site_cells.iloc[cell_idx].to_dict()

                    opt_azimuth = calculate_optimum_azimuth(
                        site_lat, site_lon, n_lats, n_lons, n_azs, 
                        sector_idx=cell_idx,
                        total_sectors=num_cells,
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
                    cell_row['PCI'] = int(selected_pci_group[cell_idx % len(selected_pci_group)])
                    cell_row['RSI'] = int(selected_rsi_group[cell_idx % len(selected_rsi_group)])
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
            
            add_log(f"HOÀN THÀNH: Đã tính toán xong cho {len(output_rows)} cells ({total_sites} sites) trong {elapsed_time} giây.")

            status_box.update(label="✅ Hoàn tất quy hoạch thành công!", state="complete", expanded=False)
            st.session_state["output_df"] = df_output
            st.session_state["logs"] = "\n".join(logs)
            st.session_state["exec_time"] = elapsed_time

        except Exception as e:
            status_box.update(label="❌ Có lỗi xảy ra trong quá trình xử lý!", state="error")
            st.error(f"Chi tiết lỗi: {str(e)}")

# ==========================================
# 6. THỐNG KÊ DASHBOARD & BẢNG KẾT QUẢ
# ==========================================
if "output_df" in st.session_state:
    st.markdown("### 📊 Kết Quả Quy Hoạch")
    df_out = st.session_state["output_df"]
    exec_t = st.session_state.get("exec_time", 0)
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Site Mới", f"{df_out['Sitename'].nunique()}")
    m2.metric("Tổng Cell Phân Bổ", f"{len(df_out)}")
    m3.metric("Góc E-Tilt Trung Bình", f"{df_out['E-Tilt'].mean():.1f}°")
    m4.metric("Thời Gian Xử Lý", f"{exec_t}s")

    tab_data, tab_log = st.tabs(["📋 Danh Sách Kết Quả (Output Data)", "📜 Nhật Ký Xử Lý (Logs)"])

    with tab_data:
        st.dataframe(df_out, use_container_width=True, height=380)
        csv_buffer = io.StringIO()
        df_out.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Tải Về Kết Quả Quy Hoạch (Output_RF_Design.csv)",
            type="primary",
            data=csv_buffer.getvalue().encode('utf-8-sig'),
            file_name="Output_RF_Design.csv",
            mime="text/csv"
        )

    with tab_log:
        st.code(st.session_state.get("logs", ""), language="text")
