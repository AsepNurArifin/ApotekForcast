import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import pickle
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="DSS Apotek Shaka Farma",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS INJECTION ──────────────────────────────────────────────────────────
css_path = Path(__file__).parent / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>",
                unsafe_allow_html=True)

# ── SECTION 1: LOAD STATIC ASSETS ──────────────────────────────────────────
@st.cache_resource
def load_assets():
    root = Path(__file__).parent.parent
    model    = joblib.load(root / "models" / "xgboost_best_model.joblib")
    dataset  = pd.read_csv(root / "data" / "final" / "dataset_xgboost_ready.csv")
    dataset['Tanggal'] = pd.to_datetime(dataset['Tanggal'])
    sku_eval = pd.read_csv(root / "data" / "final" / "sku_evaluation.csv")
    label    = pd.read_excel(root / "data" / "final" / "label.xlsx")
    with open(root / "notebook" / "winsor_bounds.pkl", "rb") as f:
        winsor_bounds = pickle.load(f)
    sku_class = pd.read_csv(root / "data" / "final" / "sku_classification.csv")
    return model, dataset, sku_eval, label, winsor_bounds, sku_class

model, dataset, sku_eval, label, winsor_bounds, sku_class = load_assets()

FEATURES  = ['Lag_1','Lag_2','Lag_3','Lag_4',
            'Rolling_Mean_4','Bulan','Pekan_Ke','Rata_Historis_SKU']
TARGET    = 'Jumlah'
Z         = 1.65

# ── SIDEBAR & WORKFLOW PROGRESS ───────────────────────────────────────────
# Header/Branding for Sidebar
st.sidebar.markdown('<div class="sidebar-title">Alur Kerja Sistem</div>', unsafe_allow_html=True)

# Step-based workflow progress indicator
def render_workflow_progress(current_step):
    steps = [
        ("Upload Histori Transaksi", 1),
        ("Upload Stok Aktual", 2),
        ("Validasi & Analisis Data", 3),
        ("Prediksi Kebutuhan", 4),
        ("Rekomendasi Restok", 5),
        ("Analisis Detail SKU", 6)
    ]
    
    html = '<div class="flow-step-container">'
    for label, step_num in steps:
        if step_num < current_step:
            status_class = "completed"
            icon = "✓"
        elif step_num == current_step:
            status_class = "active"
            icon = "●"
        else:
            status_class = "pending"
            icon = "○"
        html += f'<div class="flow-step {status_class}"><span class="flow-step-num">{icon}</span><span>{label}</span></div>'
    html += '</div>'
    st.sidebar.markdown(html, unsafe_allow_html=True)

# Define file uploaders with helpful cards detailing expected formats
with st.sidebar.container(border=True):
    st.markdown("""
    <div class="upload-header">
        <h4>1. Histori Transaksi</h4>
        <p>File CSV / Excel berisi riwayat penjualan obat.</p>
        <div class="upload-format-badge">Tanggal Transaksi | Kode Produk | Jumlah</div>
    </div>
    """, unsafe_allow_html=True)
    file_transaksi = st.file_uploader(
        "Upload Transaksi Terbaru",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed",
        key="uploader_transaksi"
    )

with st.sidebar.container(border=True):
    st.markdown("""
    <div class="upload-header">
        <h4>2. Stok Aktual</h4>
        <p>File CSV / Excel berisi sisa stok obat saat ini.</p>
        <div class="upload-format-badge">SKU | Stok Total</div>
    </div>
    """, unsafe_allow_html=True)
    file_stok = st.file_uploader(
        "Upload Stok Aktual",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed",
        key="uploader_stok"
    )

# Render steps progress in sidebar
if not file_transaksi:
    render_workflow_progress(1)
elif not file_stok:
    render_workflow_progress(2)
else:
    render_workflow_progress(6)

# Global Clean Header at the top
st.markdown("""
<div class="page-header-premium">
    <div class="header-main-info">
        <span class="header-tag">💊 Sistem Pendukung Keputusan</span>
        <h1 class="header-title">Sistem Prediksi Restock Obat</h1>
        <p class="header-description">Apotek Shaka Farma &mdash; Kelola persediaan obat menggunakan analitik prediktif cerdas</p>
    </div>
    <div class="header-meta">
        <div class="meta-item">
            <span class="meta-label">Status Sistem</span>
            <span class="meta-value status-online">● Terhubung</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── HELPER: baca file CSV atau Excel ───────────────────────────────────────
def read_file(uploaded_file, file_type='transaksi'):
    name = uploaded_file.name.lower()
    
    if file_type == 'stok':
        # Format export apotek digital — header di baris ke-11
        if name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, header=11, low_memory=False)
        else:
            df = pd.read_excel(uploaded_file, header=11)
        
        # Ambil hanya kolom yang dibutuhkan, rename
        df = df[['SKU', 'Stok Total']].dropna(subset=['SKU'])
        df = df.rename(columns={'SKU': 'Kode Produk', 'Stok Total': 'Stok_Aktual'})
        df['Stok_Aktual'] = pd.to_numeric(df['Stok_Aktual'], errors='coerce').fillna(0)
        return df
    
    else:
        # Auto-detect format: coba header=0 dulu (file sudah diproses),
        # kalau kolom kunci tidak ada, coba header=9 (raw export Apotek Digital)
        if name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, header=0, low_memory=False)
        else:
            df = pd.read_excel(uploaded_file, header=0)

        expected_cols = {'Tanggal Transaksi', 'Kode Produk', 'Jumlah'}
        if not expected_cols.issubset(set(df.columns)):
            # Kemungkinan format raw Apotek Digital → header di baris ke-9
            uploaded_file.seek(0)  # reset file pointer
            if name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, header=9, low_memory=False)
            else:
                df = pd.read_excel(uploaded_file, header=9)

            # Rename kolom agar sesuai pipeline
            if 'Tanggal' in df.columns and 'Tanggal Transaksi' not in df.columns:
                df = df.rename(columns={'Tanggal': 'Tanggal Transaksi'})

            # Mapping Nama Produk → SKU (Kode Produk) via label
            if 'Nama Produk' in df.columns and 'Kode Produk' not in df.columns:
                nama_to_sku = dict(zip(
                    label['Nama'].str.strip().str.upper(),
                    label['SKU']
                ))
                df['Kode Produk'] = (
                    df['Nama Produk'].str.strip().str.upper().map(nama_to_sku)
                )
                n_unmapped = df['Kode Produk'].isna().sum()
                if n_unmapped > 0:
                    st.warning(
                        f"⚠️ {n_unmapped} baris tidak bisa di-mapping ke SKU "
                        f"(dari total {len(df)} baris). Baris tersebut akan diabaikan."
                    )
                df = df.dropna(subset=['Kode Produk'])

        # Pastikan kolom Jumlah numerik
        if 'Jumlah' in df.columns:
            df['Jumlah'] = pd.to_numeric(df['Jumlah'], errors='coerce').fillna(0)

        return df

# ── SECTION 3: FEATURE UPDATE ENGINE ───────────────────────────────────────
def update_features(df_new_raw, dataset_hist, label, winsor_bounds):
    # Normalisasi label
    label['cat_norm'] = label['Draf_Kategori'].str.strip().str.lower()
    obat_skus = set(label[label['cat_norm'] == 'obat']['SKU'].dropna().unique())

    # Parse tanggal — format Apotek Digital: '1 Agt 2025 pukul 07.53'
    bulan_id = {
        'Jan': 'Jan', 'Feb': 'Feb', 'Mar': 'Mar', 'Apr': 'Apr',
        'Mei': 'May', 'Jun': 'Jun', 'Jul': 'Jul', 'Agt': 'Aug',
        'Agu': 'Aug', 'Sep': 'Sep', 'Okt': 'Oct', 'Nov': 'Nov',
        'Des': 'Dec',
    }
    tanggal_raw = df_new_raw['Tanggal Transaksi'].astype(str)
    # Hapus bagian waktu "pukul XX.XX"
    tanggal_raw = tanggal_raw.str.replace(r'\s*pukul\s+[\d.:]+', '', regex=True)
    # Ganti nama bulan Indonesia → Inggris
    for id_bln, en_bln in bulan_id.items():
        tanggal_raw = tanggal_raw.str.replace(id_bln, en_bln, regex=False)
    df_new_raw['Tanggal Transaksi'] = pd.to_datetime(
        tanggal_raw, dayfirst=True, errors='coerce'
    )
    df_new_raw = df_new_raw.dropna(subset=['Tanggal Transaksi'])

    # Filter SKU obat
    df_obat = df_new_raw[df_new_raw['Kode Produk'].isin(obat_skus)].copy()

    if df_obat.empty:
        st.error("Tidak ada transaksi obat ditemukan di file upload.")
        return pd.DataFrame()

    # Validasi: buang minggu yang belum genap (bukan Minggu)
    df_obat['week'] = df_obat['Tanggal Transaksi'].dt.to_period('W-MON')
    max_date = df_obat['Tanggal Transaksi'].max()
    if max_date.dayofweek != 6:  # 6 = Minggu
        minggu_terakhir = df_obat['week'].max()
        df_obat = df_obat[df_obat['week'] < minggu_terakhir]
        st.warning(
            f"⚠️ Data terakhir ({max_date.strftime('%d %b %Y')}) belum genap "
            f"7 hari (Senin–Minggu). Minggu yang belum selesai diabaikan otomatis."
        )

    if df_obat.empty:
        st.error("Tidak ada minggu lengkap (Senin–Minggu) di data upload.")
        return pd.DataFrame()

    # Agregasi mingguan
    weekly_new = df_obat.groupby(['Kode Produk','week'])['Jumlah'].sum().reset_index()
    weekly_new['Tanggal'] = weekly_new['week'].dt.start_time
    # Validasi: transaksi upload harus lebih baru dari histori
    max_hist = dataset_hist['Tanggal'].max()
    max_upload = weekly_new['Tanggal'].max() if not weekly_new.empty else None

    if max_upload is not None and max_upload <= max_hist:
        st.warning(
            f"⚠️ Data transaksi yang diupload ({max_upload.strftime('%d %b %Y')}) "
            f"tidak lebih baru dari histori sistem ({max_hist.strftime('%d %b %Y')}). "
            f"Prediksi tetap dijalankan menggunakan histori yang ada."
        )
    weekly_new = weekly_new.drop(columns='week')

    # Gabung ke histori
    hist     = dataset_hist[['Kode Produk','Tanggal','Jumlah']].copy()
    combined = pd.concat([hist, weekly_new], ignore_index=True)
    combined = combined.drop_duplicates(subset=['Kode Produk','Tanggal'])
    combined = combined.sort_values(['Kode Produk','Tanggal']).reset_index(drop=True)

    # Feature engineering
    combined['Rata_Historis_SKU'] = combined.groupby(
        'Kode Produk')['Jumlah'].transform('mean')
    combined['Lag_1'] = combined.groupby('Kode Produk')['Jumlah'].shift(1)
    combined['Lag_2'] = combined.groupby('Kode Produk')['Jumlah'].shift(2)
    combined['Lag_3'] = combined.groupby('Kode Produk')['Jumlah'].shift(3)
    combined['Lag_4'] = combined.groupby('Kode Produk')['Jumlah'].shift(4)
    combined['Rolling_Mean_4'] = combined.groupby('Kode Produk')['Jumlah'].transform(
        lambda x: x.shift(1).rolling(4).mean()
    )
    combined['Bulan']    = combined['Tanggal'].dt.month
    combined['Pekan_Ke'] = combined['Tanggal'].dt.isocalendar().week.astype(int)

    # Ambil baris terbaru per SKU
    latest = combined.groupby('Kode Produk').last().reset_index()
    latest = latest.dropna(subset=FEATURES)

    # Winsorization pakai bounds dari training
    for col in ['Lag_1','Lag_2','Lag_3','Lag_4','Rolling_Mean_4']:
        if col in winsor_bounds:
            latest[col] = latest[col].astype(float)
            for sku in latest['Kode Produk']:
                if sku in winsor_bounds[col]:
                    ub   = winsor_bounds[col][sku]
                    mask = (latest['Kode Produk'] == sku) & (latest[col] > ub)
                    latest.loc[mask, col] = ub

    return latest

# ── SECTION 4: DSS LOGIC ───────────────────────────────────────────────────
def run_dss_fastmoving(latest_features, sku_eval, model, stok_df):
    """DSS untuk 195 SKU fast-moving — XGBoost atau MA-4 per engine assignment."""
    results = []

    for _, row in latest_features.iterrows():
        sku      = row['Kode Produk']
        sku_info = sku_eval[sku_eval['Kode Produk'] == sku]

        if sku_info.empty:
            continue

        engine   = sku_info['Engine'].values[0]
        rmse_xgb = sku_info['RMSE_XGBoost'].values[0]
        rmse_ma4 = sku_info['RMSE_MA4'].values[0]

        if engine == 'XGBoost':
            pred   = float(model.predict(pd.DataFrame([row[FEATURES]]))[0])
            rmse_e = rmse_xgb
        else:
            pred   = float(row['Rolling_Mean_4'])
            rmse_e = rmse_ma4

        pred  = max(0, round(pred, 2))
        ss    = round(Z * rmse_e, 2)
        rop   = round(pred + ss, 2)

        stok_row = stok_df[stok_df['Kode Produk'] == sku]
        stok     = float(stok_row['Stok_Aktual'].values[0]) \
                    if not stok_row.empty else 0.0
        order = round(max(0, rop - stok), 2)
        alert = stok < rop

        results.append({
            'Kode Produk':  sku,
            'Engine':       engine,
            'Prediksi':     pred,
            'Safety Stock': ss,
            'ROP':          rop,
            'Stok Aktual':  stok,
            'Order Qty':    order,
            'Alert':        alert
        })

    return pd.DataFrame(results)


def run_dss_slowmoving(df_new_raw, label, stok_df, fast_moving_skus):
    """DSS untuk 742 SKU slow-moving — MA-4 sederhana dari histori upload."""
    label['cat_norm'] = label['Draf_Kategori'].str.strip().str.lower()
    obat_skus = set(label[label['cat_norm'] == 'obat']['SKU'].dropna().unique())

    # Hanya SKU obat yang bukan fast-moving
    slow_skus = obat_skus - set(fast_moving_skus)

    bulan_id = {
        'Jan':'Jan','Feb':'Feb','Mar':'Mar','Apr':'Apr',
        'Mei':'May','Jun':'Jun','Jul':'Jul','Agt':'Aug',
        'Agu':'Aug','Sep':'Sep','Okt':'Oct','Nov':'Nov','Des':'Dec'
    }
    tanggal_raw = df_new_raw['Tanggal Transaksi'].astype(str)
    tanggal_raw = tanggal_raw.str.replace(r'\s*pukul\s+[\d.:]+', '', regex=True)
    for id_bln, en_bln in bulan_id.items():
        tanggal_raw = tanggal_raw.str.replace(id_bln, en_bln, regex=False)
    df_new_raw = df_new_raw.copy()
    df_new_raw['Tanggal Transaksi'] = pd.to_datetime(
        tanggal_raw, dayfirst=True, errors='coerce')
    df_new_raw = df_new_raw.dropna(subset=['Tanggal Transaksi'])

    df_slow = df_new_raw[df_new_raw['Kode Produk'].isin(slow_skus)].copy()

    if df_slow.empty:
        return pd.DataFrame()

    # Agregasi mingguan
    df_slow['week'] = df_slow['Tanggal Transaksi'].dt.to_period('W-MON')
    weekly = df_slow.groupby(['Kode Produk','week'])['Jumlah'].sum().reset_index()

    # MA-4 per SKU dari 4 minggu terakhir di upload
    results = []
    for sku, grp in weekly.groupby('Kode Produk'):
        grp = grp.sort_values('week')
        last4 = grp['Jumlah'].tail(4).values
        pred  = round(float(np.mean(last4)), 2) if len(last4) > 0 else 0.0

        # RMSE proxy untuk slow-moving: std dari histori (konservatif)
        std   = float(np.std(last4)) if len(last4) > 1 else pred * 0.5
        ss    = round(Z * std, 2)
        rop   = round(pred + ss, 2)

        stok_row = stok_df[stok_df['Kode Produk'] == sku]
        stok     = float(stok_row['Stok_Aktual'].values[0]) \
                   if not stok_row.empty else 0.0
        order = round(max(0, rop - stok), 2)
        alert = stok < rop

        results.append({
            'Kode Produk':  sku,
            'Engine':       'MA-4',
            'Prediksi':     pred,
            'Safety Stock': ss,
            'ROP':          rop,
            'Stok Aktual':  stok,
            'Order Qty':    order,
            'Alert':        alert
        })

    return pd.DataFrame(results)

# ── SECTION 5: MAIN UI ─────────────────────────────────────────────────────
def render_kpi_cards(hasil_df):
    total_sku  = len(hasil_df)
    alert_cnt  = int(hasil_df['Alert'].sum())
    safe_cnt   = total_sku - alert_cnt
    
    # Calculate high risk stockout (Stok Aktual < Safety Stock AND alert == True)
    high_risk_cnt = int(((hasil_df['Alert']) & (hasil_df['Stok Aktual'] <= hasil_df['Safety Stock'])).sum())
    
    # Estimated Weekly Demand (sum of predictions)
    est_demand = int(hasil_df['Prediksi'].sum())

    c1, c2, c3, c4 = st.columns(4)
    st.markdown('<div style="margin-bottom: 4px"></div>', unsafe_allow_html=True)

    with c1:
        st.markdown(f"""
        <div class="kpi-card kpi-blue">
            <span class="kpi-icon">📦</span>
            <div class="kpi-value">{total_sku}</div>
            <div class="kpi-label">Total SKU Dianalisis</div>
            <div class="kpi-sub">Kebutuhan stok terdata</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="kpi-card kpi-red">
            <span class="kpi-icon">🚨</span>
            <div class="kpi-value">{alert_cnt}</div>
            <div class="kpi-label">Perlu Restok</div>
            <div class="kpi-sub">{high_risk_cnt} Kritis / Habis</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="kpi-card kpi-green">
            <span class="kpi-icon">✅</span>
            <div class="kpi-value">{safe_cnt}</div>
            <div class="kpi-label">Stok Aman</div>
            <div class="kpi-sub">Memenuhi kebutuhan</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="kpi-card kpi-slate">
            <span class="kpi-icon">📈</span>
            <div class="kpi-value">{est_demand} Unit</div>
            <div class="kpi-label">Estimasi Demand</div>
            <div class="kpi-sub">Pekan depan (total)</div>
        </div>
        """, unsafe_allow_html=True)


def render_section_header(icon, title, count=None):
    """Render a styled section header."""
    count_html = f'<span class="section-count">{count}</span>' if count else ''
    st.markdown(f"""
    <div class="section-header">
        <span class="section-icon">{icon}</span>
        <span class="section-title">{title}</span>
        {count_html}
    </div>
    """, unsafe_allow_html=True)


def add_status_column(df):
    """Add styled status badge column to dataframe."""
    df_display = df.copy()
    df_display['Status'] = df_display['Alert'].apply(
        lambda x: '🔴 PERLU ORDER' if x else '🟢 AMAN'
    )
    df_display = df_display.drop(columns=['Alert'])
    return df_display


def render_sku_scorecard(sku_sel, hasil_df):
    """Render a 4-column metrics grid for the selected SKU."""
    sku_row = hasil_df[hasil_df['Kode Produk'] == sku_sel]
    if sku_row.empty:
        return

    row = sku_row.iloc[0]
    pred = row['Prediksi']
    ss = row['Safety Stock']
    rop = row['ROP']
    stok = row['Stok Aktual']
    order = row['Order Qty']
    engine = row['Engine']
    alert = row['Alert']
    nama = row.get('Nama Produk', '-')
    kategori = row.get('Kategori', '-')
    minggu_aktif = int(row.get('Minggu_Aktif', 0))
    total_minggu = int(sku_class['Total_Minggu'].iloc[0]) if len(sku_class) > 0 else 57

    status_class = "sku-alert" if alert else "sku-safe"
    status_text = "🔴 PERLU ORDER" if alert else "🟢 STOK AMAN"
    kat_badge_class = "badge-success" if kategori == "Fast-Moving" else "badge-warning"

    st.markdown(f"""
<div class="sku-scorecard-container">
<div class="sku-scorecard-title">💊 <strong>{sku_sel}</strong> — {nama}
&nbsp;|&nbsp; <span class="badge {kat_badge_class}">{kategori}</span>
<span class="badge badge-info">Aktif {minggu_aktif}/{total_minggu} minggu</span>
&nbsp;|&nbsp; Status: <span class="badge {'badge-danger' if alert else 'badge-success'}">{status_text}</span>
</div>
<div class="sku-grid">
<div class="sku-metric-card sku-info">
<div class="metric-label">Engine Peramal</div>
<div class="metric-val">{engine}</div>
<div class="metric-sub">Model prediktif terpilih</div>
</div>
<div class="sku-metric-card">
<div class="metric-label">Prediksi Demand</div>
<div class="metric-val">{pred:.1f}</div>
<div class="metric-sub">Prediksi terjual pekan depan</div>
</div>
<div class="sku-metric-card sku-warning">
<div class="metric-label">Safety Stock & ROP</div>
<div class="metric-val">{ss:.1f} | {rop:.1f}</div>
<div class="metric-sub">Batas aman reorder point</div>
</div>
<div class="sku-metric-card {status_class}">
<div class="metric-label">Stok Aktual & Order</div>
<div class="metric-val">{stok:.0f} → +{order:.0f}</div>
<div class="metric-sub">Stok saat ini & saran order</div>
</div>
</div>
</div>
""", unsafe_allow_html=True)


def render_plotly_trend(sku_sel, sku_hist, sku_rop=None, sku_ss=None):
    """Render interactive Plotly trend chart for a selected SKU."""
    fig = go.Figure()

    # Area fill
    fig.add_trace(go.Scatter(
        x=sku_hist['Tanggal'], y=sku_hist['Jumlah'],
        fill='tozeroy',
        fillcolor='rgba(37, 99, 235, 0.04)',
        line=dict(color='rgba(37, 99, 235, 0)', width=0),
        showlegend=False, hoverinfo='skip'
    ))

    # Main line
    fig.add_trace(go.Scatter(
        x=sku_hist['Tanggal'], y=sku_hist['Jumlah'],
        mode='lines+markers',
        name='Penjualan Mingguan',
        line=dict(color='#2563EB', width=2.5, shape='spline'),
        marker=dict(size=7, color='#2563EB',
                    line=dict(width=2, color='#FFFFFF')),
        hovertemplate='<b>%{x|%d %b %Y}</b><br>Jumlah: %{y}<extra></extra>'
    ))

    # ROP reference line
    if sku_rop is not None:
        fig.add_hline(
            y=sku_rop, line_dash='dash', line_color='#DC2626', line_width=1.5,
            annotation_text=f'ROP ({sku_rop})',
            annotation_position='top right',
            annotation_font=dict(color='#DC2626', size=11)
        )

    # Safety Stock reference line
    if sku_ss is not None:
        fig.add_hline(
            y=sku_ss, line_dash='dot', line_color='#D97706', line_width=1,
            annotation_text=f'Safety Stock ({sku_ss})',
            annotation_position='bottom right',
            annotation_font=dict(color='#D97706', size=11)
        )

    fig.update_layout(
        template='plotly_white',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, sans-serif', color='#475569'),
        title=dict(
            text=f'Tren Penjualan Mingguan — {sku_sel}',
            font=dict(size=14, color='#1E293B')
        ),
        xaxis=dict(
            title='Minggu', gridcolor='#E2E8F0',
            tickformat='%d %b %y'
        ),
        yaxis=dict(
            title='Jumlah Terjual', gridcolor='#E2E8F0'
        ),
        margin=dict(l=40, r=20, t=50, b=40),
        height=380,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    st.plotly_chart(fig, use_container_width=True)


def render_engine_donut(hasil_df):
    """Render engine distribution donut chart."""
    engine_counts = hasil_df['Engine'].value_counts()
    colors = ['#2563EB', '#14B8A6']

    fig = go.Figure(data=[go.Pie(
        labels=engine_counts.index,
        values=engine_counts.values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color='#FFFFFF', width=2)),
        textinfo='label+value',
        textfont=dict(size=11, color='#1E293B'),
        hovertemplate='<b>%{label}</b><br>Jumlah: %{value}<br>(%{percent})<extra></extra>'
    )])

    fig.update_layout(
        template='plotly_white',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, sans-serif', color='#475569'),
        title=dict(
            text='Distribusi Engine Prediksi',
            font=dict(size=13, color='#1E293B')
        ),
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False,
        annotations=[dict(
            text=f'<b>{len(hasil_df)}</b><br>SKU',
            x=0.5, y=0.5, font_size=16, font_color='#1E293B',
            showarrow=False
        )]
    )

    st.plotly_chart(fig, use_container_width=True)


# ── MAIN FLOW ──────────────────────────────────────────────────────────────
if file_transaksi and file_stok:
    df_new_raw = read_file(file_transaksi, file_type='transaksi')
    stok_df    = read_file(file_stok, file_type='stok')

    fast_moving_skus = dataset['Kode Produk'].unique().tolist()
    sku_to_nama = dict(zip(label['SKU'], label['Nama']))

    with st.spinner("⏳ Memproses data dan menghitung prediksi..."):
        latest      = update_features(df_new_raw, dataset, label, winsor_bounds)
        hasil_fast  = run_dss_fastmoving(latest, sku_eval, model, stok_df)
        hasil_slow  = run_dss_slowmoving(df_new_raw, label, stok_df, fast_moving_skus)
        hasil_df    = pd.concat([hasil_fast, hasil_slow], ignore_index=True)

    # ── Hybrid: deteksi SKU baru yang tidak ada di baseline ────────────
    sku_class_live = sku_class.copy()
    existing_skus  = set(sku_class_live['Kode Produk'])
    new_skus       = set(hasil_df['Kode Produk']) - existing_skus
    if new_skus:
        new_rows = pd.DataFrame({
            'Kode Produk': list(new_skus),
            'Minggu_Aktif': 0,
            'Total_Minggu': int(sku_class_live['Total_Minggu'].iloc[0]),
            'Threshold':    int(sku_class_live['Threshold'].iloc[0]),
            'Kategori':     'Slow-Moving',
            'Nama Produk':  [sku_to_nama.get(s, '-') for s in new_skus]
        })
        sku_class_live = pd.concat([sku_class_live, new_rows], ignore_index=True)

    # Merge metadata klasifikasi ke hasil DSS
    hasil_df = hasil_df.merge(
        sku_class_live[['Kode Produk', 'Nama Produk', 'Kategori', 'Minggu_Aktif']],
        on='Kode Produk', how='left'
    )
    hasil_df['Nama Produk']  = hasil_df['Nama Produk'].fillna('-')
    hasil_df['Kategori']     = hasil_df['Kategori'].fillna('Slow-Moving')
    hasil_df['Minggu_Aktif'] = hasil_df['Minggu_Aktif'].fillna(0).astype(int)

    if hasil_df.empty:
        st.error("Tidak ada hasil. Periksa file upload.")
    else:
        # ── KPI Cards ──────────────────────────────────────────────────
        render_kpi_cards(hasil_df)
        st.markdown('<div style="height: 8px"></div>', unsafe_allow_html=True)

        # ── Tabbed Interface ───────────────────────────────────────────
        tab_summary, tab_recs, tab_details = st.tabs([
            "📊 Ringkasan & Alerts",
            "📋 Daftar Rekomendasi",
            "📈 Analisis Detail SKU"
        ])

        # ── TAB 1: RINGKASAN & ALERTS ──────────────────────────────────
        with tab_summary:
            alert_df = hasil_df[hasil_df['Alert']].sort_values('Order Qty', ascending=False)
            
            col_l, col_r = st.columns([3, 1.2])

            with col_l:
                # Prioritas Rekomendasi Restok (Decision Cards)
                if len(alert_df) > 0:
                    render_section_header('⚠️', 'Rekomendasi Restok Prioritas')
                    
                    cols_html = '<div class="decision-grid">'
                    # Display top 6 critical items needing action
                    for _, item in alert_df.head(6).iterrows():
                        sku = item['Kode Produk']
                        nama = item['Nama Produk']
                        stok = item['Stok Aktual']
                        order = item['Order Qty']
                        rop = item['ROP']
                        kategori = item['Kategori']
                        
                        ratio = stok / rop if rop > 0 else 0
                        if ratio == 0:
                            risk_badge = '<span class="badge badge-danger">SANGAT TINGGI (HABIS)</span>'
                        elif ratio < 0.3:
                            risk_badge = '<span class="badge badge-danger">TINGGI</span>'
                        else:
                            risk_badge = '<span class="badge badge-warning">SEDANG</span>'
                            
                        cols_html += f'<div class="decision-card"><div class="decision-header-info"><span class="decision-sku">{sku}</span>{risk_badge}</div><div class="decision-name">{nama}</div><div class="decision-recom-block"><div class="decision-recom-title">Rekomendasi:</div><div class="decision-recom-val">BELI SEKARANG (+{int(order)} Unit)</div></div><div class="decision-footer-stats"><span>Stok: <b>{int(stok)}</b></span><span>Batas ROP: <b>{int(rop)}</b></span><span>Kategori: <b>{kategori}</b></span></div></div>'
                    cols_html += '</div>'
                    st.markdown(cols_html, unsafe_allow_html=True)
                    
                    render_section_header('🚨', 'Alert: SKU di Bawah ROP', count=f'{len(alert_df)} SKU')
                    st.markdown(f"""
                    <div class="alert-banner">
                        <span class="alert-icon">⚠️</span>
                        <span class="alert-text">
                            <strong>{len(alert_df)} SKU</strong> membutuhkan pengadaan segera!
                            Stok saat ini berada di bawah batas Reorder Point (ROP).
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

                    alert_display = add_status_column(alert_df)
                    alert_display['Level Stok'] = alert_df.apply(
                        lambda r: min(100.0, (r['Stok Aktual'] / r['ROP']) * 100.0) if r['ROP'] > 0 else (100.0 if r['Stok Aktual'] > 0 else 0.0),
                        axis=1
                    )
                    
                    # Reorder columns for visual balance
                    cols = ['Kode Produk', 'Nama Produk', 'Kategori', 'Engine', 'Prediksi', 'Safety Stock', 'ROP', 'Stok Aktual', 'Level Stok', 'Order Qty', 'Status']
                    alert_display = alert_display[[c for c in cols if c in alert_display.columns]]

                    st.dataframe(
                        alert_display,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            'Order Qty': st.column_config.NumberColumn(
                                'Order Qty', format='%.0f'
                            ),
                            'Prediksi': st.column_config.NumberColumn(
                                'Prediksi', format='%.1f'
                            ),
                            'Safety Stock': st.column_config.NumberColumn(
                                'Safety Stock', format='%.1f'
                            ),
                            'ROP': st.column_config.NumberColumn(
                                'ROP', format='%.1f'
                            ),
                            'Stok Aktual': st.column_config.NumberColumn(
                                'Stok Aktual', format='%.0f'
                            ),
                            'Level Stok': st.column_config.ProgressColumn(
                                'Sisa Stok vs ROP', format='%.0f%%',
                                min_value=0, max_value=100
                            )
                        }
                    )
                else:
                    st.markdown("""
                    <div class="safe-banner">
                        <span class="alert-icon">✅</span>
                        <span class="alert-text">Semua stok aman — tidak ada SKU obat yang membutuhkan order segera.</span>
                    </div>
                    """, unsafe_allow_html=True)

            with col_r:
                render_section_header('🤖', 'Breakdown Engine')
                render_engine_donut(hasil_df)

                # Panel transparansi klasifikasi
                total_minggu = int(sku_class['Total_Minggu'].iloc[0])
                threshold    = int(sku_class['Threshold'].iloc[0])
                fast_cnt     = int((hasil_df['Kategori'] == 'Fast-Moving').sum())
                slow_cnt     = int((hasil_df['Kategori'] == 'Slow-Moving').sum())

                with st.expander("ℹ️ Klasifikasi Fast/Slow Moving", expanded=True):
                    st.markdown(f"""
                    **Kriteria Klasifikasi:**
                    - Histori transaksi mencakup **{total_minggu} minggu**
                    - SKU aktif **≥ {threshold} minggu** → **Fast-Moving** ({fast_cnt} SKU)
                    - SKU aktif **< {threshold} minggu** → **Slow-Moving** ({slow_cnt} SKU)
                    
                    **Implikasi Engine:**
                    - **Fast-Moving**: Seleksi XGBoost / MA-4 berdasarkan RMSE terendah.
                    - **Slow-Moving**: Prediksi otomatis menggunakan MA-4.
                    """)

        # ── TAB 2: DAFTAR REKOMENDASI LENGKAP ──────────────────────────
        with tab_recs:
            render_section_header('📋', 'Daftar Rekomendasi Pengadaan',
                                  count=f'{len(hasil_df)} SKU')

            # Search and filters layout
            c_f1, c_f2, c_f3, c_f4 = st.columns([2.5, 1, 1, 1])
            with c_f1:
                search_query = st.text_input("🔍 Cari SKU atau Nama Produk:", "", key="search_sku_input")
            with c_f2:
                filter_status = st.selectbox("Urutan Urgensi:", ["Semua", "Perlu Order", "Aman"], key="filter_status_select")
            with c_f3:
                filter_engine = st.selectbox("Model Peramal:", ["Semua", "XGBoost", "MA-4"], key="filter_engine_select")
            with c_f4:
                filter_kategori = st.selectbox("Kategori:", ["Semua", "Fast-Moving", "Slow-Moving"], key="filter_kategori_select")

            full_display = add_status_column(hasil_df)
            full_display['Level Stok'] = hasil_df.apply(
                lambda r: min(100.0, (r['Stok Aktual'] / r['ROP']) * 100.0) if r['ROP'] > 0 else (100.0 if r['Stok Aktual'] > 0 else 0.0),
                axis=1
            )
            cols = ['Kode Produk', 'Nama Produk', 'Kategori', 'Engine', 'Prediksi', 'Safety Stock', 'ROP', 'Stok Aktual', 'Level Stok', 'Order Qty', 'Status']
            full_display = full_display[[c for c in cols if c in full_display.columns]]

            # Apply filters
            filtered_df = full_display.copy()
            if search_query:
                filtered_df = filtered_df[
                    filtered_df['Kode Produk'].str.contains(search_query, case=False) |
                    filtered_df['Nama Produk'].str.contains(search_query, case=False, na=False)
                ]
            
            if filter_status == "Perlu Order":
                filtered_df = filtered_df[filtered_df['Status'] == '🔴 PERLU ORDER']
            elif filter_status == "Aman":
                filtered_df = filtered_df[filtered_df['Status'] == '🟢 AMAN']

            if filter_engine != "Semua":
                filtered_df = filtered_df[filtered_df['Engine'] == filter_engine]

            if filter_kategori != "Semua":
                filtered_df = filtered_df[filtered_df['Kategori'] == filter_kategori]

            st.dataframe(
                filtered_df.sort_values('Order Qty', ascending=False),
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Order Qty': st.column_config.NumberColumn(
                        'Order Qty', format='%.0f'
                    ),
                    'Prediksi': st.column_config.NumberColumn(
                        'Prediksi', format='%.1f'
                    ),
                    'Safety Stock': st.column_config.NumberColumn(
                        'Safety Stock', format='%.1f'
                    ),
                    'ROP': st.column_config.NumberColumn(
                        'ROP', format='%.1f'
                    ),
                    'Stok Aktual': st.column_config.NumberColumn(
                        'Stok Aktual', format='%.0f'
                    ),
                    'Level Stok': st.column_config.ProgressColumn(
                        'Sisa Stok vs ROP', format='%.0f%%',
                        min_value=0, max_value=100
                    )
                }
            )

            # Styled download section
            csv_data = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label='📥 Download Hasil Rekomendasi (CSV)',
                data=csv_data,
                file_name='rekomendasi_pengadaan_apotek.csv',
                mime='text/csv'
            )

        # ── TAB 3: ANALISIS DETAIL SKU ─────────────────────────────────
        with tab_details:
            render_section_header('📈', 'Visualisasi & Detail SKU')

            sku_to_nama_map = dict(zip(hasil_df['Kode Produk'], hasil_df['Nama Produk']))
            sku_sel = st.selectbox(
                'Pilih Kode SKU untuk analisis mendalam:',
                hasil_df['Kode Produk'].tolist(),
                format_func=lambda x: f"{x} — {sku_to_nama_map.get(x, '-')}",
                key="sku_select_combobox"
            )

            # Display new interactive SKU Scorecard
            render_sku_scorecard(sku_sel, hasil_df)

            sku_hist = dataset[dataset['Kode Produk'] == sku_sel]\
                       .sort_values('Tanggal').tail(16)

            sku_row = hasil_df[hasil_df['Kode Produk'] == sku_sel]
            sku_rop = float(sku_row['ROP'].values[0]) if not sku_row.empty else None
            sku_ss  = float(sku_row['Safety Stock'].values[0]) if not sku_row.empty else None

            # Display custom Plotly Trend Line or info message
            if sku_hist.empty:
                st.info("📊 Histori tren mingguan tidak tersedia untuk SKU slow-moving. "
                        "Prediksi dihitung dari data upload terbaru.")
            else:
                render_plotly_trend(sku_sel, sku_hist, sku_rop, sku_ss)

        # ── Footer ─────────────────────────────────────────────────────
        st.markdown("""
        <div class="footer-info">
            Sistem Prediksi Restock Obat Apotek Shaka Farma &middot; Hybrid Engine (XGBoost + MA-4)
            &middot; Safety Stock Z=1.65
        </div>
        """, unsafe_allow_html=True)

else:
    # ── Landing / Welcome Section ───────────────────────────────────────
    st.markdown("""
    <div class="hero-container">
        <h2>Sistem Prediksi Restock Obat</h2>
        <p>
            Platform pengambilan keputusan cerdas untuk memproyeksikan kebutuhan 
            stok obat mingguan dan meminimalkan risiko kekosongan stok obat (stockout).
        </p>
        <div class="steps-grid">
            <div class="step-card">
                <span class="step-icon">📤</span>
                <span class="step-num">1</span>
                <div class="step-title">Upload Transaksi</div>
                <div class="step-desc">Unggah file riwayat penjualan kasir terbaru</div>
            </div>
            <div class="step-card">
                <span class="step-icon">📊</span>
                <span class="step-num">2</span>
                <div class="step-title">Upload Stok</div>
                <div class="step-desc">Unggah data sisa stok fisik apotek</div>
            </div>
            <div class="step-card">
                <span class="step-icon">🤖</span>
                <span class="step-num">3</span>
                <div class="step-title">Analisis Prediksi</div>
                <div class="step-desc">Proyeksi demand otomatis berbasis AI & MA-4</div>
            </div>
            <div class="step-card">
                <span class="step-icon">📋</span>
                <span class="step-num">4</span>
                <div class="step-title">Rekomendasi</div>
                <div class="step-desc">Dapatkan rekomendasi & saran jumlah order</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("""
    <div style="margin-top: 10px; padding: 14px; background: rgba(37,99,235,0.04);
         border: 1px solid rgba(37,99,235,0.15); border-radius: 8px;
         font-size: 0.8rem; color: #475569; line-height: 1.5;">
        📌 <strong>Panduan Cepat:</strong><br>
        1. Upload file riwayat transaksi di panel kiri.<br>
        2. Upload file stok aktual terbaru.<br>
        3. Sistem akan otomatis memproses dan menampilkan hasil prediksi serta rekomendasi restok.
    </div>
    """, unsafe_allow_html=True)