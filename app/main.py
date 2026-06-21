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
    return model, dataset, sku_eval, label, winsor_bounds

model, dataset, sku_eval, label, winsor_bounds = load_assets()

FEATURES  = ['Lag_1','Lag_2','Lag_3','Lag_4',
             'Rolling_Mean_4','Bulan','Pekan_Ke','Rata_Historis_SKU']
TARGET    = 'Jumlah'
Z         = 1.65

# ── SECTION 2: SIDEBAR & FILE UPLOAD ───────────────────────────────────────
# Page Title
st.markdown("""
<div class="page-title"><h1>💊 DSS Prediksi Stok Obat</h1></div>
<div class="page-subtitle">Sistem Pendukung Keputusan — Apotek Shaka Farma</div>
""", unsafe_allow_html=True)

# Sidebar branding
st.sidebar.markdown("""
<div class="sidebar-brand">
    <span class="brand-icon">🏥</span>
    <div class="brand-name">Apotek Shaka Farma</div>
    <div class="brand-sub">Decision Support System v2.0</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-section-label">📂 Input Data</div>',
                    unsafe_allow_html=True)

file_transaksi = st.sidebar.file_uploader(
    "Upload Transaksi Terbaru",
    type=["csv", "xlsx", "xls"],
    help="File CSV/Excel dari kasir apotek"
)
file_stok = st.sidebar.file_uploader(
    "Upload Stok Aktual",
    type=["csv", "xlsx", "xls"],
    help="File CSV/Excel berisi SKU dan stok"
)

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
            for sku in latest['Kode Produk']:
                if sku in winsor_bounds[col]:
                    ub   = winsor_bounds[col][sku]
                    mask = (latest['Kode Produk'] == sku) & (latest[col] > ub)
                    latest.loc[mask, col] = ub

    return latest

# ── SECTION 4: DSS LOGIC ───────────────────────────────────────────────────
def run_dss(latest_features, sku_eval, model, stok_df):
    results = []

    for _, row in latest_features.iterrows():
        sku      = row['Kode Produk']
        sku_info = sku_eval[sku_eval['Kode Produk'] == sku]

        if sku_info.empty:
            continue

        engine   = sku_info['Engine'].values[0]
        rmse_xgb = sku_info['RMSE_XGBoost'].values[0]
        rmse_ma4 = sku_info['RMSE_MA4'].values[0]

        # Prediksi demand
        if engine == 'XGBoost':
            pred   = float(model.predict(
                pd.DataFrame([row[FEATURES]]))[0])
            rmse_e = rmse_xgb
        else:
            pred   = float(row['Rolling_Mean_4'])
            rmse_e = rmse_ma4

        pred = max(0, round(pred, 2))

        # Kalkulasi inventori
        ss       = round(Z * rmse_e, 2)
        rop      = round(pred + ss, 2)
        stok_row = stok_df[stok_df['Kode Produk'] == sku]
        stok     = float(stok_row['Stok_Aktual'].values[0]) \
                   if not stok_row.empty else 0.0
        order    = round(max(0, rop - stok), 2)
        alert    = stok < rop

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

# ── SECTION 5: MAIN UI ─────────────────────────────────────────────────────

def render_kpi_cards(hasil_df):
    """Render 4 KPI metric cards."""
    total_sku  = len(hasil_df)
    alert_cnt  = int(hasil_df['Alert'].sum())
    safe_cnt   = total_sku - alert_cnt
    xgb_cnt    = int((hasil_df['Engine'] == 'XGBoost').sum())
    ma4_cnt    = total_sku - xgb_cnt

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="kpi-card kpi-cyan kpi-delay-1">
            <span class="kpi-icon">📦</span>
            <div class="kpi-value">{total_sku}</div>
            <div class="kpi-label">Total SKU Dianalisis</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="kpi-card kpi-red kpi-delay-2">
            <span class="kpi-icon">🚨</span>
            <div class="kpi-value">{alert_cnt}</div>
            <div class="kpi-label">SKU Perlu Order</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="kpi-card kpi-green kpi-delay-3">
            <span class="kpi-icon">✅</span>
            <div class="kpi-value">{safe_cnt}</div>
            <div class="kpi-label">SKU Stok Aman</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="kpi-card kpi-slate kpi-delay-4">
            <span class="kpi-icon">🤖</span>
            <div class="kpi-value">XGBoost: {xgb_cnt} &nbsp;|&nbsp; MA-4: {ma4_cnt}</div>
            <div class="kpi-label">Engine Breakdown</div>
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

    status_class = "sku-alert" if alert else "sku-safe"
    status_text = "🔴 PERLU ORDER" if alert else "🟢 STOK AMAN"

    st.markdown(f"""
    <div class="sku-scorecard-container">
        <div class="sku-scorecard-title">💊 SKU Terpilih: <strong>{sku_sel}</strong> &nbsp;|&nbsp; Status: <span class="badge {'badge-danger pulse' if alert else 'badge-success'}">{status_text}</span></div>
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
        fillcolor='rgba(16, 185, 129, 0.08)',
        line=dict(color='rgba(16, 185, 129, 0)', width=0),
        showlegend=False, hoverinfo='skip'
    ))

    # Main line
    fig.add_trace(go.Scatter(
        x=sku_hist['Tanggal'], y=sku_hist['Jumlah'],
        mode='lines+markers',
        name='Penjualan Mingguan',
        line=dict(color='#10B981', width=2.5, shape='spline'),
        marker=dict(size=7, color='#10B981',
                    line=dict(width=2, color='#0F172A')),
        hovertemplate='<b>%{x|%d %b %Y}</b><br>Jumlah: %{y}<extra></extra>'
    ))

    # ROP reference line
    if sku_rop is not None:
        fig.add_hline(
            y=sku_rop, line_dash='dash', line_color='#EF4444', line_width=1.5,
            annotation_text=f'ROP ({sku_rop})',
            annotation_position='top right',
            annotation_font=dict(color='#EF4444', size=11)
        )

    # Safety Stock reference line
    if sku_ss is not None:
        fig.add_hline(
            y=sku_ss, line_dash='dot', line_color='#F59E0B', line_width=1,
            annotation_text=f'Safety Stock ({sku_ss})',
            annotation_position='bottom right',
            annotation_font=dict(color='#F59E0B', size=11)
        )

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, sans-serif', color='#94A3B8'),
        title=dict(
            text=f'Tren Penjualan Mingguan — {sku_sel}',
            font=dict(size=16, color='#F1F5F9')
        ),
        xaxis=dict(
            title='Minggu', gridcolor='rgba(148,163,184,0.08)',
            tickformat='%d %b %y'
        ),
        yaxis=dict(
            title='Jumlah Terjual', gridcolor='rgba(148,163,184,0.08)'
        ),
        margin=dict(l=40, r=20, t=50, b=40),
        height=380,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    st.plotly_chart(fig, width='stretch')


def render_engine_donut(hasil_df):
    """Render engine distribution donut chart."""
    engine_counts = hasil_df['Engine'].value_counts()
    colors = ['#10B981', '#06B6D4']

    fig = go.Figure(data=[go.Pie(
        labels=engine_counts.index,
        values=engine_counts.values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color='#0F172A', width=2)),
        textinfo='label+value',
        textfont=dict(size=12, color='#F1F5F9'),
        hovertemplate='<b>%{label}</b><br>Jumlah: %{value}<br>(%{percent})<extra></extra>'
    )])

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, sans-serif', color='#94A3B8'),
        title=dict(
            text='Distribusi Engine Prediksi',
            font=dict(size=14, color='#F1F5F9')
        ),
        height=260,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False,
        annotations=[dict(
            text=f'<b>{len(hasil_df)}</b><br>SKU',
            x=0.5, y=0.5, font_size=16, font_color='#F1F5F9',
            showarrow=False
        )]
    )

    st.plotly_chart(fig, width='stretch')


# ── MAIN FLOW ──────────────────────────────────────────────────────────────
if file_transaksi and file_stok:
    df_new_raw = read_file(file_transaksi, file_type='transaksi')
    stok_df    = read_file(file_stok, file_type='stok')

    with st.spinner("⏳ Memproses data dan menghitung prediksi..."):
        latest   = update_features(df_new_raw, dataset, label, winsor_bounds)
        hasil_df = run_dss(latest, sku_eval, model, stok_df)

    if hasil_df.empty:
        st.error("Tidak ada hasil. Periksa file upload.")
    else:
        # ── KPI Cards ──────────────────────────────────────────────────
        render_kpi_cards(hasil_df)

        st.markdown('<div style="height: 12px"></div>', unsafe_allow_html=True)

        # ── Tabbed Interface ───────────────────────────────────────────
        tab_summary, tab_recs, tab_details = st.tabs([
            "📊 Ringkasan & Alerts",
            "📋 Daftar Rekomendasi",
            "📈 Analisis Detail SKU"
        ])

        # ── TAB 1: RINGKASAN & ALERTS ──────────────────────────────────
        with tab_summary:
            alert_df = hasil_df[hasil_df['Alert']].sort_values(
                'Order Qty', ascending=False)
            
            col_l, col_r = st.columns([2, 1])

            with col_l:
                render_section_header('🚨', 'Alert: SKU di Bawah ROP',
                                      count=f'{len(alert_df)} SKU')
                if len(alert_df) > 0:
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
                    cols = ['Kode Produk', 'Engine', 'Prediksi', 'Safety Stock', 'ROP', 'Stok Aktual', 'Level Stok', 'Order Qty', 'Status']
                    alert_display = alert_display[cols]

                    st.dataframe(
                        alert_display,
                        width='stretch',
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

        # ── TAB 2: DAFTAR REKOMENDASI LENGKAP ──────────────────────────
        with tab_recs:
            render_section_header('📋', 'Daftar Rekomendasi Pengadaan',
                                  count=f'{len(hasil_df)} SKU')

            # Search and filters layout
            c_f1, c_f2, c_f3 = st.columns([2, 1, 1])
            with c_f1:
                search_query = st.text_input("🔍 Cari berdasarkan Kode SKU:", "", key="search_sku_input")
            with c_f2:
                filter_status = st.selectbox("🚦 Urgensi Stok:", ["Semua", "Perlu Order", "Aman"], key="filter_status_select")
            with c_f3:
                filter_engine = st.selectbox("🤖 Model Peramal:", ["Semua", "XGBoost", "MA-4"], key="filter_engine_select")

            full_display = add_status_column(hasil_df)
            full_display['Level Stok'] = hasil_df.apply(
                lambda r: min(100.0, (r['Stok Aktual'] / r['ROP']) * 100.0) if r['ROP'] > 0 else (100.0 if r['Stok Aktual'] > 0 else 0.0),
                axis=1
            )
            cols = ['Kode Produk', 'Engine', 'Prediksi', 'Safety Stock', 'ROP', 'Stok Aktual', 'Level Stok', 'Order Qty', 'Status']
            full_display = full_display[cols]

            # Apply filters
            filtered_df = full_display.copy()
            if search_query:
                filtered_df = filtered_df[filtered_df['Kode Produk'].str.contains(search_query, case=False)]
            
            if filter_status == "Perlu Order":
                filtered_df = filtered_df[filtered_df['Status'] == '🔴 PERLU ORDER']
            elif filter_status == "Aman":
                filtered_df = filtered_df[filtered_df['Status'] == '🟢 AMAN']

            if filter_engine != "Semua":
                filtered_df = filtered_df[filtered_df['Engine'] == filter_engine]

            st.dataframe(
                filtered_df.sort_values('Order Qty', ascending=False),
                width='stretch',
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
            st.markdown('<div class="download-section"></div>', unsafe_allow_html=True)
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

            sku_sel = st.selectbox(
                'Pilih Kode SKU untuk analisis mendalam:',
                hasil_df['Kode Produk'].tolist(),
                key="sku_select_combobox"
            )

            # Display new interactive SKU Scorecard
            render_sku_scorecard(sku_sel, hasil_df)

            sku_hist = dataset[dataset['Kode Produk'] == sku_sel]\
                       .sort_values('Tanggal').tail(16)

            sku_row = hasil_df[hasil_df['Kode Produk'] == sku_sel]
            sku_rop = float(sku_row['ROP'].values[0]) if not sku_row.empty else None
            sku_ss  = float(sku_row['Safety Stock'].values[0]) if not sku_row.empty else None

            # Display custom Plotly Trend Line
            render_plotly_trend(sku_sel, sku_hist, sku_rop, sku_ss)

        # ── Footer ─────────────────────────────────────────────────────
        st.markdown("""
        <div class="footer-info">
            DSS Apotek Shaka Farma &middot; Hybrid Engine (XGBoost + MA-4)
            &middot; Safety Stock Z=1.65
        </div>
        """, unsafe_allow_html=True)

else:
    # ── Hero / Welcome Section ─────────────────────────────────────────
    st.markdown("""
    <div class="hero-container">
        <h2>Selamat Datang di DSS Apotek Shaka Farma</h2>
        <p>
            Sistem cerdas untuk memprediksi kebutuhan stok obat dan
            memberikan rekomendasi pengadaan berdasarkan analisis
            data penjualan.
        </p>
        <div class="steps-grid">
            <div class="step-card">
                <span class="step-icon">📤</span>
                <span class="step-num">1</span>
                <div class="step-title">Upload Data</div>
                <div class="step-desc">File transaksi penjualan & stok aktual</div>
            </div>
            <div class="step-card">
                <span class="step-icon">🤖</span>
                <span class="step-num">2</span>
                <div class="step-title">Analisis AI</div>
                <div class="step-desc">Prediksi demand dengan XGBoost & MA-4</div>
            </div>
            <div class="step-card">
                <span class="step-icon">📋</span>
                <span class="step-num">3</span>
                <div class="step-title">Rekomendasi</div>
                <div class="step-desc">Daftar order & alert stok kritis</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("""
    <div style="margin-top: 20px; padding: 12px; background: rgba(16,185,129,0.06);
         border: 1px solid rgba(16,185,129,0.15); border-radius: 8px;
         font-size: 0.78rem; color: #94A3B8; line-height: 1.5;">
        📌 <strong>Cara Penggunaan:</strong><br>
        1. Upload file transaksi terbaru<br>
        2. Upload file stok aktual<br>
        3. Sistem akan otomatis menganalisis
    </div>
    """, unsafe_allow_html=True)