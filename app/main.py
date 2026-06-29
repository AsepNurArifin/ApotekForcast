import streamlit as st
import pandas as pd
import numpy as np
import io
import warnings
from pathlib import Path
from config import FEATURES, TARGET, Z
from dss_engine import load_assets, read_file, update_features, run_dss_fastmoving, run_dss_slowmoving
from ui_components import (
    render_workflow_progress, add_status_column, render_section_header,
    render_kpi_cards, render_sku_scorecard, render_plotly_trend, render_engine_donut
)
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

# ── LOAD STATIC ASSETS ──────────────────────────────────────────────────────
model, dataset, sku_eval, label, winsor_bounds, sku_class = load_assets()

# ── SIDEBAR & WORKFLOW PROGRESS ───────────────────────────────────────────
# Header/Branding for Sidebar
st.sidebar.markdown('<div class="sidebar-title">Alur Kerja Sistem</div>', unsafe_allow_html=True)

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


# ── MAIN FLOW ──────────────────────────────────────────────────────────────
# ── MAIN FLOW ──────────────────────────────────────────────────────────────
if file_transaksi and file_stok:
    df_new_raw = read_file(file_transaksi, label, file_type='transaksi')
    stok_df    = read_file(file_stok, label, file_type='stok')

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
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                filtered_df.to_excel(writer, index=False, sheet_name='Rekomendasi')
            excel_data = output.getvalue()

            st.download_button(
                label='📥 Download Hasil Rekomendasi (Excel)',
                data=excel_data,
                file_name='rekomendasi_pengadaan_apotek.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
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
            render_sku_scorecard(sku_sel, hasil_df, sku_class)

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
