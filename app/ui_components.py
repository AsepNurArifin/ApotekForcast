import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


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


def add_status_column(df):
    """Add styled status badge column to dataframe."""
    df_display = df.copy()
    df_display['Status'] = df_display['Alert'].apply(
        lambda x: '🔴 PERLU ORDER' if x else '🟢 AMAN'
    )
    df_display = df_display.drop(columns=['Alert'])
    return df_display


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


def render_kpi_cards(hasil_df):
    total_sku  = len(hasil_df)
    alert_cnt  = int(hasil_df['Alert'].sum())
    safe_cnt   = total_sku - alert_cnt

    high_risk_cnt = int(((hasil_df['Alert']) & (hasil_df['Stok Aktual'] <= hasil_df['Safety Stock'])).sum())
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


def render_sku_scorecard(sku_sel, hasil_df, sku_class):
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

    fig.add_trace(go.Scatter(
        x=sku_hist['Tanggal'], y=sku_hist['Jumlah'],
        fill='tozeroy',
        fillcolor='rgba(37, 99, 235, 0.04)',
        line=dict(color='rgba(37, 99, 235, 0)', width=0),
        showlegend=False, hoverinfo='skip'
    ))

    fig.add_trace(go.Scatter(
        x=sku_hist['Tanggal'], y=sku_hist['Jumlah'],
        mode='lines+markers',
        name='Penjualan Mingguan',
        line=dict(color='#2563EB', width=2.5, shape='spline'),
        marker=dict(size=7, color='#2563EB',
                    line=dict(width=2, color='#FFFFFF')),
        hovertemplate='<b>%{x|%d %b %Y}</b><br>Jumlah: %{y}<extra></extra>'
    ))

    if sku_rop is not None:
        fig.add_hline(
            y=sku_rop, line_dash='dash', line_color='#DC2626', line_width=1.5,
            annotation_text=f'ROP ({sku_rop})',
            annotation_position='top right',
            annotation_font=dict(color='#DC2626', size=11)
        )

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
