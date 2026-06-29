import pandas as pd
import numpy as np
import joblib
import pickle
import streamlit as st
from pathlib import Path
from config import FEATURES, TARGET, Z, RAMADAN_START, RAMADAN_END


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


def read_file(uploaded_file, label, file_type='transaksi'):
    name = uploaded_file.name.lower()

    if file_type == 'stok':
        if name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, header=11, low_memory=False)
        else:
            df = pd.read_excel(uploaded_file, header=11)

        df = df[['SKU', 'Stok Total']].dropna(subset=['SKU'])
        df = df.rename(columns={'SKU': 'Kode Produk', 'Stok Total': 'Stok_Aktual'})
        df['Stok_Aktual'] = pd.to_numeric(df['Stok_Aktual'], errors='coerce').fillna(0)
        return df

    else:
        if name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, header=0, low_memory=False)
        else:
            df = pd.read_excel(uploaded_file, header=0)

        expected_cols = {'Tanggal Transaksi', 'Kode Produk', 'Jumlah'}
        if not expected_cols.issubset(set(df.columns)):
            uploaded_file.seek(0)
            if name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, header=9, low_memory=False)
            else:
                df = pd.read_excel(uploaded_file, header=9)

            if 'Tanggal' in df.columns and 'Tanggal Transaksi' not in df.columns:
                df = df.rename(columns={'Tanggal': 'Tanggal Transaksi'})

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

        if 'Jumlah' in df.columns:
            df['Jumlah'] = pd.to_numeric(df['Jumlah'], errors='coerce').fillna(0)

        return df


def update_features(df_new_raw, dataset_hist, label, winsor_bounds):
    label['cat_norm'] = label['Draf_Kategori'].str.strip().str.lower()
    obat_skus = set(label[label['cat_norm'] == 'obat']['SKU'].dropna().unique())

    bulan_id = {
        'Jan': 'Jan', 'Feb': 'Feb', 'Mar': 'Mar', 'Apr': 'Apr',
        'Mei': 'May', 'Jun': 'Jun', 'Jul': 'Jul', 'Agt': 'Aug',
        'Agu': 'Aug', 'Sep': 'Sep', 'Okt': 'Oct', 'Nov': 'Nov',
        'Des': 'Dec',
    }
    tanggal_raw = df_new_raw['Tanggal Transaksi'].astype(str)
    tanggal_raw = tanggal_raw.str.replace(r'\s*pukul\s+[\d.:]+', '', regex=True)
    for id_bln, en_bln in bulan_id.items():
        tanggal_raw = tanggal_raw.str.replace(id_bln, en_bln, regex=False)
    df_new_raw['Tanggal Transaksi'] = pd.to_datetime(
        tanggal_raw, dayfirst=True, errors='coerce'
    )
    df_new_raw = df_new_raw.dropna(subset=['Tanggal Transaksi'])

    df_obat = df_new_raw[df_new_raw['Kode Produk'].isin(obat_skus)].copy()

    if df_obat.empty:
        st.error("Tidak ada transaksi obat ditemukan di file upload.")
        return pd.DataFrame()

    df_obat['week'] = df_obat['Tanggal Transaksi'].dt.to_period('W-MON')
    max_date = df_obat['Tanggal Transaksi'].max()
    if max_date.dayofweek != 6:
        minggu_terakhir = df_obat['week'].max()
        df_obat = df_obat[df_obat['week'] < minggu_terakhir]
        st.warning(
            f"⚠️ Data terakhir ({max_date.strftime('%d %b %Y')}) belum genap "
            f"7 hari (Senin–Minggu). Minggu yang belum selesai diabaikan otomatis."
        )

    if df_obat.empty:
        st.error("Tidak ada minggu lengkap (Senin–Minggu) di data upload.")
        return pd.DataFrame()

    weekly_new = df_obat.groupby(['Kode Produk','week'])['Jumlah'].sum().reset_index()
    weekly_new['Tanggal'] = weekly_new['week'].dt.start_time
    max_hist = dataset_hist['Tanggal'].max()
    max_upload = weekly_new['Tanggal'].max() if not weekly_new.empty else None

    if max_upload is not None and max_upload <= max_hist:
        st.warning(
            f"⚠️ Data transaksi yang diupload ({max_upload.strftime('%d %b %Y')}) "
            f"tidak lebih baru dari histori sistem ({max_hist.strftime('%d %b %Y')}). "
            f"Prediksi tetap dijalankan menggunakan histori yang ada."
        )
    weekly_new = weekly_new.drop(columns='week')

    hist     = dataset_hist[['Kode Produk','Tanggal','Jumlah']].copy()
    combined = pd.concat([hist, weekly_new], ignore_index=True)
    combined = combined.drop_duplicates(subset=['Kode Produk','Tanggal'])
    combined = combined.sort_values(['Kode Produk','Tanggal']).reset_index(drop=True)

    combined['Rata_Historis_SKU'] = combined.groupby(
        'Kode Produk')['Jumlah'].transform('mean')
    combined['Lag_1'] = combined.groupby('Kode Produk')['Jumlah'].shift(1)
    combined['Lag_2'] = combined.groupby('Kode Produk')['Jumlah'].shift(2)
    combined['Lag_3'] = combined.groupby('Kode Produk')['Jumlah'].shift(3)
    combined['Lag_4'] = combined.groupby('Kode Produk')['Jumlah'].shift(4)
    combined['Rolling_Mean_4'] = combined.groupby('Kode Produk')['Jumlah'].transform(
        lambda x: x.shift(1).rolling(4).mean()
    )
    combined['Rolling_Mean_2'] = combined.groupby('Kode Produk')['Jumlah'].transform(
        lambda x: x.shift(1).rolling(2).mean()
    )
    combined['Rolling_Std_4'] = combined.groupby('Kode Produk')['Jumlah'].transform(
        lambda x: x.shift(1).rolling(4).std()
    )
    combined['Bulan']    = combined['Tanggal'].dt.month
    combined['Pekan_Ke'] = combined['Tanggal'].dt.isocalendar().week.astype(int)
    combined['Is_Ramadan'] = combined['Tanggal'].between(RAMADAN_START, RAMADAN_END).astype(int)

    latest = combined.groupby('Kode Produk').last().reset_index()

    max_date = combined['Tanggal'].max()
    next_week = max_date + pd.Timedelta(days=7)

    pred_rows = latest.copy()
    pred_rows['Tanggal'] = next_week
    pred_rows['Jumlah'] = np.nan

    pred_rows['Lag_1'] = latest['Jumlah'].values
    pred_rows['Lag_2'] = latest['Lag_1'].values
    pred_rows['Lag_3'] = latest['Lag_2'].values
    pred_rows['Lag_4'] = latest['Lag_3'].values

    pred_rows['Rolling_Mean_2'] = (pred_rows['Lag_1'] + pred_rows['Lag_2']) / 2
    pred_rows['Rolling_Mean_4'] = (
        pred_rows['Lag_1'] + pred_rows['Lag_2'] +
        pred_rows['Lag_3'] + pred_rows['Lag_4']
    ) / 4
    pred_rows['Rolling_Std_4'] = np.std([
        pred_rows['Lag_1'].values,
        pred_rows['Lag_2'].values,
        pred_rows['Lag_3'].values,
        pred_rows['Lag_4'].values
    ], axis=0)

    pred_rows['Bulan']    = next_week.month
    pred_rows['Pekan_Ke'] = next_week.isocalendar().week
    pred_rows['Is_Ramadan'] = int(
        RAMADAN_START <= next_week <= RAMADAN_END
    )

    pred_rows = pred_rows.dropna(subset=FEATURES)

    for col in ['Lag_1','Lag_2','Lag_3','Lag_4',
                'Rolling_Mean_2','Rolling_Mean_4','Rolling_Std_4']:
        if col in winsor_bounds:
            pred_rows[col] = pred_rows[col].astype(float)
            for sku in pred_rows['Kode Produk']:
                if sku in winsor_bounds[col]:
                    ub = winsor_bounds[col][sku]
                    mask = (pred_rows['Kode Produk'] == sku) & (pred_rows[col] > ub)
                    pred_rows.loc[mask, col] = ub

    return pred_rows


def run_dss_fastmoving(latest_features, sku_eval, model, stok_df):
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
        order = int(np.ceil(max(0, rop - stok))) if order > 0 else 0
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
    label['cat_norm'] = label['Draf_Kategori'].str.strip().str.lower()
    obat_skus = set(label[label['cat_norm'] == 'obat']['SKU'].dropna().unique())

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

    df_slow['week'] = df_slow['Tanggal Transaksi'].dt.to_period('W-MON')
    weekly = df_slow.groupby(['Kode Produk','week'])['Jumlah'].sum().reset_index()
    skus_with_data = set(weekly['Kode Produk'].unique()) if not weekly.empty else set()

    results = []

    for sku, grp in weekly.groupby('Kode Produk'):
        grp = grp.sort_values('week')
        last4 = grp['Jumlah'].tail(4).values
        pred  = round(float(np.mean(last4)), 2) if len(last4) > 0 else 0.0

        std   = float(np.std(last4)) if len(last4) > 1 else pred * 0.5
        ss    = round(Z * std, 2)
        rop   = round(pred + ss, 2)

        stok_row = stok_df[stok_df['Kode Produk'] == sku]
        stok     = float(stok_row['Stok_Aktual'].values[0]) \
                   if not stok_row.empty else 0.0
        order = round(max(0, rop - stok), 2)
        order = int(np.ceil(max(0, rop - stok)))
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

    skus_no_data = slow_skus - skus_with_data
    for sku in skus_no_data:
        stok_row = stok_df[stok_df['Kode Produk'] == sku]
        stok = float(stok_row['Stok_Aktual'].values[0]) if not stok_row.empty else 0.0

        pred = 0.0
        std  = 1.0
        ss   = round(Z * std, 2)
        rop  = round(pred + ss, 2)
        order = round(max(0, rop - stok), 2)
        order = int(np.ceil(max(0, rop - stok)))
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
