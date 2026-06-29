import pandas as pd

FEATURES = ['Lag_1', 'Lag_2', 'Lag_3', 'Lag_4',
            'Rolling_Mean_2', 'Rolling_Mean_4', 'Rolling_Std_4',
            'Bulan', 'Pekan_Ke', 'Rata_Historis_SKU', 'Is_Ramadan']
TARGET = 'Jumlah'
Z = 1.65
RAMADAN_START = pd.Timestamp('2026-03-10')
RAMADAN_END = pd.Timestamp('2026-04-08')
