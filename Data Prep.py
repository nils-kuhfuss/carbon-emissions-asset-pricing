###########################################################################
# Setup
###########################################################################

import numpy as np
import pandas as pd

# Pfad zur Excel-Datei
path = ()

df_factors = pd.read_excel(path, sheet_name="factors_monthly")
df_other = pd.read_excel(path, sheet_name="carbonprice_usdeur_or")
df_stocks = pd.read_excel(path, sheet_name="stock_price")
df_carbon = pd.read_excel(path, sheet_name="carbon_panel")
df_indexmember = pd.read_excel(path, sheet_name="index_membership")
df_mcap = pd.read_excel(path, sheet_name="market_cap")
df_netsales = pd.read_excel(path, sheet_name="net_sales")

###########################################################################
# Data Prep
###########################################################################

# ------------------------------------------------------------------------
# Datumsspalte in allen Dataframes fixen und als Index setzen
# ------------------------------------------------------------------------

df_factors['Date'] = pd.to_datetime(df_factors['Date'], format='%Y%m')
df_factors.set_index('Date', inplace=True)
df_factors.index = df_factors.index.to_period('M')

df_other['Date'] = pd.to_datetime(df_other['Date'])
df_other.set_index('Date', inplace=True)
df_other.index = df_other.index.to_period('M')

df_stocks['Date'] = pd.to_datetime(df_stocks['Date'])
df_stocks.set_index('Date', inplace=True)
df_stocks.index = df_stocks.index.to_period('M')

# ------------------------------------------------------------------------
# Prep - df_other
# ------------------------------------------------------------------------

# Overnight-Rate: EONIA bis Sep-2019, danach ESTR
df_other['Overnight'] = df_other['EONIA 1M'].where(
    df_other.index < '2019-10',
    df_other['ESTR 1M']
)

df_other['RF_EUR'] = df_other[['EURIBOR 1M', 'Overnight']].min(axis=1)

# Annualisiert → monatlich, Prozent → Dezimal
df_other['RF_EUR'] = (1 + df_other['RF_EUR'] / 100) ** (1 / 12) - 1

drop_cols = ['Overnight', 'EURIBOR 1M', 'ESTR 1M', 'EONIA 1M']
df_other.drop(columns=drop_cols, inplace=True)

df_other[['EEX-EU CO2 Emissions', 'USD/EUR']] = df_other[['EEX-EU CO2 Emissions', 'USD/EUR']].pct_change()

# ------------------------------------------------------------------------
# Prep - df_factors
# ------------------------------------------------------------------------

# Einheit anpassen
df_factors = df_factors / 100

# Faktoren in EUR umrechnen
panel = df_other.join(df_factors)
panel['Mkt-RF_EUR'] = (1 / (1 + panel['USD/EUR'] )) * (1 + panel['Mkt-RF'] + panel['RF']) - 1 - panel['RF_EUR']

ls_factors = ['5 SMB', '5 HML', '5 RMW', '5 CMA', 'WML']
for factor in ls_factors:
    if factor in panel.columns:
        panel[f'{factor}_EUR'] = (1 / (1 + panel['USD/EUR'] )) * panel[factor]

usd_cols = ['USD/EUR', 'Mkt-RF', 'RF'] + ls_factors
panel.drop(columns=[c for c in usd_cols if c in panel.columns], inplace=True)

# ------------------------------------------------------------------------
# Prep - df_stocks
# ------------------------------------------------------------------------

df_stocks = df_stocks.sort_values(['ISIN','Date'])

df_stocks['Monthly_Returns'] = df_stocks.groupby('ISIN')['Price Close'].pct_change(fill_method=None)

# ------------------------------------------------------------------------
# Prep - df_carbon
# ------------------------------------------------------------------------

carbon_long = df_carbon.melt(
    id_vars=['ISIN'],
    var_name='col',
    value_name='value'
)
carbon_long[['Emission_Type', 'Year']] = carbon_long['col'].str.split(' ', expand=True)
carbon_long['Year'] = carbon_long['Year'].astype(int)
carbon_long.drop(columns='col', inplace=True)

carbon_wide = (carbon_long
                .pivot_table(index=['ISIN', 'Year'],
                            columns='Emission_Type',
                            values='value')
                .reset_index())
carbon_wide.columns.name = None
carbon_wide.rename(columns={
    'S1': 'Scope1',
    'S2': 'Scope2',
    'O/C': 'Offset/Credit'
}, inplace=True)

# ------------------------------------------------------------------------
# Prep - df_indexmember
# ------------------------------------------------------------------------

df_indexmember = df_indexmember[df_indexmember['Industry'] != 'Financials']

# ------------------------------------------------------------------------
# Prep - netsales
# ------------------------------------------------------------------------

df_netsales = df_netsales.sort_values(by=['ISIN', 'Year'])

df_netsales['Revenue'] = df_netsales['Revenue'] / 1000000

###########################################################################
# Join aller Dataframes
###########################################################################

df_merged = (panel.merge(df_stocks, on=['Date'], how='left'))

df_merged['Year'] = df_merged.index.year
df_merged = df_merged.reset_index()

df_final = (df_merged
              .merge(carbon_wide, on=['ISIN', 'Year'], how='left')
              .merge(df_indexmember, on=['ISIN'], how='left')
              .merge(df_mcap, on=['ISIN', 'Year'], how='left')
              .merge(df_netsales, on=['ISIN', 'Year'], how='left'))

df_final.drop(columns=['Year'], inplace=True)

df_final = df_final[df_final['Date'] != '2015-12']

df_final = df_final[df_final['Industry'].notna()]

# print(df_final.head())

# ------------------------------------------------------------------------
# Berechnung verschiedener Variablen
# ------------------------------------------------------------------------

df_final = df_final.sort_values(by=['ISIN', 'Date'])
df_final[['Scope1', 'Scope2', 'Offset/Credit']] = df_final[['Scope1', 'Scope2', 'Offset/Credit']].fillna(0)

df_final['Excess Return']        = df_final['Monthly_Returns'] - (df_final['RF_EUR'] / 12) 

# Betrachtung relative Emissionen

df_final['Scope1_Intensity']      = df_final['Scope1'] / df_final['Revenue']
df_final['Scope2_Intensity']      = df_final['Scope2'] / df_final['Revenue']
df_final['OC_Intensity']          = df_final['Offset/Credit'] / df_final['Revenue']
df_final['Scope12']               = df_final['Scope1'] + df_final['Scope2']
df_final['Scope12_Intensity']     = df_final['Scope12'] / df_final['Revenue']
df_final['Scope_Total']           = df_final['Scope1'] + df_final['Scope2'] - df_final['Offset/Credit']
df_final['Scope_Total_Intensity'] = df_final['Scope_Total'] / df_final['Revenue']

# Betrachtung Emission Changes 

df_final[['Scope_Total','Scope12']] = df_final[['Scope_Total','Scope12']].replace(0, np.nan)

df_final['ScopeTotal_Changes'] = df_final.groupby('ISIN')['Scope_Total'].diff()
df_final['Scope12_Changes'] = df_final.groupby('ISIN')['Scope12'].diff()

# print(df_final.columns.tolist())

###########################################################################
# Zwischenspeichern der sauberen Daten für Faktor-Konstruktion
###########################################################################

df_final.to_parquet('clean_data.parquet')

# print(df_final[['Date','Scope1','Scope2','Company Market Cap','Scope_Total', 'Scope1_Intensity', 'Scope2_Intensity', 'Scope_Total_Intensity']])
# print(df_final[['Scope1','Scope2','Company Market Cap','Scope_Total', 'Scope1_Intensity', 'Scope2_Intensity', 'Scope_Total_Intensity']].describe())