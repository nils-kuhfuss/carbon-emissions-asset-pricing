###########################################################################
# Setup 
###########################################################################

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

df = pd.read_parquet('clean_data.parquet')

###########################################################################
# Compute BMG-Faktor
###########################################################################

def safe_tercile_sort(x):
    x_valid = x.dropna()

    if len(x_valid) < 3 or x_valid.nunique() < 3:
        return pd.Series(index=x.index, dtype='object')

    try:
        return (
            pd.qcut(x_valid, q=3, labels=['Green', 'Neutral', 'Brown'])
            .astype(str)
            .reindex(x.index)  
        )
    except ValueError:
        return pd.Series(index=x.index, dtype='object')

def compute_bmg(df, emission_metric):

    df = df.copy()
    df['Date'] = df['Date'].dt.to_timestamp() if isinstance(df['Date'].dtype, pd.PeriodDtype) else pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year

    annual_sort = (
        df[['ISIN', 'Year', 'Company Market Cap', emission_metric]]
        .drop_duplicates(subset=['ISIN', 'Year'])
        .dropna(subset=['Company Market Cap', emission_metric])
        .copy()
    )

    annual_sort['Size_Group'] = annual_sort.groupby('Year')['Company Market Cap'].transform(
        lambda x: np.where(x > x.median(), 'Big', 'Small')
    )

    annual_sort['Em_Group'] = annual_sort.groupby('Year')[emission_metric].transform(
        safe_tercile_sort
    )

    annual_sort = annual_sort.dropna(subset=['Em_Group'])

    annual_sort['valid_from']  = pd.to_datetime((annual_sort['Year'] + 1).astype(str) + '-07-01')
    annual_sort['valid_until'] = pd.to_datetime((annual_sort['Year'] + 2).astype(str) + '-06-30')

    df_dates = df[['ISIN', 'Date']].drop_duplicates()

    sort_assignments = df_dates.merge(
        annual_sort[['ISIN', 'valid_from', 'valid_until', 'Size_Group', 'Em_Group']],
        on='ISIN',
        how='left'
    )

    sort_assignments = sort_assignments[
        (sort_assignments['Date'] >= sort_assignments['valid_from']) &
        (sort_assignments['Date'] <= sort_assignments['valid_until'])
    ]

    df_bmg = df.merge(
        sort_assignments[['ISIN', 'Date', 'Size_Group', 'Em_Group']],
        on=['ISIN', 'Date'],
        how='left'
    )

    df_bmg_filtered = (
        df_bmg[df_bmg['Em_Group'].isin(['Brown', 'Green'])]
        .dropna(subset=['Excess Return', 'Company Market Cap'])
        .copy()
    )

    df_bmg_filtered['weighted_ret'] = (
        df_bmg_filtered['Excess Return'] * df_bmg_filtered['Company Market Cap']
    )

    grouped = df_bmg_filtered.groupby(['Date', 'Size_Group', 'Em_Group'], observed=True)

    portfolios = (
        (grouped['weighted_ret'].sum() / grouped['Company Market Cap'].sum())
        .unstack(['Size_Group', 'Em_Group'])
    )

    portfolios.columns = [f'{size}_{em}' for size, em in portfolios.columns]

    needed_cols = ['Small_Brown', 'Big_Brown', 'Small_Green', 'Big_Green']
    portfolios = portfolios.reindex(
        columns=[*portfolios.columns, *[c for c in needed_cols if c not in portfolios.columns]]
    )

    portfolios['BMG'] = (
        0.5 * (portfolios['Small_Brown'] + portfolios['Big_Brown']) -
        0.5 * (portfolios['Small_Green'] + portfolios['Big_Green'])
    )

    portfolios = portfolios.dropna(subset=needed_cols + ['BMG'])

    return portfolios.reset_index(), df_bmg

###########################################################################
# Anwendung & Robustheitstests BMG-Faktor (Absolut, Relativ, Changes)
###########################################################################

robustness_metrics = [
    'Scope_Total_Intensity',
    'Scope12_Intensity',  

    'Scope_Total',   
    'Scope12',

    'ScopeTotal_Changes',
    'Scope12_Changes', 
]

all_bmg_factors = {}

for metric in robustness_metrics:
    portfolios, df_bmg_metric = compute_bmg(df, metric)
    all_bmg_factors[metric] = portfolios.set_index('Date')['BMG']

df_bmg_comparison = pd.DataFrame(all_bmg_factors)

###########################################################################
# BMG - Faktor (Quintile, Terzil & verschiedene Branchen & market-weighted, equal-weighted))
###########################################################################

# ------------------------------------------------------------------------
# Hilfsfunktionen
# ------------------------------------------------------------------------

def safe_quantile_sort(x, n):
    x_valid = x.dropna()
    if len(x_valid) < n or x_valid.nunique() < n:
        return pd.Series(index=x.index, dtype='float')
    try:
        return (
            pd.qcut(x_valid, q=n, labels=list(range(1, n + 1)), duplicates='drop')
            .astype(float)
            .reindex(x.index)
        )
    except ValueError:
        return pd.Series(index=x.index, dtype='float')

def to_datetime_safe(series):
    if isinstance(series.dtype, pd.PeriodDtype):
        return series.dt.to_timestamp()
    return pd.to_datetime(series)

def nested_f_test(m_restricted, m_unrestricted, y, X_r, X_u):
    n      = len(y)
    p      = X_u.shape[1]
    q      = X_u.shape[1] - X_r.shape[1]
    f_stat = ((m_restricted.ssr - m_unrestricted.ssr) / q) / (m_unrestricted.ssr / (n - p))
    f_pval = 1 - stats.f.cdf(f_stat, dfn=q, dfd=n - p)
    return round(float(f_stat), 3), round(float(f_pval), 3)


# ------------------------------------------------------------------------
# Quantil-Portfolios bauen und BMG-Faktor laden
# ------------------------------------------------------------------------
# Görgen et al.: Sortierung nach Emissionsintensität, Rebalancing Juli t+1
# Haupt-Analyse:   Quintile (n=5), Equal-Weighted
# Robustheitstests: Terzile (n=3) EW / Quintile VW / Terzile VW

factors      = ['Mkt-RF_EUR', '5 SMB_EUR', '5 HML_EUR', '5 RMW_EUR', '5 CMA_EUR', 'WML_EUR']
emission_var = 'Scope_Total_Intensity'
bmg_series   = all_bmg_factors[emission_var]  

configs = [
    {'n': 5, 'weighting': 'equal', 'label': 'Quintile_EW'},   # Haupt-Analyse
    {'n': 3, 'weighting': 'equal', 'label': 'Terzile_EW'},    # Robustheit
    {'n': 5, 'weighting': 'value', 'label': 'Quintile_VW'},   # Robustheit
    {'n': 3, 'weighting': 'value', 'label': 'Terzile_VW'},    # Robustheit
]

def build_quantile_portfolios(df, emission_var, n, weighting):
    df = df.copy()
    df['Date'] = df['Date'].dt.to_timestamp() if isinstance(df['Date'].dtype, pd.PeriodDtype) else pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year

    annual = (
        df[['ISIN', 'Year', emission_var]]
        .drop_duplicates(subset=['ISIN', 'Year'])
        .dropna(subset=[emission_var])
        .copy()
    )

    annual['Bucket'] = annual.groupby('Year')[emission_var].transform(
        lambda x: safe_quantile_sort(x, n)
    )
    annual = annual.dropna(subset=['Bucket'])
    annual['Bucket'] = annual['Bucket'].astype('Int64')

    annual['valid_from']  = pd.to_datetime((annual['Year'] + 1).astype(str) + '-07-01')
    annual['valid_until'] = pd.to_datetime((annual['Year'] + 2).astype(str) + '-06-30')

    df_dates = df[['ISIN', 'Date']].drop_duplicates()

    assignments = df_dates.merge(
        annual[['ISIN', 'valid_from', 'valid_until', 'Bucket']], on='ISIN', how='left'
    )
    assignments = assignments[
        (assignments['Date'] >= assignments['valid_from']) &
        (assignments['Date'] <= assignments['valid_until'])
    ]

    df_sorted = df.merge(assignments[['ISIN', 'Date', 'Bucket']], on=['ISIN', 'Date'], how='left')
    df_sorted = df_sorted.dropna(subset=['Bucket', 'Excess Return'])

    if weighting == 'equal':
        port_returns = (
            df_sorted.groupby(['Date', 'Bucket'], observed=True)['Excess Return']
            .mean()
            .unstack('Bucket')
        )
    else:
        df_sorted = df_sorted.dropna(subset=['Company Market Cap']).copy()
        df_sorted['_vw'] = df_sorted['Excess Return'] * df_sorted['Company Market Cap']
        grp = df_sorted.groupby(['Date', 'Bucket'], observed=True)
        port_returns = (grp['_vw'].sum() / grp['Company Market Cap'].sum()).unstack('Bucket')

    port_returns.columns = [f'P{int(c)}' for c in port_returns.columns]
    port_returns[f'LS'] = port_returns[f'P{n}'] - port_returns['P1']  
    return port_returns.astype(float)

quantile_portfolios = {}

for cfg in configs:
    quantile_portfolios[cfg['label']] = build_quantile_portfolios(
        df, emission_var, n=cfg['n'], weighting=cfg['weighting']
    )

# ------------------------------------------------------------------------
# Regressionen: Quantil-Portfolios ~ FF5 + MOM + BMG
# ------------------------------------------------------------------------

def run_portfolio_regressions(port_returns, bmg_series, factors, df_factors, label):

    factors_ts = (
        df_factors[['Date'] + factors].drop_duplicates('Date').copy()
        .assign(Date=lambda x: to_datetime_safe(x['Date']))
        .set_index('Date')
    )

    reg_data = (
        port_returns
        .join(factors_ts)
        .join(bmg_series.rename('BMG'))
        .dropna()
    )

    lags             = int(len(reg_data) ** (1 / 3))
    factors_with_bmg = factors + ['BMG']
    results          = {}

    for port in port_returns.columns:
        if port not in reg_data.columns:
            continue

        y      = reg_data[port].astype(float)
        X_base = sm.add_constant(reg_data[factors].astype(float))
        m_base = sm.OLS(y, X_base).fit(cov_type='HAC', cov_kwds={'maxlags': lags})

        X_bmg  = sm.add_constant(reg_data[factors_with_bmg].astype(float))
        m_bmg  = sm.OLS(y, X_bmg).fit(cov_type='HAC', cov_kwds={'maxlags': lags})

        f_stat, f_pval = nested_f_test(m_base, m_bmg, y, X_base, X_bmg)

        results[port] = {
            'Alpha':           round(m_bmg.params['const'], 4),
            'p(Alpha)':        round(m_bmg.pvalues['const'], 3),
            'BMG Beta':        round(m_bmg.params.get('BMG', np.nan), 4),
            'p(BMG)':          round(m_bmg.pvalues.get('BMG', np.nan), 3),
            'R² Base':         round(m_base.rsquared, 3),
            'R² +BMG':         round(m_bmg.rsquared, 3),
            '∆R²':             round(m_bmg.rsquared - m_base.rsquared, 3),
            'F-stat':          f_stat,
            'p(F)':            f_pval,
            'N':               len(reg_data),
        }

    results_df = pd.DataFrame(results).T
    return results_df


quantile_results = {}

for cfg in configs:
    quantile_results[cfg['label']] = run_portfolio_regressions(
        port_returns = quantile_portfolios[cfg['label']],
        bmg_series   = bmg_series,
        factors      = factors,
        df_factors   = df,
        label        = cfg['label'],
    )

# ------------------------------------------------------------------------
# BMG auf Branchen-Portfolios laden
# ------------------------------------------------------------------------
# Für jeden Sektor: ein VW-Portfolio → Regression auf FF5 + MOM + BMG
# γ zeigt das Carbon-Risk-Exposure des Sektors

def build_industry_portfolios(df, industry_col='Industry', weighting='value'):
    df = df.copy()
    df['Date'] = df['Date'].dt.to_timestamp() if isinstance(df['Date'].dtype, pd.PeriodDtype) else pd.to_datetime(df['Date'])
    df = df.dropna(subset=[industry_col, 'Excess Return'])

    if weighting == 'value':
        df = df.dropna(subset=['Company Market Cap']).copy()
        df['_vw'] = df['Excess Return'] * df['Company Market Cap']
        grp = df.groupby(['Date', industry_col], observed=True)
        industry_returns = (grp['_vw'].sum() / grp['Company Market Cap'].sum()).unstack(industry_col)
    else:
        industry_returns = (
            df.groupby(['Date', industry_col], observed=True)['Excess Return']
            .mean()
            .unstack(industry_col)
        )

    return industry_returns.astype(float)


def run_industry_regressions(industry_returns, bmg_series, factors, df_factors, min_obs=20):

    factors_ts = (
        df_factors[['Date'] + factors].drop_duplicates('Date').copy()
        .assign(Date=lambda x: to_datetime_safe(x['Date']))
        .set_index('Date')
    )

    factors_with_bmg = factors + ['BMG']
    results          = {}

    for sector in industry_returns.columns:
        reg_data = (
            industry_returns[[sector]]
            .join(factors_ts)
            .join(bmg_series.rename('BMG'))
            .dropna()
        )

        if len(reg_data) < min_obs:
            continue

        y      = reg_data[sector].astype(float)
        lags   = int(len(reg_data) ** (1 / 3))
        X_base = sm.add_constant(reg_data[factors].astype(float))
        m_base = sm.OLS(y, X_base).fit(cov_type='HAC', cov_kwds={'maxlags': lags})

        X_bmg  = sm.add_constant(reg_data[factors_with_bmg].astype(float))
        m_bmg  = sm.OLS(y, X_bmg).fit(cov_type='HAC', cov_kwds={'maxlags': lags})

        f_stat, f_pval = nested_f_test(m_base, m_bmg, y, X_base, X_bmg)

        results[sector] = {
            'Alpha':          round(m_bmg.params['const'], 4),
            'p(Alpha)':       round(m_bmg.pvalues['const'], 3),
            'BMG Loading γ':  round(m_bmg.params.get('BMG', np.nan), 4),
            'p(γ)':           round(m_bmg.pvalues.get('BMG', np.nan), 3),
            'R² Base':        round(m_base.rsquared, 3),
            'R² +BMG':        round(m_bmg.rsquared, 3),
            '∆R²':            round(m_bmg.rsquared - m_base.rsquared, 3),
            'F-stat':         f_stat,
            'p(F)':           f_pval,
            'N':              len(reg_data),
        }

    results_df = (
        pd.DataFrame(results).T
        .sort_values('BMG Loading γ', ascending=False) 
    )

    return results_df

industry_returns = build_industry_portfolios(df, industry_col='Industry', weighting='value')

industry_results = run_industry_regressions(
    industry_returns = industry_returns,
    bmg_series       = bmg_series,
    factors          = factors,
    df_factors       = df,
)

###########################################################################
# Makrofaktor - CO2-Returns
###########################################################################

co2_factor = (
    df[['Date', 'EEX-EU CO2 Emissions']]
    .drop_duplicates('Date')
    .assign(Date=lambda x: to_datetime_safe(x['Date']))
    .sort_values('Date')
    .set_index('Date')
    .rename(columns={'EEX-EU CO2 Emissions': 'CO2_Ret'})
    .dropna()
)

# ------------------------------------------------------------------------
# Regression: FF5 + MOM vs. FF5 + MOM + CO2 (Nested F-Test)
# ------------------------------------------------------------------------

def run_co2_regressions(port_returns, factors, df_factors, label, min_obs=20):
    factors_ts = (
        df_factors[['Date'] + factors].drop_duplicates('Date').copy()
        .assign(Date=lambda x: to_datetime_safe(x['Date']))
        .set_index('Date')
        .join(co2_factor)
    )

    reg_data = port_returns.join(factors_ts).dropna()

    if len(reg_data) < min_obs:
        return pd.DataFrame()

    lags             = int(len(reg_data) ** (1 / 3))
    factors_with_co2 = factors + ['CO2_Ret']
    results          = {}

    for port in port_returns.columns:
        if port not in reg_data.columns:
            continue

        y      = reg_data[port].astype(float)
        X_base = sm.add_constant(reg_data[factors].astype(float))
        m_base = sm.OLS(y, X_base).fit(cov_type='HAC', cov_kwds={'maxlags': lags})

        X_co2  = sm.add_constant(reg_data[factors_with_co2].astype(float))
        m_co2  = sm.OLS(y, X_co2).fit(cov_type='HAC', cov_kwds={'maxlags': lags})

        f_stat, f_pval = nested_f_test(m_base, m_co2, y, X_base, X_co2)

        results[port] = {
            'Alpha':       round(m_co2.params['const'], 4),
            'p(Alpha)':    round(m_co2.pvalues['const'], 3),
            'β CO2':       round(m_co2.params.get('CO2_Ret', np.nan), 4),
            'p(CO2)':      round(m_co2.pvalues.get('CO2_Ret', np.nan), 3),
            'R² Base':     round(m_base.rsquared, 3),
            'R² +CO2':     round(m_co2.rsquared, 3),
            '∆R²':         round(m_co2.rsquared - m_base.rsquared, 3),
            'F-stat':      f_stat,
            'p(F)':        f_pval,
            'N':           len(reg_data),
        }

    results_df = pd.DataFrame(results).T
    return results_df

# ------------------------------------------------------------------------
# Anwendung
# ------------------------------------------------------------------------

co2_quantile_results = {}

for cfg in configs:
    co2_quantile_results[cfg['label']] = run_co2_regressions(
        port_returns = quantile_portfolios[cfg['label']],
        factors      = factors,
        df_factors   = df,
        label        = f"CO2 | {cfg['label']}",
    )

co2_industry_results = run_co2_regressions(
    port_returns = industry_returns,
    factors      = factors,
    df_factors   = df,
    label        = 'CO2 | Branchen (VW)',
)

co2_industry_results = co2_industry_results.sort_values('β CO2', ascending=False)

###########################################################################
# BMG - Faktor + CO2-Returns (Quintile, Terzil & verschiedene Branchen & market-weighted, equal-weighted))
###########################################################################

def run_bmg_co2_regressions(port_returns, bmg_series, factors, df_factors, label, min_obs=20):
    factors_ts = (
        df_factors[['Date'] + factors].drop_duplicates('Date').copy()
        .assign(Date=lambda x: to_datetime_safe(x['Date']))
        .set_index('Date')
        .join(co2_factor)
        .join(bmg_series.rename('BMG'))
    )

    reg_data = port_returns.join(factors_ts).dropna()

    if len(reg_data) < min_obs:
        return pd.DataFrame()

    lags                  = int(len(reg_data) ** (1 / 3))
    factors_with_bmg_co2  = factors + ['BMG', 'CO2_Ret']
    results               = {}

    for port in port_returns.columns:
        if port not in reg_data.columns:
            continue

        y      = reg_data[port].astype(float)
        X_base = sm.add_constant(reg_data[factors].astype(float))
        m_base = sm.OLS(y, X_base).fit(cov_type='HAC', cov_kwds={'maxlags': lags})

        X_full = sm.add_constant(reg_data[factors_with_bmg_co2].astype(float))
        m_full = sm.OLS(y, X_full).fit(cov_type='HAC', cov_kwds={'maxlags': lags})

        f_stat, f_pval = nested_f_test(m_base, m_full, y, X_base, X_full)

        results[port] = {
            'Alpha':    round(m_full.params['const'], 4),
            'p(Alpha)': round(m_full.pvalues['const'], 3),
            'β BMG':    round(m_full.params.get('BMG', np.nan), 4),
            'p(BMG)':   round(m_full.pvalues.get('BMG', np.nan), 3),
            'β CO2':    round(m_full.params.get('CO2_Ret', np.nan), 4),
            'p(CO2)':   round(m_full.pvalues.get('CO2_Ret', np.nan), 3),
            'R² Base':  round(m_base.rsquared, 3),
            'R² +Both': round(m_full.rsquared, 3),
            '∆R²':      round(m_full.rsquared - m_base.rsquared, 3),
            'F-stat':   f_stat,
            'p(F)':     f_pval,
            'N':        len(reg_data),
        }

    results_df = pd.DataFrame(results).T
    return results_df

# ------------------------------------------------------------------------
# Anwendung
# ------------------------------------------------------------------------

bmg_co2_quantile_results = {}

for cfg in configs:
    bmg_co2_quantile_results[cfg['label']] = run_bmg_co2_regressions(
        port_returns = quantile_portfolios[cfg['label']],
        bmg_series   = bmg_series,
        factors      = factors,
        df_factors   = df,
        label        = f"BMG + CO2 | {cfg['label']}",
    )

bmg_co2_industry_results = run_bmg_co2_regressions(
    port_returns = industry_returns,
    bmg_series   = bmg_series,
    factors      = factors,
    df_factors   = df,
    label        = 'BMG + CO2 | Branchen (VW)',
)

bmg_co2_industry_results = bmg_co2_industry_results.sort_values('β BMG', ascending=False)

###########################################################################
# Erweiterung 1 – Industry-Neutral BMG (Meyer 2022)
###########################################################################
# Unterschied zum Standard-BMG: Terzile werden INNERHALB jeder Branche
# gebildet → kein Short-Energie / Long-Tech Bias mehr.
# Size-Split bleibt global (Median über alle Unternehmen).

def compute_bmg_industry_neutral(df, emission_metric, industry_col='Industry'):

    df = df.copy()
    df['Date'] = df['Date'].dt.to_timestamp() if isinstance(df['Date'].dtype, pd.PeriodDtype) else pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year

    annual_sort = (
        df[['ISIN', 'Year', 'Company Market Cap', emission_metric, industry_col]]
        .drop_duplicates(subset=['ISIN', 'Year'])
        .dropna(subset=['Company Market Cap', emission_metric, industry_col])
        .copy()
    )

    # Size-Split: global (wie Standard-BMG)
    annual_sort['Size_Group'] = annual_sort.groupby('Year')['Company Market Cap'].transform(
        lambda x: np.where(x > x.median(), 'Big', 'Small')
    )

    # Emissionsgruppe: INNERHALB jeder Branche pro Jahr
    annual_sort['Em_Group'] = annual_sort.groupby(['Year', industry_col])[emission_metric].transform(
        safe_tercile_sort
    )

    annual_sort = annual_sort.dropna(subset=['Em_Group'])

    annual_sort['valid_from']  = pd.to_datetime((annual_sort['Year'] + 1).astype(str) + '-07-01')
    annual_sort['valid_until'] = pd.to_datetime((annual_sort['Year'] + 2).astype(str) + '-06-30')

    df_dates = df[['ISIN', 'Date']].drop_duplicates()

    sort_assignments = df_dates.merge(
        annual_sort[['ISIN', 'valid_from', 'valid_until', 'Size_Group', 'Em_Group']],
        on='ISIN', how='left'
    )
    sort_assignments = sort_assignments[
        (sort_assignments['Date'] >= sort_assignments['valid_from']) &
        (sort_assignments['Date'] <= sort_assignments['valid_until'])
    ]

    df_bmg = df.merge(
        sort_assignments[['ISIN', 'Date', 'Size_Group', 'Em_Group']],
        on=['ISIN', 'Date'], how='left'
    )

    df_bmg_filtered = (
        df_bmg[df_bmg['Em_Group'].isin(['Brown', 'Green'])]
        .dropna(subset=['Excess Return', 'Company Market Cap'])
        .copy()
    )

    df_bmg_filtered['weighted_ret'] = (
        df_bmg_filtered['Excess Return'] * df_bmg_filtered['Company Market Cap']
    )

    grouped = df_bmg_filtered.groupby(['Date', 'Size_Group', 'Em_Group'], observed=True)

    portfolios = (
        (grouped['weighted_ret'].sum() / grouped['Company Market Cap'].sum())
        .unstack(['Size_Group', 'Em_Group'])
    )

    portfolios.columns = [f'{size}_{em}' for size, em in portfolios.columns]

    needed_cols = ['Small_Brown', 'Big_Brown', 'Small_Green', 'Big_Green']
    portfolios = portfolios.reindex(
        columns=[*portfolios.columns, *[c for c in needed_cols if c not in portfolios.columns]]
    )

    portfolios['BMG_IN'] = (
        0.5 * (portfolios['Small_Brown'] + portfolios['Big_Brown']) -
        0.5 * (portfolios['Small_Green'] + portfolios['Big_Green'])
    )

    portfolios = portfolios.dropna(subset=needed_cols + ['BMG_IN'])

    return portfolios.reset_index()


# Anwendung für alle Metriken
all_bmg_industry_neutral = {}

for metric in robustness_metrics:
    port = compute_bmg_industry_neutral(df, metric)
    all_bmg_industry_neutral[metric] = port.set_index('Date')['BMG_IN']

df_bmg_in_comparison = pd.DataFrame(all_bmg_industry_neutral)


###########################################################################
# Erweiterung 2 – Spanning Regressions (Faktor-Einzigartigkeit)
###########################################################################
# Regrediert BMG auf alle anderen Faktoren.
# Ein signifikantes Alpha zeigt: BMG liefert Information die nicht durch
# bestehende Faktoren erklärt wird (Meyer 2022, S. 12).

def run_spanning_regression(bmg_series, df_factors, factors, label='BMG'):

    factors_ts = (
        df_factors[['Date'] + factors].drop_duplicates('Date').copy()
        .assign(Date=lambda x: to_datetime_safe(x['Date']))
        .set_index('Date')
    )

    reg_data = (
        bmg_series.rename('BMG')
        .to_frame()
        .join(factors_ts)
        .dropna()
    )

    y    = reg_data['BMG'].astype(float)
    X    = sm.add_constant(reg_data[factors].astype(float))
    lags = int(len(reg_data) ** (1 / 3))
    m    = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': lags})

    result = {
        'Alpha':         round(m.params['const'], 4),
        'p(Alpha)':      round(m.pvalues['const'], 3),
        't(Alpha)':      round(m.tvalues['const'], 3),
        'R²':            round(m.rsquared, 3),
        'Adj. R²':       round(m.rsquared_adj, 3),
        'N':             len(reg_data),
        'Unique?':       'Yes' if m.pvalues['const'] < 0.05 else 'No',
    }

    # Faktor-Betas
    for fac in factors:
        result[f'β {fac}'] = round(m.params[fac], 4)
        result[f'p {fac}'] = round(m.pvalues[fac], 3)

    return result, m


# Spanning für alle BMG-Varianten
spanning_results = {}

for metric, bmg_series in all_bmg_factors.items():
    result, _ = run_spanning_regression(bmg_series, df, factors, label=metric)
    spanning_results[metric] = result

# Zusätzlich: Industry-Neutral BMG
for metric, bmg_series in all_bmg_industry_neutral.items():
    result, _ = run_spanning_regression(bmg_series, df, factors, label=f'{metric}_IN')
    spanning_results[f'{metric}_IN'] = result

df_spanning = pd.DataFrame(spanning_results).T


###########################################################################
# Erweiterung 3 – Fama-MacBeth Regression (1973)
###########################################################################
# Prüft ob Carbon Risk eine bepreiste Risikoprämie im Querschnitt liefert.
#
# Schritt 1: Full-Sample Beta-Schätzung pro ISIN
#            R_i,t = α_i + β_i,BMG * BMG_t + Kontrollvariablen + ε_i,t
#
# Schritt 2: Monatliche Querschnittsregression
#            R_i,t = λ_0,t + λ_BMG,t * β_i,BMG + ε_i,t
#
# Risikoprämie λ = Zeitreihendurchschnitt der monatlichen λ_t
# t-Stat: Newey-West HAC auf die Zeitreihe der λ_t

def run_fama_macbeth(df, bmg_series, factors, emission_var,
                     min_obs=24):

    df_fm = df.copy()
    df_fm['Date'] = df_fm['Date'].dt.to_timestamp() if isinstance(df_fm['Date'].dtype, pd.PeriodDtype) else pd.to_datetime(df_fm['Date'])

    factors_ts = (
        df_fm[['Date'] + factors].drop_duplicates('Date').copy()
        .set_index('Date')
    )

    # --- Schritt 1: Beta-Schätzung pro ISIN (Full Sample) ---
    betas = {}
    factor_cols = factors + ['BMG']

    for isin, grp in df_fm.groupby('ISIN'):
        grp = grp.set_index('Date')[['Excess Return']].join(factors_ts).join(bmg_series.rename('BMG')).dropna()

        if len(grp) < min_obs:
            continue

        y = grp['Excess Return'].astype(float)
        X = sm.add_constant(grp[factor_cols].astype(float))
        m = sm.OLS(y, X).fit()

        betas[isin] = {
            'beta_BMG': m.params.get('BMG', np.nan),
            **{f'beta_{f}': m.params.get(f, np.nan) for f in factors}
        }

    df_betas = pd.DataFrame(betas).T
    df_betas.index.name = 'ISIN'

    # --- Schritt 2: Monatliche Querschnittsregressionen ---
    lambda_t = []

    for date, grp in df_fm.groupby('Date'):
        grp = grp.set_index('ISIN')[['Excess Return']].join(df_betas).dropna()

        if len(grp) < 10:   # mind. 10 Aktien pro Monat
            continue

        y_cs = grp['Excess Return'].astype(float)
        X_cs = sm.add_constant(grp[['beta_BMG']].astype(float))
        m_cs = sm.OLS(y_cs, X_cs).fit()

        lambda_t.append({
            'Date':         date,
            'lambda_0':     m_cs.params['const'],
            'lambda_BMG':   m_cs.params.get('beta_BMG', np.nan),
        })

    df_lambda = pd.DataFrame(lambda_t).set_index('Date')

    # --- Risikoprämie: Durchschnitt der λ_t mit Newey-West t-Stat ---
    lags = int(len(df_lambda) ** (1 / 3))
    results = {}

    for col in ['lambda_0', 'lambda_BMG']:
        y_lam = df_lambda[col].dropna().astype(float)
        X_lam = sm.add_constant(pd.Series(np.ones(len(y_lam)), index=y_lam.index, name='const'))
        m_lam = sm.OLS(y_lam, X_lam).fit(cov_type='HAC', cov_kwds={'maxlags': lags})

        results[col] = {
            'λ (mean)':  round(m_lam.params['const'], 4),
            't-stat':    round(m_lam.tvalues['const'], 3),
            'p-value':   round(m_lam.pvalues['const'], 3),
            'Sig.':      '***' if m_lam.pvalues['const'] < 0.01 else
                         '**'  if m_lam.pvalues['const'] < 0.05 else
                         '*'   if m_lam.pvalues['const'] < 0.10 else '',
            'N (months)': len(y_lam),
        }

    df_fmb_results = pd.DataFrame(results).T
    return df_fmb_results, df_lambda, df_betas


# Anwendung: Hauptmetrik
bmg_main = all_bmg_factors['Scope_Total_Intensity']

df_fmb_results, df_lambda, df_betas = run_fama_macbeth(
    df           = df,
    bmg_series   = bmg_main,
    factors      = factors,
    emission_var = 'Scope_Total_Intensity',
)

###########################################################################
# Erweiterung 4 – Emission-Weighted Quintile & Terzile
###########################################################################
# Gewichtung: Return_i × Emissionsintensität_i / Σ Emissionsintensität
# Unternehmen mit höherer Intensität bekommen mehr Gewicht.
# Methodisch: zeigt ob das Signal durch emissions-intensivere Firmen getrieben wird.

def build_quantile_portfolios_ew_emission(df, emission_var, n):

    df = df.copy()
    df['Date'] = df['Date'].dt.to_timestamp() if isinstance(df['Date'].dtype, pd.PeriodDtype) else pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year

    annual = (
        df[['ISIN', 'Year', emission_var]]
        .drop_duplicates(subset=['ISIN', 'Year'])
        .dropna(subset=[emission_var])
        .copy()
    )

    annual['Bucket'] = annual.groupby('Year')[emission_var].transform(
        lambda x: safe_quantile_sort(x, n)
    )
    annual = annual.dropna(subset=['Bucket'])
    annual['Bucket']      = annual['Bucket'].astype('Int64')
    annual['valid_from']  = pd.to_datetime((annual['Year'] + 1).astype(str) + '-07-01')
    annual['valid_until'] = pd.to_datetime((annual['Year'] + 2).astype(str) + '-06-30')

    df_dates    = df[['ISIN', 'Date']].drop_duplicates()
    assignments = df_dates.merge(
        annual[['ISIN', 'valid_from', 'valid_until', 'Bucket']], on='ISIN', how='left'
    )
    assignments = assignments[
        (assignments['Date'] >= assignments['valid_from']) &
        (assignments['Date'] <= assignments['valid_until'])
    ]

    df_sorted = (
        df.merge(assignments[['ISIN', 'Date', 'Bucket']], on=['ISIN', 'Date'], how='left')
          .dropna(subset=['Bucket', 'Excess Return', emission_var])
          .copy()
    )

    # Nur positive Emissionswerte als Gewicht sinnvoll
    df_sorted = df_sorted[df_sorted[emission_var] > 0].copy()
    df_sorted['_ew_num'] = df_sorted['Excess Return'] * df_sorted[emission_var]

    grp = df_sorted.groupby(['Date', 'Bucket'], observed=True)
    port_returns = (grp['_ew_num'].sum() / grp[emission_var].sum()).unstack('Bucket')

    port_returns.columns = [f'P{int(c)}' for c in port_returns.columns]
    port_returns['LS']   = port_returns[f'P{n}'] - port_returns['P1']
    return port_returns.astype(float)


# Emission-Weighted zu configs und quantile_portfolios hinzufügen
configs += [
    {'n': 5, 'weighting': 'emission', 'label': 'Quintile_EW_Emission'},
    {'n': 3, 'weighting': 'emission', 'label': 'Terzile_EW_Emission'},
]

quantile_portfolios['Quintile_EW_Emission'] = build_quantile_portfolios_ew_emission(
    df, emission_var, n=5
)
quantile_portfolios['Terzile_EW_Emission'] = build_quantile_portfolios_ew_emission(
    df, emission_var, n=3
)

# Regressionen für Emission-Weighted Portfolios
for label in ['Quintile_EW_Emission', 'Terzile_EW_Emission']:
    quantile_results[label] = run_portfolio_regressions(
        port_returns = quantile_portfolios[label],
        bmg_series   = bmg_series,
        factors      = factors,
        df_factors   = df,
        label        = label,
    )
    co2_quantile_results[label] = run_co2_regressions(
        port_returns = quantile_portfolios[label],
        factors      = factors,
        df_factors   = df,
        label        = f'CO2 | {label}',
    )
    bmg_co2_quantile_results[label] = run_bmg_co2_regressions(
        port_returns = quantile_portfolios[label],
        bmg_series   = bmg_series,
        factors      = factors,
        df_factors   = df,
        label        = f'BMG + CO2 | {label}',
    )