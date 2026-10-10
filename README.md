# Carbon Emissions and European Equity Returns

An empirical asset-pricing study of company-level carbon emissions and European carbon-allowance futures, based on my master's thesis at Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU).

Author: Nils Kuhfuß
Degree: M.Sc. Industrial Engineering and Management
Chair: Finance and Banking
Supervision: Prof. Dr. Hendrik Scholz and Anja Stiller
Submitted: 12 June 2026
Thesis language: German; this README provides an English research summary.

Research question
Do carbon-emissions characteristics and exposure to carbon-price movements help explain European equity returns beyond conventional asset-pricing factors?
The thesis examines two complementary dimensions:
- Company-level emissions: absolute emissions, emissions intensity relative to revenue, and changes in emissions.
- Carbon-market exposure: monthly returns on European Union Allowance (EUA) futures.
The analysis focuses on factor exposures and incremental explanatory power. It does not establish a causal effect of emissions on returns or demonstrate a profitable trading strategy.
Data
The study covers monthly equity returns from January 2016 to December 2025, using historical STOXX Europe 600 membership to construct the investment universe. Excluding financial firms leaves an initial universe of 770 companies, before further data-availability filters. The number of firms and observations therefore varies across analyses.
Component	Source and scope
Equity returns, revenue, market capitalisation and exchange rates	LSEG Workspace / Datastream
Corporate emissions	Scope 1 and Scope 2 emissions from LSEG Workspace
Conventional factors	Fama–French five factors plus momentum, using the Kenneth French Data Library
Carbon-market factor	Monthly returns on European EUA futures

Returns and conventional factors are aligned in euros. Scope 3 emissions are excluded because of coverage and consistency limitations. Carbon-offset data are explored, but the core analysis focuses on gross emissions.
Methodology
1. Prepare the data: align company and return histories, apply currency conversions, and winsorise emissions variables and related metrics at the 1st and 99th percentiles.
2. Account for reporting delays: apply a six-month lag to annual emissions information and rebalance portfolios annually in July to reduce look-ahead risk.
3. Construct carbon factors: build Brown-Minus-Green (BMG) long–short factors using size and emissions-based sorts, including industry-neutral variants.
4. Evaluate factor contribution: use spanning regressions and portfolio time-series regressions against a six-factor benchmark comprising the Fama–French five factors and momentum. Statistical inference uses Newey–West heteroskedasticity- and autocorrelation-consistent standard errors.
5. Assess robustness: compare value-, equal- and emissions-weighted portfolios, examine industry portfolios, and analyse the 2021–2025 subperiod.
The empirical work was conducted using Python and Excel.

Main findings
- Emissions intensity provides the clearest additional explanatory contribution. Low-emissions-intensity portfolios generally exhibit negative exposure to the corresponding BMG factor, while high-emissions-intensity portfolios exhibit positive exposure.
- Emissions changes provide weaker and less stable evidence. Their contribution depends more strongly on portfolio construction and sample choice.
- EUA futures capture a distinct source of return variation, but their additional explanatory contribution to equity portfolios is limited and less robust than that of the emissions-intensity factor.
- In the specification combining the six-factor benchmark, emissions-intensity and emissions-change factors, and EUA futures returns, adjusted R² for the high-minus-low emissions-intensity portfolio reaches 79.55%, an increase of 31.73 percentage points over the benchmark (Table 10, thesis p. 41).
These are in-sample explanatory results. Higher adjusted R² does not establish forecasting ability, out-of-sample alpha or an independently identified carbon-risk premium.

Limitations
Emissions reporting is incomplete and uneven across firms and years, and Scope 1 and Scope 2 do not capture the full emissions footprint of every business. Historical index membership is unavailable before June 2016 in the source used, requiring a static composition for the first six months. Some factor specifications have shorter histories than the full return sample.
Sector composition, firm size, portfolio weighting and major macroeconomic events may influence the results. EUA futures also reflect energy prices, economic conditions and other influences, so their returns are not a pure measure of regulatory shocks.
Repository contents and reproducibility
Publication status: The thesis is complete; the PDF and analysis scripts are being prepared for upload.

Intended location	Contents
thesis/master_thesis.pdf	Full master's thesis in German
code/	Python analysis scripts and any accompanying notebooks
README.md	English research summary


The underlying LSEG / Datastream datasets are not distributed here. Reproducing the empirical results requires appropriate access to the original data. Dependency versions, input-file specifications and execution order will be documented once the scripts have been organised for publication; an end-to-end reproduction workflow has not yet been verified for this repository.

Citation: Kuhfuß, N. (2026). CO2-Emissionen und Aktienperformance: Eine exemplarische empirische Analyse für den europäischen Kapitalmarkt. Master's thesis, Friedrich-Alexander-Universität Erlangen-Nürnberg, Chair of Finance and Banking.
