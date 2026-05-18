BETA_NEUTRALIZE_POSITIONS = True
BETA_NEUTRAL_LOOKBACK = 126
BETA_NEUTRAL_MIN_PERIODS = 40
BETA_NEUTRAL_EPS = 1e-10


def _coerce_signal(signal_df, close_df):
    """Coerce candidate output into a numeric frame aligned to prices."""
    sig = signal_df.copy()
    sig = sig.reindex(index=close_df.index, columns=close_df.columns)
    sig = sig.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return sig.clip(-1.0, 1.0)


def signal_quality(signal_df, close_df):
    """Diagnostics on the raw signal before market-neutral normalization."""
    raw = _coerce_signal(signal_df, close_df)
    demeaned = raw.sub(raw.mean(axis=1), axis=0)
    gross = demeaned.abs().sum(axis=1)

    return {
        "raw_cs_std": float(raw.std(axis=1).mean()),
        "raw_abs_row_mean": float(raw.mean(axis=1).abs().mean()),
        "raw_long_frac": float((raw > 0).mean().mean()),
        "raw_short_frac": float((raw < 0).mean().mean()),
        "signal_activity": float((gross > 1e-8).mean()),
        "gross_exposure": float(gross.mean()),
    }


def _unit_gross_positions(centered):
    gross = centered.abs().sum(axis=1).replace(0.0, np.nan)
    return centered.div(gross, axis=0).fillna(0.0)


def _rolling_asset_betas(close_df, lookback=BETA_NEUTRAL_LOOKBACK):
    """Estimate per-asset beta using only returns available before each signal date."""
    ret = close_df.pct_change().fillna(0.0)
    bench = ret.mean(axis=1)
    min_periods = min(BETA_NEUTRAL_MIN_PERIODS, int(lookback))
    asset_mean = ret.rolling(lookback, min_periods=min_periods).mean()
    bench_mean = bench.rolling(lookback, min_periods=min_periods).mean()
    cov = ret.mul(bench, axis=0).rolling(lookback, min_periods=min_periods).mean()
    cov = cov - asset_mean.mul(bench_mean, axis=0)
    var = bench.pow(2).rolling(lookback, min_periods=min_periods).mean() - bench_mean.pow(2)
    betas = cov.div(var.replace(0.0, np.nan), axis=0)
    return betas.shift(1).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5.0, 5.0)


def _beta_neutralize_positions(pos, close_df):
    """Project daily weights away from lagged equal-weight market beta."""
    centered = pos.sub(pos.mean(axis=1), axis=0)
    betas = _rolling_asset_betas(close_df).reindex_like(centered).fillna(0.0)
    beta_exposure = (centered * betas).sum(axis=1)
    beta_norm = betas.pow(2).sum(axis=1).replace(0.0, np.nan)
    hedge = betas.mul(beta_exposure.div(beta_norm).fillna(0.0), axis=0)
    adjusted = centered - hedge
    adjusted = adjusted.sub(adjusted.mean(axis=1), axis=0)
    return _unit_gross_positions(adjusted)


def _market_neutral_positions(signal_df, close_df, beta_neutralize=BETA_NEUTRALIZE_POSITIONS):
    """Convert raw scores to dollar-neutral unit-gross weights per date."""
    raw = _coerce_signal(signal_df, close_df)
    centered = raw.sub(raw.mean(axis=1), axis=0)
    pos = _unit_gross_positions(centered)
    if beta_neutralize:
        return _beta_neutralize_positions(pos, close_df)
    return pos


def _series_metrics(series):
    series = series.dropna()
    ann_ret = float(series.mean() * 252)
    ann_vol = float(series.std() * np.sqrt(252))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    equity = (1 + series).cumprod()
    max_dd = float((equity / equity.cummax() - 1).min()) if len(equity) else 0.0
    hit = float((series > 0).mean()) if len(series) else 0.0
    return ann_ret, ann_vol, sharpe, max_dd, hit, equity


def _annual_consistency(series):
    series = series.dropna()
    annual_sharpes = []
    if len(series):
        for yr in sorted(series.index.year.unique()):
            part = series[series.index.year == yr]
            if len(part) < 50:
                continue
            vol = float(part.std() * np.sqrt(252))
            annual_sharpes.append(float(part.mean() * 252 / vol) if vol > 0 else 0.0)
    consistency = float(sum(1 for s in annual_sharpes if s > 0) / max(len(annual_sharpes), 1))
    min_annual = float(min(annual_sharpes)) if annual_sharpes else 0.0
    return consistency, annual_sharpes, min_annual


def backtest(signal_df, close_df, cost_bps=1.0):
    # Enforce the competition objective in the harness itself: active,
    # cross-sectional, market-neutral predictions. This closes the loophole
    # where all-long signals scored as equity beta.
    quality = signal_quality(signal_df, close_df)
    raw_sig = _market_neutral_positions(signal_df, close_df, beta_neutralize=False)
    sig = _market_neutral_positions(signal_df, close_df)
    pos = sig.shift(1).fillna(0.0)
    raw_pos = raw_sig.shift(1).fillna(0.0)
    ret = close_df.pct_change().fillna(0.0)

    raw_gross = (raw_pos * ret).sum(axis=1)
    gross = (pos * ret).sum(axis=1)
    turnover = 0.5 * pos.diff().abs().sum(axis=1).fillna(0.0)
    cost = turnover * (cost_bps / 10000.0)
    net = gross - cost

    ann_ret, ann_vol, sharpe, max_dd, hit, equity = _series_metrics(net)
    consistency, annual_sharpes, min_annual = _annual_consistency(net)
    gross_ann_ret, gross_ann_vol, gross_sharpe, gross_dd, gross_hit, _ = _series_metrics(gross)

    inv_gross = -gross
    inv_net = inv_gross - cost
    inv_ann_ret, inv_ann_vol, inv_sharpe, inv_dd, inv_hit, inv_equity = _series_metrics(inv_net)
    inv_consistency, inv_annual_sharpes, inv_min_annual = _annual_consistency(inv_net)

    bench = ret.mean(axis=1)
    spread = net - bench
    spread_ann_ret, spread_ann_vol, spread_sharpe, spread_dd, spread_hit, spread_equity = _series_metrics(spread)

    if bench.std() > 0 and net.std() > 0:
        beta = float(np.cov(net, bench)[0, 1] / np.var(bench))
    else:
        beta = 0.0
    if bench.std() > 0 and raw_gross.std() > 0:
        beta_raw = float(np.cov(raw_gross, bench)[0, 1] / np.var(bench))
    else:
        beta_raw = 0.0
    if bench.std() > 0 and inv_net.std() > 0:
        beta_inv = float(np.cov(inv_net, bench)[0, 1] / np.var(bench))
    else:
        beta_inv = 0.0

    out = {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sharpe_raw": gross_sharpe,
        "max_dd": max_dd,
        "hit_rate": hit,
        "avg_turnover": float(turnover.mean()),
        "beta": beta,
        "beta_neutralized": bool(BETA_NEUTRALIZE_POSITIONS),
        "beta_raw": beta_raw,
        "beta_neutralized_value": beta,
        "beta_reduction": float(abs(beta_raw) - abs(beta)),
        "consistency": consistency,
        "annual_sharpes": annual_sharpes,
        "min_annual": min_annual,
        "equity": equity,
        "gross_ann_return": gross_ann_ret,
        "gross_ann_vol": gross_ann_vol,
        "gross_max_dd": gross_dd,
        "gross_hit_rate": gross_hit,
        "inverted_ann_return": inv_ann_ret,
        "inverted_ann_vol": inv_ann_vol,
        "inverted_sharpe": inv_sharpe,
        "inverted_max_dd": inv_dd,
        "inverted_hit_rate": inv_hit,
        "inverted_equity": inv_equity,
        "inverted_beta": beta_inv,
        "inverted_consistency": inv_consistency,
        "inverted_annual_sharpes": inv_annual_sharpes,
        "inverted_min_annual": inv_min_annual,
        "benchmark_spread_ann_return": spread_ann_ret,
        "benchmark_spread_ann_vol": spread_ann_vol,
        "benchmark_spread_sharpe": spread_sharpe,
        "benchmark_spread_max_dd": spread_dd,
        "benchmark_spread_hit_rate": spread_hit,
        "benchmark_spread_equity": spread_equity,
        "prefer_inverted": bool(inv_sharpe > sharpe + 0.10),
    }
    out.update(quality)

    # Backward-compatible aliases used by later notebook cells.
    out["sharpe_inverted"] = inv_sharpe
    out["ann_return_inverted"] = inv_ann_ret
    out["ann_vol_inverted"] = inv_ann_vol
    out["max_dd_inverted"] = inv_dd
    out["hit_rate_inverted"] = inv_hit
    out["equity_inverted"] = inv_equity
    out["beta_inverted"] = beta_inv
    out["consistency_inverted"] = inv_consistency
    out["annual_sharpes_inverted"] = inv_annual_sharpes
    out["min_annual_inverted"] = inv_min_annual
    out["excess_ann_return"] = spread_ann_ret
    out["excess_ann_vol"] = spread_ann_vol
    out["excess_sharpe"] = spread_sharpe
    out["excess_max_dd"] = spread_dd
    out["excess_hit_rate"] = spread_hit
    out["excess_equity"] = spread_equity
    return out
