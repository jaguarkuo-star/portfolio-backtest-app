import json
import math
from datetime import date
from urllib.parse import quote

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf


TRADING_DAYS = 252

DEFAULT_HOLDINGS = pd.DataFrame(
    [
        {"name": "0050", "ticker": "0050.TW", "shares": 0, "amount": 250000, "currency": "TWD"},
        {"name": "台積電", "ticker": "2330.TW", "shares": 0, "amount": 250000, "currency": "TWD"},
        {"name": "鴻海", "ticker": "2317.TW", "shares": 0, "amount": 250000, "currency": "TWD"},
        {"name": "聯發科", "ticker": "2454.TW", "shares": 0, "amount": 250000, "currency": "TWD"},
    ]
)


st.set_page_config(page_title="資產配置回測工作台", layout="wide")


def raise_supabase_error(response: requests.Response) -> None:
    if response.ok:
        return
    detail = response.text.strip()
    raise RuntimeError(f"Supabase HTTP {response.status_code}: {detail[:800]}")


def clean_fx(series: pd.Series, max_daily_move: float = 0.20) -> pd.Series:
    clean = series.astype(float).copy()
    bad = clean.pct_change(fill_method=None).abs() > max_daily_move
    clean.loc[bad] = np.nan
    return clean.ffill().bfill()


@st.cache_data(ttl=60 * 60 * 8, show_spinner=False)
def download_adjusted_prices(tickers: tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    data = yf.download(
        list(tickers),
        start=start,
        end=(pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=True,
        actions=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    if data.empty:
        raise RuntimeError("yfinance 沒有回傳資料，請檢查 ticker 或日期。")
    close = data["Close"].copy() if isinstance(data.columns, pd.MultiIndex) else data[["Close"]].copy()
    if not isinstance(close, pd.DataFrame):
        close = close.to_frame()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.sort_index().ffill().dropna(how="all")


@st.cache_data(ttl=60 * 60 * 8, show_spinner=False)
def latest_usd_twd() -> tuple[float, str]:
    data = yf.download("TWD=X", period="10d", auto_adjust=True, actions=False, progress=False)
    if data.empty:
        raise RuntimeError("無法從 yfinance 取得 USD/TWD 匯率。")
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = clean_fx(close).dropna()
    return float(close.iloc[-1]), pd.Timestamp(close.index[-1]).strftime("%Y-%m-%d")


@st.cache_data(ttl=60 * 60 * 8, show_spinner=False)
def download_latest_prices(tickers: tuple[str, ...]) -> dict[str, float]:
    data = yf.download(
        list(tickers),
        period="10d",
        auto_adjust=True,
        actions=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    if data.empty:
        raise RuntimeError("yfinance 沒有回傳最新股價，請檢查 ticker。")
    close = data["Close"].copy() if isinstance(data.columns, pd.MultiIndex) else data["Close"].copy()
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    prices = {}
    for ticker in tickers:
        if ticker in close.columns and not close[ticker].dropna().empty:
            prices[ticker] = float(close[ticker].dropna().iloc[-1])
    return prices


@st.cache_data(ttl=60 * 60 * 8, show_spinner=False)
def download_analyst_targets(tickers: tuple[str, ...]) -> dict[str, dict[str, float]]:
    targets = {}
    for ticker in tickers:
        try:
            raw = yf.Ticker(ticker).analyst_price_targets
            if isinstance(raw, dict) and raw:
                targets[ticker] = {
                    key: float(value)
                    for key, value in raw.items()
                    if value is not None and pd.notna(value)
                }
        except Exception:
            continue
    return targets


def ensure_holdings_schema(holdings: pd.DataFrame) -> pd.DataFrame:
    df = holdings.copy()
    for col, default in {
        "name": "",
        "ticker": "",
        "shares": 0.0,
        "amount": 0.0,
        "currency": "TWD",
    }.items():
        if col not in df.columns:
            df[col] = default
    return df[["name", "ticker", "shares", "amount", "currency"]]


def ensure_valuation_schema(valuation: pd.DataFrame, holdings: pd.DataFrame) -> pd.DataFrame:
    base = ensure_holdings_schema(holdings)[["name", "ticker", "currency"]].copy()
    df = valuation.copy() if valuation is not None and not valuation.empty else pd.DataFrame()
    for col, default in {
        "ticker": "",
        "target_price": 0.0,
        "fcf_per_share": 0.0,
        "growth_1_5": 0.10,
        "terminal_growth": 0.02,
        "discount_rate": 0.10,
        "net_cash_per_share": 0.0,
        "margin_safety": 0.25,
    }.items():
        if col not in df.columns:
            df[col] = default

    if not df.empty:
        df = df.drop_duplicates(subset=["ticker"], keep="last")
        base = base.merge(
            df[
                [
                    "ticker",
                    "target_price",
                    "fcf_per_share",
                    "growth_1_5",
                    "terminal_growth",
                    "discount_rate",
                    "net_cash_per_share",
                    "margin_safety",
                ]
            ],
            on="ticker",
            how="left",
        )

    defaults = {
        "target_price": 0.0,
        "fcf_per_share": 0.0,
        "growth_1_5": 0.10,
        "terminal_growth": 0.02,
        "discount_rate": 0.10,
        "net_cash_per_share": 0.0,
        "margin_safety": 0.25,
    }
    for col, default in defaults.items():
        base[col] = pd.to_numeric(base[col], errors="coerce").fillna(default)
    return base[
        [
            "name",
            "ticker",
            "currency",
            "target_price",
            "fcf_per_share",
            "growth_1_5",
            "terminal_growth",
            "discount_rate",
            "net_cash_per_share",
            "margin_safety",
        ]
    ]


def dcf_two_stage(
    fcf_per_share: float,
    growth_1_5: float,
    terminal_growth: float,
    discount_rate: float,
    net_cash_per_share: float,
    margin_safety: float,
) -> dict[str, float]:
    if fcf_per_share <= 0 or discount_rate <= terminal_growth:
        return {"fair_value": np.nan, "buy_below": np.nan}

    value = 0.0
    fcf = fcf_per_share
    for year in range(1, 6):
        fcf *= 1 + growth_1_5
        value += fcf / ((1 + discount_rate) ** year)

    terminal_value = fcf * (1 + terminal_growth) / (discount_rate - terminal_growth)
    value += terminal_value / ((1 + discount_rate) ** 5)
    value += net_cash_per_share
    return {
        "fair_value": float(value),
        "buy_below": float(value * (1 - margin_safety)),
    }


def tw_stock_code(ticker: str) -> str:
    return ticker.replace(".TW", "").replace(".TWO", "").strip()


def mops_link(ticker: str) -> str:
    code = tw_stock_code(ticker)
    return f"https://mops.twse.com.tw/mops/web/t100sb07_1?co_id={quote(code)}"


def add_twd_values(
    holdings: pd.DataFrame,
    usd_twd: float,
    latest_prices: dict[str, float] | None = None,
) -> pd.DataFrame:
    df = holdings.copy()
    latest_prices = latest_prices or {}
    df = ensure_holdings_schema(df)
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0.0)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    df["currency"] = df["currency"].fillna("TWD")
    price_amount = df.apply(
        lambda r: r["shares"] * latest_prices.get(r["ticker"], np.nan),
        axis=1,
    )
    df["amount_used"] = np.where(
        (df["shares"] > 0) & price_amount.notna(),
        price_amount,
        df["amount"],
    )
    df["amount_source"] = np.where(
        (df["shares"] > 0) & price_amount.notna(),
        "股數x最新價",
        "手動金額",
    )
    df["twd_value"] = np.where(df["currency"] == "USD", df["amount_used"] * usd_twd, df["amount_used"])
    total = df["twd_value"].sum()
    df["weight"] = df["twd_value"] / total if total > 0 else 0.0
    return df


def portfolio_settings_payload(holdings: pd.DataFrame) -> list[dict]:
    cols = ["name", "ticker", "shares", "amount", "currency"]
    holdings = ensure_holdings_schema(holdings)
    payload = holdings[cols].copy()
    payload["shares"] = pd.to_numeric(payload["shares"], errors="coerce").fillna(0.0)
    payload["amount"] = pd.to_numeric(payload["amount"], errors="coerce").fillna(0.0)
    return payload.to_dict("records")


def valuation_settings_payload(valuation: pd.DataFrame) -> list[dict]:
    cols = [
        "ticker",
        "target_price",
        "fcf_per_share",
        "growth_1_5",
        "terminal_growth",
        "discount_rate",
        "net_cash_per_share",
        "margin_safety",
    ]
    payload = valuation.copy()
    for col in cols:
        if col not in payload.columns:
            payload[col] = 0.0
    for col in cols[1:]:
        payload[col] = pd.to_numeric(payload[col], errors="coerce").fillna(0.0)
    return payload[cols].to_dict("records")


def app_settings_payload(holdings: pd.DataFrame, valuation: pd.DataFrame) -> dict:
    holdings = ensure_holdings_schema(holdings)
    valuation = ensure_valuation_schema(valuation, holdings)
    return {
        "holdings": portfolio_settings_payload(holdings),
        "valuation": valuation_settings_payload(valuation),
    }


def parse_settings_payload(settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    if isinstance(settings, dict):
        holdings = ensure_holdings_schema(pd.DataFrame(settings.get("holdings", [])))
        valuation = ensure_valuation_schema(pd.DataFrame(settings.get("valuation", [])), holdings)
        return holdings, valuation
    holdings = ensure_holdings_schema(pd.DataFrame(settings))
    return holdings, ensure_valuation_schema(pd.DataFrame(), holdings)


def supabase_configured() -> bool:
    return bool(st.secrets.get("SUPABASE_URL")) and bool(supabase_key())


def supabase_key() -> str:
    return st.secrets.get("SUPABASE_SERVICE_ROLE_KEY") or st.secrets.get("SUPABASE_KEY") or ""


def supabase_key_label() -> str:
    key = supabase_key()
    if not key:
        return "未設定"
    if st.secrets.get("SUPABASE_SERVICE_ROLE_KEY"):
        return "service_role"
    if key.startswith("eyJ"):
        return "legacy anon JWT"
    if key.startswith("sb_publishable_"):
        return "publishable key"
    return "unknown"


def supabase_headers() -> dict[str, str]:
    key = supabase_key()
    headers = {
        "apikey": key,
        "Content-Type": "application/json",
    }
    if key.startswith("eyJ"):
        headers["Authorization"] = f"Bearer {key}"
    return headers


def supabase_rest_url() -> str:
    base_url = st.secrets["SUPABASE_URL"].rstrip("/")
    if base_url.endswith("/rest/v1"):
        return base_url
    return f"{base_url}/rest/v1"


def test_supabase_connection() -> tuple[bool, str]:
    url = f"{supabase_rest_url()}/portfolio_settings?select=user_key&limit=1"
    response = requests.get(url, headers=supabase_headers(), timeout=15)
    if response.ok:
        return True, f"連線成功：HTTP {response.status_code}"
    return False, f"HTTP {response.status_code}: {response.text[:300]}"


def load_settings_from_db(user_key: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    url = f"{supabase_rest_url()}/portfolio_settings"
    params = f"?user_key=eq.{quote(user_key)}&select=settings&limit=1"
    response = requests.get(url + params, headers=supabase_headers(), timeout=15)
    raise_supabase_error(response)
    rows = response.json()
    if not rows:
        return None
    return parse_settings_payload(rows[0]["settings"])


def save_settings_to_db(user_key: str, holdings: pd.DataFrame, valuation: pd.DataFrame) -> None:
    url = f"{supabase_rest_url()}/portfolio_settings"
    headers = supabase_headers() | {"Prefer": "resolution=merge-duplicates"}
    body = {
        "user_key": user_key,
        "settings": app_settings_payload(holdings, valuation),
    }
    response = requests.post(url, headers=headers, json=body, timeout=15)
    raise_supabase_error(response)


def rebalance_months(freq: str) -> int | None:
    return {
        "不再平衡": None,
        "三個月": 3,
        "半年": 6,
        "一年": 12,
        "兩年": 24,
        "三年": 36,
    }[freq]


def sell_tax_rate(ticker: str, stock_tax: float, etf_tax: float) -> float:
    if ticker == "0050.TW":
        return etf_tax
    if ticker.endswith(".TW") or ticker.endswith(".TWO"):
        return stock_tax
    return 0.0


def trade_cost(
    old_w: pd.Series,
    new_w: pd.Series,
    commission: float,
    stock_tax: float,
    etf_tax: float,
) -> float:
    tickers = old_w.index.union(new_w.index)
    trade = new_w.reindex(tickers, fill_value=0) - old_w.reindex(tickers, fill_value=0)
    cost = trade.clip(lower=0).sum() * commission
    for ticker, sell in (-trade.clip(upper=0)).items():
        cost += sell * (commission + sell_tax_rate(ticker, stock_tax, etf_tax))
    return float(cost)


def portfolio_returns(
    prices: pd.DataFrame,
    weights: pd.Series,
    frequency: str,
    commission: float,
    stock_tax: float,
    etf_tax: float,
) -> pd.Series:
    returns = prices.pct_change(fill_method=None).dropna(how="all")
    weights = weights / weights.sum()
    months = rebalance_months(frequency)

    initial_cost = weights.sum() * commission
    if months is None:
        active = returns.notna().mul(weights, axis=1).sum(axis=1)
        port = returns.mul(weights, axis=1).sum(axis=1) / active
        if len(port):
            port.iloc[0] -= initial_cost
        return port.replace([np.inf, -np.inf], np.nan).dropna()

    current = weights.copy()
    target = weights.copy()
    next_reb = returns.index[0] + pd.DateOffset(months=months)
    out = []

    for i, (dt, row) in enumerate(returns.iterrows()):
        cost = initial_cost if i == 0 else 0.0
        if dt >= next_reb:
            cost += trade_cost(current, target, commission, stock_tax, etf_tax)
            current = target.copy()
            while dt >= next_reb:
                next_reb += pd.DateOffset(months=months)

        valid = row.dropna()
        day_w = current.reindex(valid.index).dropna()
        if day_w.sum() <= 0:
            continue
        day_w = day_w / day_w.sum()
        day_ret = float((valid * day_w).sum() - cost)
        out.append((dt, day_ret))

        grown = day_w * (1 + valid)
        current = grown / grown.sum()

    return pd.Series(dict(out), name="portfolio").sort_index()


def stats(returns: pd.Series, inflation: float) -> dict[str, float]:
    wealth = (1 + returns).cumprod()
    years = (returns.index[-1] - returns.index[0]).days / 365.25
    cagr = wealth.iloc[-1] ** (1 / years) - 1
    dd = wealth / wealth.cummax() - 1
    return {
        "total_return": float(wealth.iloc[-1] - 1),
        "cagr": float(cagr),
        "real_cagr": float((1 + cagr) / (1 + inflation) - 1),
        "volatility": float(returns.std(ddof=1) * math.sqrt(TRADING_DAYS)),
        "max_drawdown": float(dd.min()),
    }


def alpha_beta(port: pd.Series, bench: pd.Series, risk_free: float) -> dict[str, float]:
    aligned = pd.concat({"p": port, "b": bench}, axis=1).dropna()
    rf = (1 + risk_free) ** (1 / TRADING_DAYS) - 1
    y = aligned["p"] - rf
    x = aligned["b"] - rf
    beta, alpha_daily = np.polyfit(x.to_numpy(), y.to_numpy(), 1)
    return {"alpha": float((1 + alpha_daily) ** TRADING_DAYS - 1), "beta": float(beta)}


def bootstrap(returns: pd.Series, samples: int, draws: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sampled = rng.choice(returns.to_numpy(), size=(samples, draws), replace=True)
    total = np.prod(1 + sampled, axis=1) - 1
    annualized = (1 + total) ** (TRADING_DAYS / draws) - 1
    return pd.DataFrame({"total_return": total, "annualized_return": annualized})


st.title("資產配置回測工作台")

if "holdings_default" not in st.session_state:
    st.session_state.holdings_default = ensure_holdings_schema(DEFAULT_HOLDINGS)
if "editor_data" not in st.session_state:
    st.session_state.editor_data = ensure_holdings_schema(st.session_state.holdings_default)
if "editor_key" not in st.session_state:
    st.session_state.editor_key = 0
st.session_state.holdings_default = ensure_holdings_schema(st.session_state.holdings_default)
st.session_state.editor_data = ensure_holdings_schema(st.session_state.editor_data)
if "valuation_data" not in st.session_state:
    st.session_state.valuation_data = ensure_valuation_schema(pd.DataFrame(), st.session_state.editor_data)
if "valuation_key" not in st.session_state:
    st.session_state.valuation_key = 0
if "latest_prices" not in st.session_state:
    st.session_state.latest_prices = {}
st.session_state.valuation_data = ensure_valuation_schema(
    st.session_state.valuation_data,
    st.session_state.editor_data,
)
if "usd_twd" not in st.session_state:
    st.session_state.usd_twd = 31.673
if "usd_twd_date" not in st.session_state:
    st.session_state.usd_twd_date = "預設值"

with st.sidebar:
    st.header("回測設定")
    start = st.date_input("開始日", date(2014, 8, 31))
    end = st.date_input("結束日", date.today())
    benchmark = st.text_input("Benchmark", "0050.TW")
    frequency = st.selectbox("再平衡", ["不再平衡", "三個月", "半年", "一年", "兩年", "三年"])
    inflation = st.number_input("年通膨假設", value=0.02, step=0.005, format="%.4f")
    risk_free = st.number_input("年無風險利率", value=0.0, step=0.005, format="%.4f")
    commission = st.number_input("手續費率", value=0.001425, step=0.0001, format="%.6f")
    stock_tax = st.number_input("台股賣出交易稅", value=0.003, step=0.0005, format="%.6f")
    etf_tax = st.number_input("ETF 賣出交易稅", value=0.001, step=0.0005, format="%.6f")
    samples = st.number_input("Bootstrap 次數", min_value=1000, max_value=50000, value=10000, step=1000)
    draws = st.number_input("每次抽樣日數", min_value=20, max_value=2000, value=650, step=10)
    refresh = st.button("清除快取，重新抓資料")
    if refresh:
        st.cache_data.clear()

st.subheader("持股與比例")
fx_col1, fx_col2 = st.columns([1, 3])
with fx_col1:
    if st.button("更新 USD/TWD 匯率"):
        try:
            st.session_state.usd_twd, st.session_state.usd_twd_date = latest_usd_twd()
            st.success("匯率已更新。")
        except Exception as exc:
            st.warning(f"暫時無法更新匯率，先使用目前畫面匯率：{exc}")
with fx_col2:
    st.session_state.usd_twd = st.number_input(
        "USD/TWD 匯率",
        min_value=1.0,
        value=float(st.session_state.usd_twd),
        step=0.01,
        format="%.4f",
    )
usd_twd = float(st.session_state.usd_twd)
st.caption(f"目前畫面換算使用 USD/TWD：{usd_twd:.4f}，來源日期：{st.session_state.usd_twd_date}")

with st.expander("每個人自己的預設設定", expanded=False):
    user_key = st.text_input("資料庫保存代號", value=st.session_state.get("user_key", ""))
    st.session_state.user_key = user_key.strip()

    if supabase_configured():
        st.caption(f"Supabase URL: {supabase_rest_url()}")
        st.caption(f"Supabase key 類型: {supabase_key_label()}")
        if st.button("測試 Supabase 連線"):
            try:
                ok, message = test_supabase_connection()
                if ok:
                    st.success(message)
                else:
                    st.error(message)
            except Exception as exc:
                st.error(f"連線測試失敗：{exc}")

        db_col1, db_col2 = st.columns(2)
        if db_col1.button("從資料庫載入"):
            if not st.session_state.user_key:
                st.error("請先輸入保存代號。")
            else:
                try:
                    loaded = load_settings_from_db(st.session_state.user_key)
                    if loaded is None:
                        st.warning("找不到這個保存代號的設定。")
                    else:
                        loaded_holdings, loaded_valuation = loaded
                        st.session_state.holdings_default = ensure_holdings_schema(loaded_holdings)
                        st.session_state.editor_data = ensure_holdings_schema(loaded_holdings)
                        st.session_state.valuation_data = ensure_valuation_schema(loaded_valuation, loaded_holdings)
                        st.session_state.editor_key += 1
                        st.session_state.valuation_key += 1
                        st.success("已從資料庫載入設定。")
                        st.rerun()
                except Exception as exc:
                    st.error(f"資料庫載入失敗：{exc}")
        if db_col2.button("儲存目前設定到資料庫"):
            if not st.session_state.user_key:
                st.error("請先輸入保存代號。")
            else:
                try:
                    save_settings_to_db(
                        st.session_state.user_key,
                        st.session_state.editor_data,
                        st.session_state.valuation_data,
                    )
                    st.success("已儲存。之後用同一個保存代號即可載入。")
                except Exception as exc:
                    st.error(f"資料庫儲存失敗：{exc}")
    else:
        st.info("尚未設定 Supabase secrets；目前只能用 JSON 下載/上傳保存。")

    uploaded = st.file_uploader("上傳自己的設定 JSON", type=["json"])
    if uploaded is not None:
        try:
            rows = json.loads(uploaded.getvalue().decode("utf-8"))
            loaded_holdings, loaded_valuation = parse_settings_payload(rows)
            required = ["name", "ticker", "amount", "currency"]
            if not set(required).issubset(loaded_holdings.columns):
                st.error("設定檔需要包含 name、ticker、amount、currency 欄位。")
            else:
                st.session_state.holdings_default = loaded_holdings.copy()
                st.session_state.editor_data = loaded_holdings.copy()
                st.session_state.valuation_data = ensure_valuation_schema(loaded_valuation, loaded_holdings)
                st.session_state.editor_key += 1
                st.session_state.valuation_key += 1
                st.success("已載入你的設定，本次使用這份作為預設。")
                st.rerun()
        except Exception as exc:
            st.error(f"設定檔讀取失敗：{exc}")

    st.download_button(
        "下載目前預設設定 JSON",
        json.dumps(
            app_settings_payload(st.session_state.editor_data, st.session_state.valuation_data),
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8"),
        "my_portfolio_settings.json",
        "application/json",
    )

with st.form("holdings_form"):
    edited_raw = st.data_editor(
        st.session_state.editor_data,
        num_rows="dynamic",
        use_container_width=True,
        key=f"holdings_editor_{st.session_state.editor_key}",
        column_config={
            "name": "名稱",
            "ticker": "Ticker",
            "shares": st.column_config.NumberColumn("股數", min_value=0, step=1),
            "amount": st.column_config.NumberColumn("原幣金額", min_value=0, step=100),
            "currency": st.column_config.SelectboxColumn("幣別", options=["TWD", "USD"]),
        },
    )
    form_col1, form_col2 = st.columns(2)
    with form_col1:
        apply_holdings = st.form_submit_button("套用持股變更")
    with form_col2:
        auto_amount = st.form_submit_button("股數換算金額")

if apply_holdings or auto_amount:
    edited_raw = edited_raw.dropna(subset=["ticker"]).copy()
    edited_raw["ticker"] = edited_raw["ticker"].astype(str).str.strip()
    edited_raw = ensure_holdings_schema(edited_raw)
    if auto_amount:
        priced_rows = edited_raw[pd.to_numeric(edited_raw["shares"], errors="coerce").fillna(0) > 0]
        tickers_to_price = tuple(dict.fromkeys(priced_rows["ticker"].dropna().astype(str).str.strip()))
        if tickers_to_price:
            try:
                latest_prices = download_latest_prices(tickers_to_price)
                if not latest_prices:
                    raise RuntimeError("沒有抓到可用的最新價格。")
                st.session_state.latest_prices.update(latest_prices)
                shares = pd.to_numeric(edited_raw["shares"], errors="coerce").fillna(0.0)
                edited_raw["amount"] = edited_raw.apply(
                    lambda r: shares.loc[r.name] * latest_prices[r["ticker"]]
                    if shares.loc[r.name] > 0 and r["ticker"] in latest_prices
                    else r["amount"],
                    axis=1,
                )
                st.session_state.auto_amount_message = "已用最新 yfinance 股價更新有填股數的原幣金額。"
            except Exception as exc:
                st.session_state.auto_amount_message = f"股數換算失敗：{exc}"
        else:
            st.session_state.auto_amount_message = "沒有填股數大於 0 的持股，所以沒有更新金額。"
    st.session_state.editor_data = edited_raw
    if auto_amount:
        st.session_state.editor_key += 1
        st.rerun()

edited = add_twd_values(st.session_state.editor_data, usd_twd)
st.caption("輸入表格後請先按「套用持股變更」，再儲存或執行回測。")
if "auto_amount_message" in st.session_state:
    message = st.session_state.pop("auto_amount_message")
    if message.startswith("股數換算失敗"):
        st.warning(message)
    else:
        st.success(message)

col1, col2, col3 = st.columns(3)
col1.metric("總資產台幣等值", f"{edited['twd_value'].sum():,.0f}")
col2.metric("台股台幣金額", f"{edited.loc[edited['currency'] == 'TWD', 'amount'].sum():,.0f}")
col3.metric("美股美元金額", f"US$ {edited.loc[edited['currency'] == 'USD', 'amount'].sum():,.0f}")

display_cols = edited[["name", "ticker", "currency", "shares", "amount", "amount_used", "amount_source", "twd_value", "weight"]].copy()
display_cols["shares"] = display_cols["shares"].map(lambda x: f"{x:,.2f}".rstrip("0").rstrip("."))
display_cols["amount"] = display_cols.apply(
    lambda r: f"US$ {r['amount']:,.0f}" if r["currency"] == "USD" else f"NT$ {r['amount']:,.0f}",
    axis=1,
)
display_cols["amount_used"] = display_cols.apply(
    lambda r: f"US$ {r['amount_used']:,.0f}" if r["currency"] == "USD" else f"NT$ {r['amount_used']:,.0f}",
    axis=1,
)
display_cols["twd_value"] = display_cols["twd_value"].map(lambda x: f"NT$ {x:,.0f}")
display_cols["weight"] = display_cols["weight"].map(lambda x: f"{x:.2%}")
st.dataframe(
    display_cols.rename(
        columns={
            "name": "名稱",
            "ticker": "Ticker",
            "currency": "幣別",
            "shares": "股數",
            "amount": "原幣金額",
            "amount_used": "計算用金額",
            "amount_source": "金額來源",
            "twd_value": "台幣等值",
            "weight": "權重",
        }
    ),
    use_container_width=True,
)

st.plotly_chart(
    px.pie(edited, names="name", values="weight", title="資產配置", hole=0.38),
    use_container_width=True,
)

st.session_state.valuation_data = ensure_valuation_schema(st.session_state.valuation_data, st.session_state.editor_data)

with st.expander("估值與研究", expanded=False):
    st.caption("DCF 使用兩階段模型：前 5 年成長率 + 永續成長率。FCF 可以先用每股自由現金流或你想估的每股盈餘替代。")
    research_col1, research_col2 = st.columns(2)
    current_tickers = tuple(dict.fromkeys(edited["ticker"].dropna().astype(str).str.strip()))

    if research_col1.button("抓最新股價"):
        try:
            latest_prices = download_latest_prices(current_tickers)
            st.session_state.latest_prices.update(latest_prices)
            st.success("已更新最新股價。")
        except Exception as exc:
            st.warning(f"最新股價更新失敗：{exc}")

    if research_col2.button("抓 Yahoo 法人目標價"):
        try:
            st.session_state.analyst_targets = download_analyst_targets(current_tickers)
            if st.session_state.analyst_targets:
                st.success("已更新 Yahoo 目標價。")
            else:
                st.warning("Yahoo 沒有回傳可用目標價，台股通常比較容易缺資料。")
        except Exception as exc:
            st.warning(f"Yahoo 目標價更新失敗：{exc}")

    with st.form("valuation_form"):
        valuation_raw = st.data_editor(
            st.session_state.valuation_data,
            num_rows="fixed",
            use_container_width=True,
            key=f"valuation_editor_{st.session_state.valuation_key}",
            disabled=["name", "ticker", "currency"],
            column_config={
                "name": st.column_config.TextColumn("名稱"),
                "ticker": st.column_config.TextColumn("Ticker"),
                "currency": st.column_config.TextColumn("幣別"),
                "target_price": st.column_config.NumberColumn("手動目標價", min_value=0.0, step=1.0),
                "fcf_per_share": st.column_config.NumberColumn("每股FCF", min_value=0.0, step=1.0),
                "growth_1_5": st.column_config.NumberColumn("前5年成長率", step=0.01, format="%.4f"),
                "terminal_growth": st.column_config.NumberColumn("永續成長率", step=0.005, format="%.4f"),
                "discount_rate": st.column_config.NumberColumn("折現率", step=0.005, format="%.4f"),
                "net_cash_per_share": st.column_config.NumberColumn("每股淨現金", step=1.0),
                "margin_safety": st.column_config.NumberColumn("安全邊際", min_value=0.0, max_value=0.95, step=0.05, format="%.4f"),
            },
        )
        apply_valuation = st.form_submit_button("套用估值參數")

    if apply_valuation:
        st.session_state.valuation_data = ensure_valuation_schema(valuation_raw, st.session_state.editor_data)
        st.success("已套用估值參數。")

    valuation = ensure_valuation_schema(st.session_state.valuation_data, st.session_state.editor_data)
    analyst_targets = st.session_state.get("analyst_targets", {})
    rows = []
    for row in valuation.to_dict("records"):
        dcf = dcf_two_stage(
            float(row["fcf_per_share"]),
            float(row["growth_1_5"]),
            float(row["terminal_growth"]),
            float(row["discount_rate"]),
            float(row["net_cash_per_share"]),
            float(row["margin_safety"]),
        )
        ticker = row["ticker"]
        latest = st.session_state.latest_prices.get(ticker, np.nan)
        yahoo_target = analyst_targets.get(ticker, {}).get("mean", np.nan)
        if pd.isna(yahoo_target):
            yahoo_target = analyst_targets.get(ticker, {}).get("current", np.nan)
        manual_target = float(row["target_price"])
        preferred_target = manual_target if manual_target > 0 else yahoo_target
        upside = preferred_target / latest - 1 if pd.notna(preferred_target) and pd.notna(latest) and latest > 0 else np.nan
        dcf_gap = dcf["fair_value"] / latest - 1 if pd.notna(dcf["fair_value"]) and pd.notna(latest) and latest > 0 else np.nan
        rows.append(
            {
                "名稱": row["name"],
                "Ticker": ticker,
                "幣別": row["currency"],
                "最新價": latest,
                "DCF合理價": dcf["fair_value"],
                "安全買進價": dcf["buy_below"],
                "DCF空間": dcf_gap,
                "手動目標價": manual_target if manual_target > 0 else np.nan,
                "Yahoo目標價": yahoo_target,
                "目標價空間": upside,
                "法說會": mops_link(ticker) if ticker.endswith((".TW", ".TWO")) else "",
            }
        )

    valuation_result = pd.DataFrame(rows)
    for col in ["最新價", "DCF合理價", "安全買進價", "手動目標價", "Yahoo目標價"]:
        valuation_result[col] = valuation_result[col].map(lambda x: "" if pd.isna(x) else f"{x:,.2f}")
    for col in ["DCF空間", "目標價空間"]:
        valuation_result[col] = valuation_result[col].map(lambda x: "" if pd.isna(x) else f"{x:.2%}")

    st.dataframe(
        valuation_result,
        use_container_width=True,
        column_config={
            "法說會": st.column_config.LinkColumn("法說會"),
        },
    )
    st.caption("台股法說會連結會開到公開資訊觀測站；Yahoo 目標價對美股較常有資料，台股可能空白。")

if st.button("執行回測", type="primary"):
    with st.spinner("正在連接 yfinance 並計算..."):
        holdings = ensure_holdings_schema(st.session_state.editor_data).to_dict("records")
        tickers = [h["ticker"] for h in holdings] + [benchmark]
        if any(h.get("currency") == "USD" for h in holdings):
            tickers.append("TWD=X")

        close = download_adjusted_prices(tuple(dict.fromkeys(tickers)), str(start), str(end))
        prices = close[[h["ticker"] for h in holdings]].copy()
        if "TWD=X" in close.columns:
            fx = clean_fx(close["TWD=X"])
            if not fx.dropna().empty:
                usd_twd = float(fx.dropna().iloc[-1])
                st.session_state.usd_twd = usd_twd
                st.session_state.usd_twd_date = pd.Timestamp(fx.dropna().index[-1]).strftime("%Y-%m-%d")
            for h in holdings:
                if h.get("currency") == "USD":
                    prices[h["ticker"]] = prices[h["ticker"]] * fx

        latest_prices = {
            h["ticker"]: float(close[h["ticker"]].dropna().iloc[-1])
            for h in holdings
            if h["ticker"] in close.columns and not close[h["ticker"]].dropna().empty
        }
        edited_for_backtest = add_twd_values(pd.DataFrame(holdings), usd_twd, latest_prices)
        weights = pd.Series({h["ticker"]: float(h["weight"]) for h in edited_for_backtest.to_dict("records")})
        port = portfolio_returns(prices, weights, frequency, commission, stock_tax, etf_tax)
        bench = close[benchmark].pct_change(fill_method=None).dropna()
        idx = port.index.intersection(bench.index)
        port = port.loc[idx]
        bench = bench.loc[idx]

        s = stats(port, inflation)
        ab = alpha_beta(port, bench, risk_free)
        wealth = pd.DataFrame({"投組": (1 + port).cumprod(), benchmark: (1 + bench).cumprod()})

        b = bootstrap(port, int(samples), int(draws))

    st.subheader("回測結果")
    c1, c2, c3 = st.columns(3)
    c1.metric("總報酬", f"{s['total_return']:.2%}")
    c2.metric("CAGR", f"{s['cagr']:.2%}")
    c3.metric("實質 CAGR", f"{s['real_cagr']:.2%}")
    c4, c5, c6 = st.columns(3)
    c4.metric("波動", f"{s['volatility']:.2%}")
    c5.metric("最大回撤", f"{s['max_drawdown']:.2%}")
    c6.metric("Beta", f"{ab['beta']:.3f}")
    st.metric("Alpha 年化", f"{ab['alpha']:.2%}")

    fig = go.Figure()
    for column in wealth.columns:
        fig.add_trace(go.Scatter(x=wealth.index, y=wealth[column], mode="lines", name=column))
    fig.update_layout(title="回測走勢圖", yaxis_title="Growth of 1.00", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Bootstrap 報酬分布")
    hist = px.histogram(b, x="total_return", nbins=80, title=f"{int(samples):,} 次，每次抽 {int(draws)} 筆日報酬")
    hist.update_layout(xaxis_tickformat=".0%", yaxis_title="次數")
    st.plotly_chart(hist, use_container_width=True)

    summary = pd.DataFrame(
        {
            "統計": ["平均", "中位數", "虧損機率", "5% 分位", "10% 分位", "90% 分位", "95% 分位"],
            "650日總報酬": [
                b["total_return"].mean(),
                b["total_return"].median(),
                (b["total_return"] < 0).mean(),
                b["total_return"].quantile(0.05),
                b["total_return"].quantile(0.10),
                b["total_return"].quantile(0.90),
                b["total_return"].quantile(0.95),
            ],
        }
    )
    st.dataframe(summary, use_container_width=True)

    st.download_button("下載回測序列 CSV", wealth.to_csv().encode("utf-8"), "wealth_curve.csv", "text/csv")
    st.download_button("下載 Bootstrap CSV", b.to_csv(index=False).encode("utf-8"), "bootstrap.csv", "text/csv")
else:
    st.info("按「執行回測」後，網站會即時連 yfinance 抓最新含息價格。")
