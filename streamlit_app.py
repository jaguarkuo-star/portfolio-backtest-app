from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET
from datetime import date
from urllib.parse import quote, quote_plus

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf


TRADING_DAYS = 252
MACRO_CACHE_VERSION = "2026-09-02-world-indices-v2"

DEFAULT_HOLDINGS = pd.DataFrame(
    [
        {"name": "0050", "ticker": "0050.TW", "shares": 0, "amount": 250000, "currency": "TWD"},
        {"name": "台積電", "ticker": "2330.TW", "shares": 0, "amount": 250000, "currency": "TWD"},
        {"name": "鴻海", "ticker": "2317.TW", "shares": 0, "amount": 250000, "currency": "TWD"},
        {"name": "聯發科", "ticker": "2454.TW", "shares": 0, "amount": 250000, "currency": "TWD"},
    ]
)

MACRO_INDICATORS = {
    "VIX 恐慌指數": "^VIX",
    "美元指數 DXY": "DX-Y.NYB",
    "USD/TWD": "TWD=X",
    "黃金期貨": "GC=F",
    "WTI 原油": "CL=F",
    "銅期貨": "HG=F",
    "天然氣": "NG=F",
    "小麥": "ZW=F",
    "美股 S&P 500": "^GSPC",
    "NASDAQ 100": "^NDX",
    "台灣加權指數": "^TWII",
    "日本 Nikkei 225": "^N225",
    "歐洲 Euro Stoxx 50": "^STOXX50E",
    "英國 FTSE 100": "^FTSE",
    "新加坡 STI": "^STI",
    "中國上證指數": "000001.SS",
    "中國A股 ASHR ETF": "ASHR",
    "香港恆生指數": "^HSI",
    "韓國 KOSPI": "^KS11",
    "費城半導體": "^SOX",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "USD/CNY": "CNY=X",
    "USD/HKD": "HKD=X",
    "USD/SGD": "SGD=X",
    "USD/KRW": "KRW=X",
    "AUD/USD": "AUDUSD=X",
    "美債 3M": "^IRX",
    "美債 2Y": "^UST2Y",
    "美債 5Y": "^FVX",
    "美債 10Y": "^TNX",
    "美債 30Y": "^TYX",
    "Fed Funds Futures": "ZQ=F",
}

YIELD_TICKERS = {"^IRX", "^FVX", "^TNX", "^TYX", "^UST2Y"}

VERA_RUBIN_SUPPLY_CHAIN = [
    {"category": "平台核心", "company": "NVIDIA", "ticker": "NVDA", "role": "Vera Rubin GPU/CPU/networking platform owner", "source": "NVIDIA official"},
    {"category": "晶圓/先進製程", "company": "TSMC 台積電", "ticker": "2330.TW", "role": "key wafer and chip partner", "source": "NVIDIA official"},
    {"category": "封測/測試", "company": "ASE 日月光投控", "ticker": "3711.TW", "role": "SPIL ecosystem exposure", "source": "NVIDIA official"},
    {"category": "封測/測試", "company": "KYEC 京元電子", "ticker": "2449.TW", "role": "key wafer and chip partner", "source": "NVIDIA official"},
    {"category": "載板/PCB", "company": "Kinsus 景碩", "ticker": "3189.TW", "role": "key wafer and chip partner / substrate exposure", "source": "NVIDIA official"},
    {"category": "載板/PCB", "company": "Unimicron 欣興", "ticker": "3037.TW", "role": "PCB/substrate exposure", "source": "industry map"},
    {"category": "載板/PCB", "company": "Gold Circuit 金像電", "ticker": "2368.TW", "role": "AI server PCB exposure", "source": "industry map"},
    {"category": "主板/配卡/I/O board", "company": "Bojen 博智", "ticker": "8155.TWO", "role": "server motherboard, adapter card and I/O board exposure", "source": "user watchlist"},
    {"category": "主板/配卡/I/O board", "company": "Catcher 可成", "ticker": "2474.TW", "role": "MGX mechanical / structural component supply-chain watchlist", "source": "user watchlist"},
    {"category": "背板/PCB", "company": "Gold Circuit 金像電", "ticker": "2368.TW", "role": "AI server backplane / PCB exposure", "source": "industry map"},
    {"category": "背板/PCB", "company": "Bojen 博智", "ticker": "8155.TWO", "role": "AI server backplane / board exposure", "source": "user watchlist"},
    {"category": "背板/PCB", "company": "HannStar Board 瀚宇博", "ticker": "5469.TW", "role": "server PCB / backplane watchlist", "source": "industry watchlist"},
    {"category": "背板/PCB", "company": "Tripod 健鼎", "ticker": "3044.TW", "role": "server PCB / backplane watchlist", "source": "industry watchlist"},
    {"category": "背板/PCB", "company": "Compeq 華通", "ticker": "2313.TW", "role": "PCB / server board watchlist", "source": "industry watchlist"},
    {"category": "CCL/材料", "company": "Elite Material 台燿", "ticker": "6274.TW", "role": "high-speed CCL material exposure", "source": "industry watchlist"},
    {"category": "CCL/材料", "company": "ITEQ 聯茂", "ticker": "6213.TW", "role": "high-speed CCL material exposure", "source": "industry watchlist"},
    {"category": "HBM 記憶體", "company": "SK hynix", "ticker": "000660.KS", "role": "HBM supplier", "source": "industry reports"},
    {"category": "HBM 記憶體", "company": "Samsung Electronics", "ticker": "005930.KS", "role": "HBM supplier", "source": "industry reports"},
    {"category": "HBM 記憶體", "company": "Micron", "ticker": "MU", "role": "HBM supplier", "source": "industry reports"},
    {"category": "系統/機櫃組裝", "company": "Foxconn 鴻海", "ticker": "2317.TW", "role": "Vera Rubin system manufacturing partner", "source": "NVIDIA official"},
    {"category": "系統/機櫃組裝", "company": "Quanta 廣達/QCT 雲達", "ticker": "2382.TW", "role": "Vera Rubin system manufacturing partner", "source": "NVIDIA official"},
    {"category": "系統/機櫃組裝", "company": "Wistron 緯創", "ticker": "3231.TW", "role": "Vera Rubin system manufacturing partner", "source": "NVIDIA official"},
    {"category": "系統/機櫃組裝", "company": "Wiwynn 緯穎", "ticker": "6669.TW", "role": "Vera Rubin system manufacturing partner", "source": "NVIDIA official"},
    {"category": "系統/機櫃組裝", "company": "Inventec 英業達", "ticker": "2356.TW", "role": "Vera Rubin system manufacturing partner", "source": "NVIDIA official"},
    {"category": "系統/機櫃組裝", "company": "Pegatron 和碩", "ticker": "4938.TW", "role": "Vera Rubin system manufacturing partner", "source": "NVIDIA official"},
    {"category": "系統/機櫃組裝", "company": "Gigabyte 技嘉", "ticker": "2376.TW", "role": "Vera Rubin system manufacturing partner", "source": "NVIDIA official"},
    {"category": "系統/機櫃組裝", "company": "ASUS 華碩", "ticker": "2357.TW", "role": "Vera Rubin system manufacturing partner", "source": "NVIDIA official"},
    {"category": "系統/機櫃組裝", "company": "Compal 仁寶", "ticker": "2324.TW", "role": "Vera Rubin system manufacturing partner", "source": "NVIDIA official"},
    {"category": "系統/機櫃組裝", "company": "MSI 微星", "ticker": "2377.TW", "role": "Vera Rubin system manufacturing partner", "source": "NVIDIA official"},
    {"category": "機櫃/機構件", "company": "Chenbro 勤誠", "ticker": "8210.TW", "role": "AI server chassis / rack mechanical exposure", "source": "industry watchlist"},
    {"category": "機櫃/機構件", "company": "AIC 營邦", "ticker": "3693.TWO", "role": "Vera Rubin system manufacturing partner / chassis and server platform exposure", "source": "NVIDIA official"},
    {"category": "散熱/液冷", "company": "Auras 雙鴻", "ticker": "3324.TW", "role": "AI server cooling / liquid cooling exposure", "source": "industry map"},
    {"category": "散熱/液冷", "company": "AVC 奇鋐", "ticker": "3017.TW", "role": "AI server cooling / liquid cooling exposure", "source": "industry map"},
    {"category": "散熱/液冷", "company": "Jentech 健策", "ticker": "3653.TW", "role": "thermal module / cooling exposure", "source": "industry map"},
    {"category": "電源", "company": "Delta 台達電", "ticker": "2308.TW", "role": "AI server power exposure", "source": "industry map"},
    {"category": "電源", "company": "Lite-On 光寶科", "ticker": "2301.TW", "role": "power supply exposure", "source": "industry map"},
    {"category": "網通/交換器", "company": "Accton 智邦", "ticker": "2345.TW", "role": "data center switch / networking exposure", "source": "industry map"},
    {"category": "網通/交換器晶片", "company": "Marvell", "ticker": "MRVL", "role": "AI networking, custom silicon and optical interconnect exposure", "source": "industry watchlist"},
    {"category": "網通/ASIC", "company": "MediaTek 聯發科", "ticker": "2454.TW", "role": "ASIC / high-speed networking and AI infrastructure watchlist", "source": "user watchlist"},
    {"category": "連接器/線材", "company": "BizLink 貿聯-KY", "ticker": "3665.TW", "role": "cabling / connector exposure", "source": "industry map"},
    {"category": "連接器/線材", "company": "Lotes 嘉澤", "ticker": "3533.TW", "role": "server connector exposure", "source": "industry watchlist"},
    {"category": "滑軌/機構件", "company": "King Slide 川湖", "ticker": "2059.TW", "role": "server rails exposure", "source": "industry map"},
]

CSP_SUPPLY_CHAIN = [
    {"csp": "Microsoft Azure", "category": "CSP", "company": "Microsoft", "ticker": "MSFT", "role": "Azure AI data center capex / Maia ASIC / NVIDIA rack demand", "source": "TrendForce / company filings"},
    {"csp": "Microsoft Azure", "category": "GPU平台", "company": "NVIDIA", "ticker": "NVDA", "role": "rack-scale GPU platform supplier", "source": "TrendForce"},
    {"csp": "Microsoft Azure", "category": "ODM/系統組裝", "company": "Quanta 廣達", "ticker": "2382.TW", "role": "AI server / rack ODM watchlist", "source": "industry watchlist"},
    {"csp": "Microsoft Azure", "category": "ODM/系統組裝", "company": "Foxconn 鴻海", "ticker": "2317.TW", "role": "AI server / rack ODM watchlist", "source": "industry watchlist"},
    {"csp": "Microsoft Azure", "category": "ODM/系統組裝", "company": "Wiwynn 緯穎", "ticker": "6669.TW", "role": "cloud AI server ODM watchlist", "source": "industry watchlist"},
    {"csp": "Microsoft Azure", "category": "ODM/系統組裝", "company": "Wistron 緯創", "ticker": "3231.TW", "role": "AI server / rack ODM watchlist", "source": "industry watchlist"},
    {"csp": "Microsoft Azure", "category": "ASIC/網通晶片", "company": "Marvell", "ticker": "MRVL", "role": "custom silicon / networking ASIC exposure", "source": "industry watchlist"},
    {"csp": "Microsoft Azure", "category": "先進製程", "company": "TSMC 台積電", "ticker": "2330.TW", "role": "advanced-node foundry exposure", "source": "industry watchlist"},

    {"csp": "AWS", "category": "CSP", "company": "Amazon", "ticker": "AMZN", "role": "AWS AI capex / Trainium / NVIDIA rack demand", "source": "TrendForce"},
    {"csp": "AWS", "category": "GPU平台", "company": "NVIDIA", "ticker": "NVDA", "role": "GB300 / V200 rack-scale system demand", "source": "TrendForce"},
    {"csp": "AWS", "category": "ASIC", "company": "Amazon Trainium", "ticker": "AMZN", "role": "in-house Trainium ASIC platform", "source": "TrendForce"},
    {"csp": "AWS", "category": "ODM/系統組裝", "company": "Foxconn 鴻海", "ticker": "2317.TW", "role": "AI server / rack ODM watchlist", "source": "industry watchlist"},
    {"csp": "AWS", "category": "ODM/系統組裝", "company": "Quanta 廣達", "ticker": "2382.TW", "role": "AI server / rack ODM watchlist", "source": "industry watchlist"},
    {"csp": "AWS", "category": "ODM/系統組裝", "company": "Wiwynn 緯穎", "ticker": "6669.TW", "role": "cloud AI server ODM watchlist", "source": "industry watchlist"},
    {"csp": "AWS", "category": "ASIC/網通晶片", "company": "Broadcom", "ticker": "AVGO", "role": "custom silicon / networking exposure watchlist", "source": "industry watchlist"},
    {"csp": "AWS", "category": "ASIC/網通晶片", "company": "Marvell", "ticker": "MRVL", "role": "custom silicon / networking exposure watchlist", "source": "industry watchlist"},

    {"csp": "Google Cloud", "category": "CSP", "company": "Alphabet", "ticker": "GOOGL", "role": "Google Cloud / Gemini / TPU capex", "source": "TrendForce"},
    {"csp": "Google Cloud", "category": "ASIC/TPU", "company": "Broadcom", "ticker": "AVGO", "role": "TPU custom silicon supply-chain watchlist", "source": "industry watchlist"},
    {"csp": "Google Cloud", "category": "先進製程", "company": "TSMC 台積電", "ticker": "2330.TW", "role": "advanced-node foundry exposure", "source": "industry watchlist"},
    {"csp": "Google Cloud", "category": "ODM/系統組裝", "company": "Inventec 英業達", "ticker": "2356.TW", "role": "Google TPU server ODM watchlist", "source": "Digitimes / industry watchlist"},
    {"csp": "Google Cloud", "category": "ODM/系統組裝", "company": "Wiwynn 緯穎", "ticker": "6669.TW", "role": "ASIC server assembly watchlist", "source": "industry watchlist"},
    {"csp": "Google Cloud", "category": "ODM/系統組裝", "company": "Quanta 廣達", "ticker": "2382.TW", "role": "ASIC server allocation watchlist", "source": "industry watchlist"},
    {"csp": "Google Cloud", "category": "ODM/系統組裝", "company": "Foxconn 鴻海", "ticker": "2317.TW", "role": "ASIC/GPU server allocation watchlist", "source": "industry watchlist"},

    {"csp": "Meta", "category": "CSP", "company": "Meta Platforms", "ticker": "META", "role": "AI capex / GPU-heavy server build-out / MTIA ASIC", "source": "TrendForce"},
    {"csp": "Meta", "category": "GPU平台", "company": "NVIDIA", "ticker": "NVDA", "role": "primary GPU platform exposure", "source": "TrendForce"},
    {"csp": "Meta", "category": "GPU平台", "company": "AMD", "ticker": "AMD", "role": "GPU platform exposure watchlist", "source": "TrendForce"},
    {"csp": "Meta", "category": "ASIC", "company": "Meta MTIA", "ticker": "META", "role": "in-house MTIA ASIC platform", "source": "TrendForce"},
    {"csp": "Meta", "category": "ODM/系統組裝", "company": "Quanta 廣達", "ticker": "2382.TW", "role": "AI server / rack ODM watchlist", "source": "industry watchlist"},
    {"csp": "Meta", "category": "ODM/系統組裝", "company": "Wiwynn 緯穎", "ticker": "6669.TW", "role": "cloud AI server ODM watchlist", "source": "industry watchlist"},
    {"csp": "Meta", "category": "ODM/系統組裝", "company": "Inventec 英業達", "ticker": "2356.TW", "role": "ASIC/GPU server ODM watchlist", "source": "industry watchlist"},

    {"csp": "Oracle Cloud", "category": "CSP", "company": "Oracle", "ticker": "ORCL", "role": "OCI GPU rack-scale deployment / OpenAI-related cloud demand", "source": "TrendForce"},
    {"csp": "Oracle Cloud", "category": "GPU平台", "company": "NVIDIA", "ticker": "NVDA", "role": "GPU rack-scale platform supplier", "source": "TrendForce"},
    {"csp": "Oracle Cloud", "category": "ODM/系統組裝", "company": "Supermicro", "ticker": "SMCI", "role": "AI GPU server OEM watchlist", "source": "industry watchlist"},
    {"csp": "Oracle Cloud", "category": "ODM/系統組裝", "company": "Dell", "ticker": "DELL", "role": "AI GPU server OEM watchlist", "source": "industry watchlist"},
    {"csp": "Oracle Cloud", "category": "ODM/系統組裝", "company": "Quanta 廣達", "ticker": "2382.TW", "role": "rack-scale server ODM watchlist", "source": "industry watchlist"},

    {"csp": "CoreWeave", "category": "NeoCloud", "company": "CoreWeave", "ticker": "CRWV", "role": "GPU cloud / NVIDIA rack demand watchlist", "source": "industry watchlist"},
    {"csp": "CoreWeave", "category": "GPU平台", "company": "NVIDIA", "ticker": "NVDA", "role": "GPU platform supplier", "source": "industry watchlist"},
    {"csp": "CoreWeave", "category": "ODM/系統組裝", "company": "Dell", "ticker": "DELL", "role": "AI GPU server OEM watchlist", "source": "industry watchlist"},
    {"csp": "CoreWeave", "category": "ODM/系統組裝", "company": "Supermicro", "ticker": "SMCI", "role": "AI GPU server OEM watchlist", "source": "industry watchlist"},

    {"csp": "Tesla / xAI", "category": "AI Cloud", "company": "Tesla", "ticker": "TSLA", "role": "xAI / Dojo / GPU cluster demand watchlist", "source": "industry watchlist"},
    {"csp": "Tesla / xAI", "category": "GPU平台", "company": "NVIDIA", "ticker": "NVDA", "role": "GPU platform supplier", "source": "industry watchlist"},
    {"csp": "Tesla / xAI", "category": "ODM/系統組裝", "company": "Quanta 廣達", "ticker": "2382.TW", "role": "AI server ODM watchlist", "source": "industry watchlist"},
    {"csp": "Tesla / xAI", "category": "散熱/電源", "company": "Delta 台達電", "ticker": "2308.TW", "role": "AI data center power exposure", "source": "industry watchlist"},

    {"csp": "Chinese CSP", "category": "CSP", "company": "Tencent", "ticker": "0700.HK", "role": "China AI cloud capex watchlist", "source": "TrendForce"},
    {"csp": "Chinese CSP", "category": "CSP", "company": "Alibaba", "ticker": "9988.HK", "role": "China AI cloud capex watchlist", "source": "TrendForce"},
    {"csp": "Chinese CSP", "category": "CSP", "company": "Baidu", "ticker": "9888.HK", "role": "China AI cloud capex watchlist", "source": "TrendForce"},
    {"csp": "Chinese CSP", "category": "ODM/系統組裝", "company": "Lenovo", "ticker": "0992.HK", "role": "China server OEM watchlist", "source": "industry watchlist"},
]


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
def download_close_prices(tickers: tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    data = yf.download(
        list(tickers),
        start=start,
        end=(pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=False,
        actions=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    if data.empty:
        raise RuntimeError("yfinance 沒有回傳股價，請檢查 ticker 或日期。")
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


def clean_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", str(text or ""))
    return re.sub(r"\s+", " ", text).strip()


@st.cache_data(ttl=60 * 30, show_spinner=False)
def fetch_google_news(query: str, limit: int, language: str = "zh-TW") -> list[dict]:
    hl = "zh-TW" if language == "zh-TW" else "en-US"
    gl = "TW" if language == "zh-TW" else "US"
    ceid = "TW:zh-Hant" if language == "zh-TW" else "US:en"
    url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl={hl}&gl={gl}&ceid={ceid}"
    response = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    root = ET.fromstring(response.content)
    rows = []
    for item in root.findall(".//item")[:limit]:
        source = item.find("source")
        rows.append(
            {
                "標題": clean_html(item.findtext("title")),
                "來源": clean_html(source.text if source is not None else ""),
                "時間": clean_html(item.findtext("pubDate")),
                "摘要": clean_html(item.findtext("description")),
                "連結": clean_html(item.findtext("link")),
                "查詢": query,
            }
        )
    return rows


def holding_news_queries(holdings: pd.DataFrame) -> list[tuple[str, str]]:
    rows = []
    for row in ensure_holdings_schema(holdings).to_dict("records"):
        name = str(row["name"]).strip()
        ticker = str(row["ticker"]).strip()
        code = tw_stock_code(ticker)
        label = name or ticker
        if ticker.endswith((".TW", ".TWO")):
            query = f'"{name}" OR "{code}" 股票'
        else:
            query = f'"{name}" OR "{ticker}" stock earnings'
        rows.append((label, query))
    return rows


@st.cache_data(ttl=60 * 30, show_spinner=False)
def fetch_portfolio_news(
    holdings_records: tuple[tuple[str, str], ...],
    per_query_limit: int,
    include_macro: bool,
    language: str,
) -> pd.DataFrame:
    all_rows = []
    for label, query in holdings_records:
        for row in fetch_google_news(query, per_query_limit, language):
            row["分類"] = "持股新聞"
            row["標的"] = label
            all_rows.append(row)

    if include_macro:
        macro_queries = [
            "世界經濟 利率 通膨 聯準會 美元",
            "global economy interest rates inflation Federal Reserve markets",
            "台灣經濟 匯率 半導體 景氣",
        ]
        for query in macro_queries:
            for row in fetch_google_news(query, per_query_limit, language):
                row["分類"] = "世界經濟"
                row["標的"] = "Macro"
                all_rows.append(row)

    if not all_rows:
        return pd.DataFrame(columns=["分類", "標的", "時間", "來源", "標題", "摘要", "連結", "查詢"])
    news = pd.DataFrame(all_rows)
    news = news.drop_duplicates(subset=["標題", "來源"], keep="first")
    return news[["分類", "標的", "時間", "來源", "標題", "摘要", "連結", "查詢"]]


@st.cache_data(ttl=60 * 60, show_spinner=False)
def download_macro_series(start: str, end: str, cache_version: str) -> pd.DataFrame:
    pieces = []
    for label, ticker in MACRO_INDICATORS.items():
        try:
            data = yf.download(
                ticker,
                start=start,
                end=(pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                auto_adjust=False,
                actions=False,
                progress=False,
                group_by="column",
                threads=False,
            )
            if data.empty:
                continue
            close = data["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = pd.to_numeric(close, errors="coerce").dropna()
            if close.empty:
                continue
            close.index = pd.to_datetime(close.index).tz_localize(None)
            pieces.append(close.sort_index().rename(label))
        except Exception:
            continue

    if not pieces:
        raise RuntimeError("yfinance 沒有回傳總經資料。")
    return pd.concat(pieces, axis=1).sort_index().ffill().dropna(how="all")


def macro_summary(series: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label in series.columns:
        data = series[label].dropna()
        if data.empty:
            continue
        latest = float(data.iloc[-1])
        first = float(data.iloc[0])
        change = latest - first
        change_pct = latest / first - 1 if first else np.nan
        rows.append(
            {
                "指標": label,
                "最新日期": data.index[-1].strftime("%Y-%m-%d"),
                "最新值": latest,
                "區間變化": change,
                "區間變化率": change_pct,
            }
        )
    return pd.DataFrame(rows)


def add_yield_spreads(series: pd.DataFrame) -> pd.DataFrame:
    out = series.copy()
    if "美債 10Y" in out.columns and "美債 2Y" in out.columns:
        out["10Y-2Y 利差"] = out["美債 10Y"] - out["美債 2Y"]
    if "美債 10Y" in out.columns and "美債 3M" in out.columns:
        out["10Y-3M 利差"] = out["美債 10Y"] - out["美債 3M"]
    if "美債 30Y" in out.columns and "美債 10Y" in out.columns:
        out["30Y-10Y 利差"] = out["美債 30Y"] - out["美債 10Y"]
    if "Fed Funds Futures" in out.columns:
        out["Fed Funds 隱含利率"] = 100 - out["Fed Funds Futures"]
    return out


def format_macro_table(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.copy()
    for col in ["最新值", "區間變化"]:
        formatted[col] = formatted[col].map(lambda x: "" if pd.isna(x) else f"{x:,.2f}")
    formatted["區間變化率"] = formatted["區間變化率"].map(lambda x: "" if pd.isna(x) else f"{x:.2%}")
    return formatted


def vera_rubin_supply_chain_df() -> pd.DataFrame:
    return pd.DataFrame(VERA_RUBIN_SUPPLY_CHAIN)


def ensure_vera_custom_schema(rows: pd.DataFrame) -> pd.DataFrame:
    df = rows.copy() if rows is not None and not rows.empty else pd.DataFrame()
    defaults = {
        "category": "",
        "company": "",
        "ticker": "",
        "role": "",
        "source": "custom",
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
    df["ticker"] = df["ticker"].astype(str).str.strip()
    return df[["category", "company", "ticker", "role", "source"]]


def csp_supply_chain_df() -> pd.DataFrame:
    return pd.DataFrame(CSP_SUPPLY_CHAIN)


def ensure_csp_custom_schema(rows: pd.DataFrame) -> pd.DataFrame:
    df = rows.copy() if rows is not None and not rows.empty else pd.DataFrame()
    defaults = {
        "csp": "",
        "category": "",
        "company": "",
        "ticker": "",
        "role": "",
        "source": "custom",
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
    df["ticker"] = df["ticker"].astype(str).str.strip()
    return df[["csp", "category", "company", "ticker", "role", "source"]]


def price_performance_summary(prices: pd.DataFrame) -> pd.DataFrame:
    rows = []
    returns = prices.pct_change(fill_method=None)
    for ticker in prices.columns:
        series = prices[ticker].dropna()
        ret = returns[ticker].dropna()
        if series.empty:
            continue
        years = max((series.index[-1] - series.index[0]).days / 365.25, 1 / 365.25)
        total_return = series.iloc[-1] / series.iloc[0] - 1 if series.iloc[0] > 0 else np.nan
        cagr = (series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1 if series.iloc[0] > 0 else np.nan
        drawdown = series / series.cummax() - 1
        rows.append(
            {
                "Ticker": ticker,
                "起始價": series.iloc[0],
                "最新價": series.iloc[-1],
                "區間報酬": total_return,
                "年化報酬": cagr,
                "年化波動": ret.std(ddof=1) * math.sqrt(TRADING_DAYS) if len(ret) > 1 else np.nan,
                "最大回撤": drawdown.min(),
            }
        )
    return pd.DataFrame(rows)


def format_price_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["起始價", "最新價"]:
        out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:,.2f}")
    for col in ["區間報酬", "年化報酬", "年化波動", "最大回撤"]:
        out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.2%}")
    return out


@st.cache_data(ttl=60 * 60 * 8, show_spinner=False)
def download_analyst_targets(tickers: tuple[str, ...]) -> dict[str, dict[str, float]]:
    targets = {}
    for ticker in tickers:
        try:
            raw = yf.Ticker(ticker).analyst_price_targets
            if isinstance(raw, dict) and raw:
                clean = {}
                for key, value in raw.items():
                    if value is None or not pd.notna(value):
                        continue
                    try:
                        clean[key] = float(value)
                    except (TypeError, ValueError):
                        clean[key] = str(value)
                if clean:
                    targets[ticker] = clean
        except Exception:
            continue
    return targets


@st.cache_data(ttl=60 * 60 * 8, show_spinner=False)
def download_trailing_eps(tickers: tuple[str, ...]) -> dict[str, float]:
    eps = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            value = info.get("trailingEps") if isinstance(info, dict) else None
            if value is not None and pd.notna(value):
                eps[ticker] = float(value)
        except Exception:
            continue
    return eps


def naive_datetime_index(index) -> pd.DatetimeIndex:
    idx = pd.to_datetime(index)
    if getattr(idx, "tz", None) is not None:
        return idx.tz_convert(None)
    return idx.tz_localize(None)


def ttm_from_quarterly_eps(eps: pd.Series) -> pd.Series:
    eps = pd.to_numeric(eps, errors="coerce").dropna()
    if eps.empty:
        return pd.Series(dtype=float)
    eps.index = naive_datetime_index(eps.index)
    eps = eps.sort_index()
    eps = eps[~eps.index.duplicated(keep="last")]
    return eps.rolling("365D", min_periods=4).sum().dropna()


def month_starts(start: str, end: str) -> list[pd.Timestamp]:
    start_ts = pd.Timestamp(start).replace(day=1)
    end_ts = pd.Timestamp(end).replace(day=1)
    return list(pd.date_range(start_ts, end_ts, freq="MS"))


def parse_twse_date(value):
    text = str(value).strip()
    if not text:
        return pd.NaT
    text = text.replace("年", "/").replace("月", "/").replace("日", "")
    parts = text.replace("-", "/").split("/")
    try:
        if len(parts) == 3:
            year = int(parts[0])
            if year < 1911:
                year += 1911
            return pd.Timestamp(year, int(parts[1]), int(parts[2]))
        return pd.to_datetime(text, errors="coerce")
    except Exception:
        return pd.NaT


def parse_number(value) -> float:
    text = str(value).replace(",", "").replace("--", "").replace("-", "").strip()
    if not text:
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


@st.cache_data(ttl=60 * 60 * 8, show_spinner=False)
def download_twse_pe_series(ticker: str, start: str, end: str) -> pd.Series:
    if not ticker.endswith(".TW"):
        return pd.Series(dtype=float)

    code = tw_stock_code(ticker)
    rows = []
    for month in month_starts(start, end):
        date_text = month.strftime("%Y%m%d")
        urls = [
            f"https://www.twse.com.tw/rwd/zh/afterTrading/BWIBBU?date={date_text}&stockNo={code}&response=json",
            f"https://www.twse.com.tw/exchangeReport/BWIBBU?date={date_text}&stockNo={code}&response=json",
        ]
        for url in urls:
            try:
                response = requests.get(url, timeout=15)
                if not response.ok:
                    continue
                payload = response.json()
                data = payload.get("data") or payload.get("aaData") or []
                fields = payload.get("fields") or payload.get("headers") or []
                if not data:
                    continue
                for item in data:
                    if isinstance(item, dict):
                        record = item
                    else:
                        record = dict(zip(fields, item))
                    date_value = record.get("日期") or record.get("Date")
                    pe_value = record.get("本益比") or record.get("P/E ratio") or record.get("PEratio")
                    dt = parse_twse_date(date_value)
                    pe = parse_number(pe_value)
                    if pd.notna(dt) and pd.notna(pe) and pe > 0:
                        rows.append((dt, pe))
                break
            except Exception:
                continue

    if not rows:
        return pd.Series(dtype=float)
    pe = pd.Series(dict(rows)).sort_index()
    pe = pe[(pe.index >= pd.Timestamp(start)) & (pe.index <= pd.Timestamp(end))]
    return pe[~pe.index.duplicated(keep="last")]


@st.cache_data(ttl=60 * 60 * 8, show_spinner=False)
def download_historical_eps_ttm(tickers: tuple[str, ...]) -> dict[str, pd.Series]:
    out = {}
    preferred_rows = ["Diluted EPS", "DilutedEPS", "Basic EPS", "BasicEPS"]
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            eps_candidates = []

            earnings_history = stock.get_earnings_history(as_dict=False)
            if earnings_history is not None and not earnings_history.empty:
                actual_col = next(
                    (col for col in earnings_history.columns if str(col).replace(" ", "").lower() in {"epsactual", "reportedeps"}),
                    None,
                )
                if actual_col is not None:
                    eps_candidates.append(pd.to_numeric(earnings_history[actual_col], errors="coerce").dropna())

            earnings_dates = stock.get_earnings_dates(limit=100)
            if earnings_dates is not None and not earnings_dates.empty:
                reported_col = next(
                    (col for col in earnings_dates.columns if str(col).replace(" ", "").lower() in {"reportedeps", "epsactual"}),
                    None,
                )
                if reported_col is not None:
                    eps_candidates.append(pd.to_numeric(earnings_dates[reported_col], errors="coerce").dropna())

            income = stock.quarterly_income_stmt
            if income is not None and not income.empty:
                eps_row = None
                normalized = {str(idx).replace(" ", "").lower(): idx for idx in income.index}
                for name in preferred_rows:
                    key = name.replace(" ", "").lower()
                    if key in normalized:
                        eps_row = normalized[key]
                        break
                if eps_row is not None:
                    eps_candidates.append(pd.to_numeric(income.loc[eps_row], errors="coerce").dropna())
                elif "netincome" in normalized and "dilutedaverageshares" in normalized:
                    net_income = pd.to_numeric(income.loc[normalized["netincome"]], errors="coerce")
                    shares = pd.to_numeric(income.loc[normalized["dilutedaverageshares"]], errors="coerce")
                    eps_candidates.append((net_income / shares).dropna())
                elif "netincome" in normalized and "basicaverageshares" in normalized:
                    net_income = pd.to_numeric(income.loc[normalized["netincome"]], errors="coerce")
                    shares = pd.to_numeric(income.loc[normalized["basicaverageshares"]], errors="coerce")
                    eps_candidates.append((net_income / shares).dropna())

            ttm_candidates = [ttm_from_quarterly_eps(eps) for eps in eps_candidates if eps is not None and not eps.empty]
            ttm_candidates = [ttm for ttm in ttm_candidates if not ttm.empty]
            if not ttm_candidates:
                continue
            ttm = max(ttm_candidates, key=len)
            if not ttm.empty:
                out[ticker] = ttm
        except Exception:
            continue
    return out


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
        "eps_ttm": 0.0,
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
                    "eps_ttm",
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
        "eps_ttm": 0.0,
        "fcf_per_share": 0.0,
        "growth_1_5": 0.10,
        "terminal_growth": 0.02,
        "discount_rate": 0.10,
        "net_cash_per_share": 0.0,
        "margin_safety": 0.25,
    }
    for col, default in defaults.items():
        if col not in base.columns:
            base[col] = default
        base[col] = pd.to_numeric(base[col], errors="coerce").fillna(default)
    return base[
        [
            "name",
            "ticker",
            "currency",
            "target_price",
            "eps_ttm",
            "fcf_per_share",
            "growth_1_5",
            "terminal_growth",
            "discount_rate",
            "net_cash_per_share",
            "margin_safety",
        ]
    ]


def ensure_target_reports_schema(reports: pd.DataFrame, holdings: pd.DataFrame) -> pd.DataFrame:
    holding_tickers = ensure_holdings_schema(holdings)["ticker"].dropna().astype(str).str.strip().tolist()
    df = reports.copy() if reports is not None and not reports.empty else pd.DataFrame()
    defaults = {
        "ticker": holding_tickers[0] if holding_tickers else "",
        "institution": "",
        "report_date": "",
        "rating": "",
        "target_price": 0.0,
        "source_url": "",
        "note": "",
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["target_price"] = pd.to_numeric(df["target_price"], errors="coerce").fillna(0.0)
    return df[["ticker", "institution", "report_date", "rating", "target_price", "source_url", "note"]]


def target_report_summary(reports: pd.DataFrame, holdings: pd.DataFrame) -> pd.DataFrame:
    reports = ensure_target_reports_schema(reports, holdings)
    valid = reports[reports["target_price"] > 0].copy()
    if valid.empty:
        return pd.DataFrame(columns=["ticker", "manual_count", "manual_mean", "manual_median", "manual_low", "manual_high", "manual_latest"])
    valid["report_date_dt"] = pd.to_datetime(valid["report_date"], errors="coerce")
    grouped = valid.groupby("ticker")["target_price"]
    latest = (
        valid.sort_values(["ticker", "report_date_dt"])
        .dropna(subset=["report_date_dt"])
        .groupby("ticker")
        .tail(1)
        .set_index("ticker")["target_price"]
    )
    summary = pd.DataFrame(
        {
            "manual_count": grouped.count(),
            "manual_mean": grouped.mean(),
            "manual_median": grouped.median(),
            "manual_low": grouped.min(),
            "manual_high": grouped.max(),
        }
    )
    summary["manual_latest"] = latest
    return summary.reset_index()


def format_count(value) -> str:
    if pd.isna(value):
        return ""
    try:
        return f"{int(value)}"
    except (TypeError, ValueError):
        return str(value)


def parse_pe_multiples(text: str) -> list[float]:
    values = []
    for part in text.replace("，", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = float(part)
            if value > 0:
                values.append(value)
        except ValueError:
            continue
    return sorted(dict.fromkeys(values))


def pe_river_figure(
    series: pd.Series,
    eps_ttm: pd.Series,
    pe_multiples: list[float],
    title: str,
    y_title: str,
) -> go.Figure:
    fig = go.Figure()
    aligned = pd.concat({"value": series, "eps": eps_ttm}, axis=1).dropna()
    if aligned.empty:
        fig.update_layout(title=title, yaxis_title=y_title, hovermode="x unified")
        return fig

    for pe in pe_multiples:
        fig.add_trace(
            go.Scatter(
                x=aligned.index,
                y=aligned["eps"] * pe,
                mode="lines",
                name=f"{pe:g}x",
                line={"width": 1},
                line_shape="hv",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=aligned.index,
            y=aligned["value"],
            mode="lines",
            name="價格/市值",
            line={"width": 3, "color": "#111827"},
        )
    )
    fig.update_layout(title=title, yaxis_title=y_title, hovermode="x unified")
    return fig


def inferred_position_shares(holdings: pd.DataFrame, latest_prices: dict[str, float]) -> dict[str, float]:
    df = ensure_holdings_schema(holdings)
    out = {}
    for row in df.to_dict("records"):
        ticker = row["ticker"]
        shares = float(pd.to_numeric(row["shares"], errors="coerce")) if pd.notna(row["shares"]) else 0.0
        if shares <= 0:
            latest = latest_prices.get(ticker, np.nan)
            amount = float(pd.to_numeric(row["amount"], errors="coerce")) if pd.notna(row["amount"]) else 0.0
            shares = amount / latest if pd.notna(latest) and latest > 0 else 0.0
        out[ticker] = shares
    return out


def eps_series_for_prices(
    ticker: str,
    price_index: pd.DatetimeIndex,
    price_series: pd.Series,
    official_pe: dict[str, pd.Series],
    historical_eps: dict[str, pd.Series],
    manual_eps: float,
    report_lag_days: int,
) -> pd.Series:
    pe = official_pe.get(ticker)
    if pe is not None and not pe.empty:
        aligned_price = price_series.reindex(price_index).ffill()
        aligned_pe = pe.reindex(price_index.union(pe.index)).sort_index().ffill().reindex(price_index)
        implied_eps = aligned_price / aligned_pe
        return implied_eps.where(implied_eps > 0)

    eps = historical_eps.get(ticker)
    if eps is not None and not eps.empty:
        shifted = eps.copy()
        shifted.index = shifted.index + pd.to_timedelta(report_lag_days, unit="D")
        series = shifted.reindex(price_index.union(shifted.index)).sort_index().ffill().reindex(price_index)
        return series.where(series > 0)
    if manual_eps > 0:
        return pd.Series(manual_eps, index=price_index)
    return pd.Series(np.nan, index=price_index)


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
        "eps_ttm",
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


def target_reports_payload(reports: pd.DataFrame, holdings: pd.DataFrame) -> list[dict]:
    reports = ensure_target_reports_schema(reports, holdings)
    reports["target_price"] = pd.to_numeric(reports["target_price"], errors="coerce").fillna(0.0)
    return reports.to_dict("records")


def vera_custom_chain_payload(custom_chain: pd.DataFrame | None = None) -> list[dict]:
    return ensure_vera_custom_schema(custom_chain if custom_chain is not None else pd.DataFrame()).to_dict("records")


def csp_custom_chain_payload(custom_chain: pd.DataFrame | None = None) -> list[dict]:
    return ensure_csp_custom_schema(custom_chain if custom_chain is not None else pd.DataFrame()).to_dict("records")


def app_settings_payload(
    holdings: pd.DataFrame,
    valuation: pd.DataFrame,
    reports: pd.DataFrame | None = None,
    custom_chain: pd.DataFrame | None = None,
    csp_custom_chain: pd.DataFrame | None = None,
) -> dict:
    holdings = ensure_holdings_schema(holdings)
    valuation = ensure_valuation_schema(valuation, holdings)
    reports = ensure_target_reports_schema(reports if reports is not None else pd.DataFrame(), holdings)
    return {
        "holdings": portfolio_settings_payload(holdings),
        "valuation": valuation_settings_payload(valuation),
        "target_reports": target_reports_payload(reports, holdings),
        "vera_custom_chain": vera_custom_chain_payload(custom_chain),
        "csp_custom_chain": csp_custom_chain_payload(csp_custom_chain),
    }


def parse_settings_payload(settings) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if isinstance(settings, dict):
        holdings = ensure_holdings_schema(pd.DataFrame(settings.get("holdings", [])))
        valuation = ensure_valuation_schema(pd.DataFrame(settings.get("valuation", [])), holdings)
        reports = ensure_target_reports_schema(pd.DataFrame(settings.get("target_reports", [])), holdings)
        custom_chain = ensure_vera_custom_schema(pd.DataFrame(settings.get("vera_custom_chain", [])))
        csp_custom_chain = ensure_csp_custom_schema(pd.DataFrame(settings.get("csp_custom_chain", [])))
        return holdings, valuation, reports, custom_chain, csp_custom_chain
    holdings = ensure_holdings_schema(pd.DataFrame(settings))
    return (
        holdings,
        ensure_valuation_schema(pd.DataFrame(), holdings),
        ensure_target_reports_schema(pd.DataFrame(), holdings),
        ensure_vera_custom_schema(pd.DataFrame()),
        ensure_csp_custom_schema(pd.DataFrame()),
    )


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


def load_settings_from_db(user_key: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    url = f"{supabase_rest_url()}/portfolio_settings"
    params = f"?user_key=eq.{quote(user_key)}&select=settings&limit=1"
    response = requests.get(url + params, headers=supabase_headers(), timeout=15)
    raise_supabase_error(response)
    rows = response.json()
    if not rows:
        return None
    return parse_settings_payload(rows[0]["settings"])


def save_settings_to_db(
    user_key: str,
    holdings: pd.DataFrame,
    valuation: pd.DataFrame,
    reports: pd.DataFrame,
    custom_chain: pd.DataFrame,
    csp_custom_chain: pd.DataFrame,
) -> None:
    url = f"{supabase_rest_url()}/portfolio_settings"
    headers = supabase_headers() | {"Prefer": "resolution=merge-duplicates"}
    body = {
        "user_key": user_key,
        "settings": app_settings_payload(holdings, valuation, reports, custom_chain, csp_custom_chain),
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
if "target_reports" not in st.session_state:
    st.session_state.target_reports = ensure_target_reports_schema(pd.DataFrame(), st.session_state.editor_data)
if "target_reports_key" not in st.session_state:
    st.session_state.target_reports_key = 0
if "vera_custom_chain" not in st.session_state:
    st.session_state.vera_custom_chain = ensure_vera_custom_schema(pd.DataFrame())
if "vera_custom_chain_key" not in st.session_state:
    st.session_state.vera_custom_chain_key = 0
if "csp_custom_chain" not in st.session_state:
    st.session_state.csp_custom_chain = ensure_csp_custom_schema(pd.DataFrame())
if "csp_custom_chain_key" not in st.session_state:
    st.session_state.csp_custom_chain_key = 0
if "latest_prices" not in st.session_state:
    st.session_state.latest_prices = {}
st.session_state.valuation_data = ensure_valuation_schema(
    st.session_state.valuation_data,
    st.session_state.editor_data,
)
st.session_state.target_reports = ensure_target_reports_schema(
    st.session_state.target_reports,
    st.session_state.editor_data,
)
st.session_state.vera_custom_chain = ensure_vera_custom_schema(st.session_state.vera_custom_chain)
st.session_state.csp_custom_chain = ensure_csp_custom_schema(st.session_state.csp_custom_chain)
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
                        loaded_holdings, loaded_valuation, loaded_reports, loaded_custom_chain, loaded_csp_custom_chain = loaded
                        st.session_state.holdings_default = ensure_holdings_schema(loaded_holdings)
                        st.session_state.editor_data = ensure_holdings_schema(loaded_holdings)
                        st.session_state.valuation_data = ensure_valuation_schema(loaded_valuation, loaded_holdings)
                        st.session_state.target_reports = ensure_target_reports_schema(loaded_reports, loaded_holdings)
                        st.session_state.vera_custom_chain = ensure_vera_custom_schema(loaded_custom_chain)
                        st.session_state.csp_custom_chain = ensure_csp_custom_schema(loaded_csp_custom_chain)
                        st.session_state.editor_key += 1
                        st.session_state.valuation_key += 1
                        st.session_state.target_reports_key += 1
                        st.session_state.vera_custom_chain_key += 1
                        st.session_state.csp_custom_chain_key += 1
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
                        st.session_state.target_reports,
                        st.session_state.vera_custom_chain,
                        st.session_state.csp_custom_chain,
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
            loaded_holdings, loaded_valuation, loaded_reports, loaded_custom_chain, loaded_csp_custom_chain = parse_settings_payload(rows)
            required = ["name", "ticker", "amount", "currency"]
            if not set(required).issubset(loaded_holdings.columns):
                st.error("設定檔需要包含 name、ticker、amount、currency 欄位。")
            else:
                st.session_state.holdings_default = loaded_holdings.copy()
                st.session_state.editor_data = loaded_holdings.copy()
                st.session_state.valuation_data = ensure_valuation_schema(loaded_valuation, loaded_holdings)
                st.session_state.target_reports = ensure_target_reports_schema(loaded_reports, loaded_holdings)
                st.session_state.vera_custom_chain = ensure_vera_custom_schema(loaded_custom_chain)
                st.session_state.csp_custom_chain = ensure_csp_custom_schema(loaded_csp_custom_chain)
                st.session_state.editor_key += 1
                st.session_state.valuation_key += 1
                st.session_state.target_reports_key += 1
                st.session_state.vera_custom_chain_key += 1
                st.session_state.csp_custom_chain_key += 1
                st.success("已載入你的設定，本次使用這份作為預設。")
                st.rerun()
        except Exception as exc:
            st.error(f"設定檔讀取失敗：{exc}")

    st.download_button(
        "下載目前預設設定 JSON",
        json.dumps(
            app_settings_payload(
                st.session_state.editor_data,
                st.session_state.valuation_data,
                st.session_state.target_reports,
                st.session_state.vera_custom_chain,
                st.session_state.csp_custom_chain,
            ),
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
    research_col1, research_col2, research_col3 = st.columns(3)
    current_tickers = tuple(t for t in dict.fromkeys(edited["ticker"].dropna().astype(str).str.strip()) if t)

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

    if research_col3.button("抓 EPS TTM"):
        try:
            eps_data = download_trailing_eps(current_tickers)
            if not eps_data:
                st.warning("yfinance 沒有回傳可用 EPS，請手動填 EPS TTM。")
            else:
                valuation_for_eps = ensure_valuation_schema(st.session_state.valuation_data, st.session_state.editor_data)
                valuation_for_eps["eps_ttm"] = valuation_for_eps.apply(
                    lambda r: eps_data.get(r["ticker"], r["eps_ttm"]),
                    axis=1,
                )
                st.session_state.valuation_data = valuation_for_eps
                st.session_state.valuation_key += 1
                st.success("已更新 EPS TTM。")
                st.rerun()
        except Exception as exc:
            st.warning(f"EPS 更新失敗：{exc}")

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
                "eps_ttm": st.column_config.NumberColumn("EPS TTM", step=1.0),
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

    st.markdown("**多筆法人目標價**")
    st.caption("可以逐筆填入不同券商或研究來源。若 Yahoo 目標價缺資料，這張表會是主要參考。")
    with st.form("target_reports_form"):
        target_reports_raw = st.data_editor(
            st.session_state.target_reports,
            num_rows="dynamic",
            use_container_width=True,
            key=f"target_reports_editor_{st.session_state.target_reports_key}",
            column_config={
                "ticker": st.column_config.SelectboxColumn("Ticker", options=list(current_tickers) if current_tickers else [""]),
                "institution": st.column_config.TextColumn("券商/來源"),
                "report_date": st.column_config.TextColumn("日期"),
                "rating": st.column_config.TextColumn("評等"),
                "target_price": st.column_config.NumberColumn("目標價", min_value=0.0, step=1.0),
                "source_url": st.column_config.LinkColumn("來源連結"),
                "note": st.column_config.TextColumn("備註"),
            },
        )
        apply_target_reports = st.form_submit_button("套用目標價資料")

    if apply_target_reports:
        st.session_state.target_reports = ensure_target_reports_schema(target_reports_raw, st.session_state.editor_data)
        st.success("已套用目標價資料。")

    valuation = ensure_valuation_schema(st.session_state.valuation_data, st.session_state.editor_data)
    analyst_targets = st.session_state.get("analyst_targets", {})
    manual_targets = target_report_summary(st.session_state.target_reports, st.session_state.editor_data)
    manual_targets = manual_targets.set_index("ticker") if not manual_targets.empty else pd.DataFrame()
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
        yahoo_data = analyst_targets.get(ticker, {})
        yahoo_current = yahoo_data.get("current", np.nan)
        yahoo_low = yahoo_data.get("low", np.nan)
        yahoo_high = yahoo_data.get("high", np.nan)
        yahoo_mean = yahoo_data.get("mean", np.nan)
        yahoo_median = yahoo_data.get("median", np.nan)
        yahoo_count = yahoo_data.get("numberOfAnalystOpinions", np.nan)
        manual_target = float(row["target_price"])
        manual_count = manual_mean = manual_median = manual_low = manual_high = manual_latest = np.nan
        if not manual_targets.empty and ticker in manual_targets.index:
            manual_row = manual_targets.loc[ticker]
            manual_count = manual_row.get("manual_count", np.nan)
            manual_mean = manual_row.get("manual_mean", np.nan)
            manual_median = manual_row.get("manual_median", np.nan)
            manual_low = manual_row.get("manual_low", np.nan)
            manual_high = manual_row.get("manual_high", np.nan)
            manual_latest = manual_row.get("manual_latest", np.nan)
        preferred_target = manual_mean if pd.notna(manual_mean) else manual_target if manual_target > 0 else yahoo_mean
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
                "多筆目標價數": manual_count,
                "多筆平均": manual_mean,
                "多筆中位數": manual_median,
                "多筆最低": manual_low,
                "多筆最高": manual_high,
                "最新一筆": manual_latest,
                "Yahoo目前": yahoo_current,
                "Yahoo平均": yahoo_mean,
                "Yahoo中位數": yahoo_median,
                "Yahoo最低": yahoo_low,
                "Yahoo最高": yahoo_high,
                "Yahoo分析師數": yahoo_count,
                "目標價空間": upside,
                "法說會": mops_link(ticker) if ticker.endswith((".TW", ".TWO")) else "",
            }
        )

    valuation_result = pd.DataFrame(rows)
    for col in [
        "最新價",
        "DCF合理價",
        "安全買進價",
        "手動目標價",
        "多筆平均",
        "多筆中位數",
        "多筆最低",
        "多筆最高",
        "最新一筆",
        "Yahoo目前",
        "Yahoo平均",
        "Yahoo中位數",
        "Yahoo最低",
        "Yahoo最高",
    ]:
        valuation_result[col] = valuation_result[col].map(lambda x: "" if pd.isna(x) else f"{x:,.2f}")
    valuation_result["多筆目標價數"] = valuation_result["多筆目標價數"].map(format_count)
    valuation_result["Yahoo分析師數"] = valuation_result["Yahoo分析師數"].map(format_count)
    for col in ["DCF空間", "目標價空間"]:
        valuation_result[col] = valuation_result[col].map(lambda x: "" if pd.isna(x) else f"{x:.2%}")

    st.dataframe(
        valuation_result,
        use_container_width=True,
        column_config={
            "法說會": st.column_config.LinkColumn("法說會"),
        },
    )
    if analyst_targets:
        yahoo_rows = []
        for ticker, data in analyst_targets.items():
            yahoo_rows.append({"Ticker": ticker, **data})
        yahoo_summary = pd.DataFrame(yahoo_rows)
        price_cols = [col for col in yahoo_summary.columns if col != "Ticker"]
        for col in price_cols:
            yahoo_summary[col] = yahoo_summary[col].map(lambda x: "" if pd.isna(x) else f"{x:,.2f}")
        st.markdown("**Yahoo 目標價摘要明細**")
        st.dataframe(yahoo_summary, use_container_width=True)

    detailed_reports = ensure_target_reports_schema(st.session_state.target_reports, st.session_state.editor_data)
    detailed_reports = detailed_reports[detailed_reports["target_price"] > 0].copy()
    if not detailed_reports.empty:
        st.markdown("**目標價明細**")
        st.dataframe(
            detailed_reports.rename(
                columns={
                    "ticker": "Ticker",
                    "institution": "券商/來源",
                    "report_date": "日期",
                    "rating": "評等",
                    "target_price": "目標價",
                    "source_url": "來源連結",
                    "note": "備註",
                }
            ),
            use_container_width=True,
            column_config={"來源連結": st.column_config.LinkColumn("來源連結")},
        )
    st.caption("Yahoo 目標價是摘要資料，不一定含券商明細；若要有可追溯性，請在多筆法人目標價表填入來源連結。")

with st.expander("Latest News", expanded=False):
    st.caption("使用 Google News RSS 搜尋持股相關新聞與世界經濟新聞；按鈕觸發抓取，避免首頁載入過慢。")
    news_col1, news_col2, news_col3 = st.columns(3)
    per_query_limit = news_col1.number_input("每個查詢最多篇數", min_value=3, max_value=30, value=8, step=1)
    news_language = news_col2.selectbox("新聞語言", ["zh-TW", "en-US"])
    include_macro_news = news_col3.checkbox("包含世界經濟新聞", value=True)
    news_queries = holding_news_queries(st.session_state.editor_data)

    if st.button("更新 Latest News"):
        if not news_queries and not include_macro_news:
            st.warning("沒有持股可以搜尋。")
        else:
            try:
                news = fetch_portfolio_news(
                    tuple(news_queries),
                    int(per_query_limit),
                    bool(include_macro_news),
                    news_language,
                )
                st.session_state.latest_news = news
                st.success(f"已更新新聞，共 {len(news):,} 則。")
            except Exception as exc:
                st.error(f"新聞抓取失敗：{exc}")

    latest_news = st.session_state.get("latest_news", pd.DataFrame())
    if not latest_news.empty:
        selected_categories = st.multiselect(
            "分類",
            sorted(latest_news["分類"].dropna().unique().tolist()),
            default=sorted(latest_news["分類"].dropna().unique().tolist()),
        )
        selected_labels = st.multiselect(
            "標的",
            sorted(latest_news["標的"].dropna().unique().tolist()),
            default=sorted(latest_news["標的"].dropna().unique().tolist()),
        )
        filtered_news = latest_news[
            latest_news["分類"].isin(selected_categories) & latest_news["標的"].isin(selected_labels)
        ].copy()
        st.dataframe(
            filtered_news,
            use_container_width=True,
            column_config={"連結": st.column_config.LinkColumn("連結")},
            hide_index=True,
        )
        st.download_button(
            "下載 Latest News CSV",
            filtered_news.to_csv(index=False).encode("utf-8-sig"),
            "latest_news.csv",
            "text/csv",
        )
    else:
        st.info("按「更新 Latest News」後會顯示新聞列表。")

with st.expander("NVIDIA Vera Rubin 供應鏈", expanded=False):
    st.caption("追蹤 NVIDIA Vera Rubin / AI factory 生態系與相關零組件族群；official 是 NVIDIA 點名，watchlist 是產業追蹤，不代表 NVIDIA 已揭露實際訂單或分配比例。")
    with st.form("vera_custom_chain_form"):
        custom_chain_raw = st.data_editor(
            st.session_state.vera_custom_chain,
            num_rows="dynamic",
            use_container_width=True,
            key=f"vera_custom_chain_editor_{st.session_state.vera_custom_chain_key}",
            column_config={
                "category": st.column_config.TextColumn("分類"),
                "company": st.column_config.TextColumn("公司"),
                "ticker": st.column_config.TextColumn("Ticker"),
                "role": st.column_config.TextColumn("供應鏈角色"),
                "source": st.column_config.TextColumn("來源類型"),
            },
        )
        apply_custom_chain = st.form_submit_button("套用自訂供應鏈名單")

    if apply_custom_chain:
        st.session_state.vera_custom_chain = ensure_vera_custom_schema(custom_chain_raw)
        st.success("已套用自訂供應鏈名單。")

    chain = pd.concat(
        [vera_rubin_supply_chain_df(), ensure_vera_custom_schema(st.session_state.vera_custom_chain)],
        ignore_index=True,
    )
    chain = chain.drop_duplicates(subset=["ticker", "company", "category"], keep="last")
    chain_categories = sorted(chain["category"].unique().tolist())
    chain_col1, chain_col2 = st.columns(2)
    selected_chain_categories = chain_col1.multiselect("供應鏈分類", chain_categories, default=chain_categories)
    selected_chain_source = chain_col2.multiselect(
        "資料來源類型",
        sorted(chain["source"].unique().tolist()),
        default=sorted(chain["source"].unique().tolist()),
    )
    filtered_chain = chain[
        chain["category"].isin(selected_chain_categories) & chain["source"].isin(selected_chain_source)
    ].copy()
    st.dataframe(
        filtered_chain.rename(
            columns={
                "category": "分類",
                "company": "公司",
                "ticker": "Ticker",
                "role": "供應鏈角色",
                "source": "來源類型",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    price_col1, price_col2 = st.columns(2)
    chain_start = price_col1.date_input("供應鏈股價開始日", date(2024, 1, 1))
    chain_end = price_col2.date_input("供應鏈股價結束日", date.today())
    chain_tickers = filtered_chain["ticker"].dropna().astype(str).str.strip()
    chain_tickers = tuple(t for t in dict.fromkeys(chain_tickers) if t)

    if st.button("更新 Vera Rubin 供應鏈股價"):
        if not chain_tickers:
            st.warning("目前篩選條件下沒有可抓取的 ticker。")
        else:
            try:
                chain_prices = download_adjusted_prices(chain_tickers, str(chain_start), str(chain_end))
                chain_prices = chain_prices[[ticker for ticker in chain_tickers if ticker in chain_prices.columns]].dropna(how="all")
                st.session_state.vera_rubin_prices = chain_prices
                st.success(f"已更新 {len(chain_prices.columns):,} 檔供應鏈股價。")
            except Exception as exc:
                st.error(f"Vera Rubin 供應鏈股價更新失敗：{exc}")

    chain_prices = st.session_state.get("vera_rubin_prices", pd.DataFrame())
    if not chain_prices.empty:
        visible_chain_tickers = st.multiselect(
            "顯示股票",
            list(chain_prices.columns),
            default=list(chain_prices.columns[: min(10, len(chain_prices.columns))]),
        )
        if visible_chain_tickers:
            normalized_chain = chain_prices[visible_chain_tickers].dropna(how="all")
            normalized_chain = normalized_chain / normalized_chain.ffill().bfill().iloc[0]
            chain_fig = go.Figure()
            for ticker in normalized_chain.columns:
                name = chain.loc[chain["ticker"] == ticker, "company"].dropna()
                label = f"{ticker} {name.iloc[0]}" if not name.empty else ticker
                chain_fig.add_trace(go.Scatter(x=normalized_chain.index, y=normalized_chain[ticker], mode="lines", name=label))
            chain_fig.update_layout(title="Vera Rubin 供應鏈股價走勢，起點=1.00", yaxis_title="Growth of 1.00", hovermode="x unified")
            st.plotly_chart(chain_fig, use_container_width=True)

            summary = price_performance_summary(chain_prices[visible_chain_tickers])
            chain_meta = (
                chain.groupby("ticker", as_index=False)
                .agg(
                    company=("company", "first"),
                    category=("category", lambda x: " / ".join(dict.fromkeys(x.dropna().astype(str)))),
                )
            )
            summary = summary.merge(chain_meta, left_on="Ticker", right_on="ticker", how="left")
            summary = summary.drop(columns=["ticker"]).rename(columns={"company": "公司", "category": "分類"})
            st.dataframe(format_price_summary(summary), use_container_width=True, hide_index=True)
            st.download_button(
                "下載 Vera Rubin 供應鏈股價 CSV",
                chain_prices[visible_chain_tickers].to_csv().encode("utf-8-sig"),
                "vera_rubin_supply_chain_prices.csv",
                "text/csv",
            )
    else:
        st.info("按「更新 Vera Rubin 供應鏈股價」後會顯示走勢與統計。")

with st.expander("CSP 供應鏈", expanded=False):
    st.caption("依 Microsoft、AWS、Google、Meta、Oracle 等 CSP 分別追蹤 AI server / GPU rack / ASIC 供應鏈；watchlist 不代表客戶或訂單已由公司正式確認。")
    with st.form("csp_custom_chain_form"):
        csp_custom_raw = st.data_editor(
            st.session_state.csp_custom_chain,
            num_rows="dynamic",
            use_container_width=True,
            key=f"csp_custom_chain_editor_{st.session_state.csp_custom_chain_key}",
            column_config={
                "csp": st.column_config.TextColumn("CSP"),
                "category": st.column_config.TextColumn("分類"),
                "company": st.column_config.TextColumn("公司"),
                "ticker": st.column_config.TextColumn("Ticker"),
                "role": st.column_config.TextColumn("供應鏈角色"),
                "source": st.column_config.TextColumn("來源類型"),
            },
        )
        apply_csp_custom = st.form_submit_button("套用自訂 CSP 供應鏈")

    if apply_csp_custom:
        st.session_state.csp_custom_chain = ensure_csp_custom_schema(csp_custom_raw)
        st.success("已套用自訂 CSP 供應鏈。")

    csp_chain = pd.concat(
        [csp_supply_chain_df(), ensure_csp_custom_schema(st.session_state.csp_custom_chain)],
        ignore_index=True,
    )
    csp_chain = csp_chain.drop_duplicates(subset=["csp", "ticker", "company", "category"], keep="last")
    csp_col1, csp_col2, csp_col3 = st.columns(3)
    selected_csps = csp_col1.multiselect("CSP", sorted(csp_chain["csp"].dropna().unique().tolist()), default=sorted(csp_chain["csp"].dropna().unique().tolist()))
    selected_csp_categories = csp_col2.multiselect("分類", sorted(csp_chain["category"].dropna().unique().tolist()), default=sorted(csp_chain["category"].dropna().unique().tolist()))
    selected_csp_sources = csp_col3.multiselect("來源類型", sorted(csp_chain["source"].dropna().unique().tolist()), default=sorted(csp_chain["source"].dropna().unique().tolist()))

    filtered_csp_chain = csp_chain[
        csp_chain["csp"].isin(selected_csps)
        & csp_chain["category"].isin(selected_csp_categories)
        & csp_chain["source"].isin(selected_csp_sources)
    ].copy()
    st.dataframe(
        filtered_csp_chain.rename(
            columns={
                "csp": "CSP",
                "category": "分類",
                "company": "公司",
                "ticker": "Ticker",
                "role": "供應鏈角色",
                "source": "來源類型",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    csp_price_col1, csp_price_col2 = st.columns(2)
    csp_start = csp_price_col1.date_input("CSP 供應鏈股價開始日", date(2024, 1, 1))
    csp_end = csp_price_col2.date_input("CSP 供應鏈股價結束日", date.today())
    csp_tickers = filtered_csp_chain["ticker"].dropna().astype(str).str.strip()
    csp_tickers = tuple(t for t in dict.fromkeys(csp_tickers) if t)

    if st.button("更新 CSP 供應鏈股價"):
        if not csp_tickers:
            st.warning("目前篩選條件下沒有可抓取的 ticker。")
        else:
            try:
                csp_prices = download_adjusted_prices(csp_tickers, str(csp_start), str(csp_end))
                csp_prices = csp_prices[[ticker for ticker in csp_tickers if ticker in csp_prices.columns]].dropna(how="all")
                st.session_state.csp_chain_prices = csp_prices
                st.success(f"已更新 {len(csp_prices.columns):,} 檔 CSP 供應鏈股價。")
            except Exception as exc:
                st.error(f"CSP 供應鏈股價更新失敗：{exc}")

    csp_prices = st.session_state.get("csp_chain_prices", pd.DataFrame())
    if not csp_prices.empty:
        visible_csp_tickers = st.multiselect(
            "顯示 CSP 供應鏈股票",
            list(csp_prices.columns),
            default=list(csp_prices.columns[: min(10, len(csp_prices.columns))]),
        )
        if visible_csp_tickers:
            normalized_csp = csp_prices[visible_csp_tickers].dropna(how="all")
            normalized_csp = normalized_csp / normalized_csp.ffill().bfill().iloc[0]
            csp_fig = go.Figure()
            for ticker in normalized_csp.columns:
                name = csp_chain.loc[csp_chain["ticker"] == ticker, "company"].dropna()
                label = f"{ticker} {name.iloc[0]}" if not name.empty else ticker
                csp_fig.add_trace(go.Scatter(x=normalized_csp.index, y=normalized_csp[ticker], mode="lines", name=label))
            csp_fig.update_layout(title="CSP 供應鏈股價走勢，起點=1.00", yaxis_title="Growth of 1.00", hovermode="x unified")
            st.plotly_chart(csp_fig, use_container_width=True)

            csp_meta = (
                csp_chain.groupby("ticker", as_index=False)
                .agg(
                    company=("company", "first"),
                    csp=("csp", lambda x: " / ".join(dict.fromkeys(x.dropna().astype(str)))),
                    category=("category", lambda x: " / ".join(dict.fromkeys(x.dropna().astype(str)))),
                )
            )
            csp_summary = price_performance_summary(csp_prices[visible_csp_tickers])
            csp_summary = csp_summary.merge(csp_meta, left_on="Ticker", right_on="ticker", how="left")
            csp_summary = csp_summary.drop(columns=["ticker"]).rename(columns={"company": "公司", "csp": "CSP", "category": "分類"})
            st.dataframe(format_price_summary(csp_summary), use_container_width=True, hide_index=True)
            st.download_button(
                "下載 CSP 供應鏈股價 CSV",
                csp_prices[visible_csp_tickers].to_csv().encode("utf-8-sig"),
                "csp_supply_chain_prices.csv",
                "text/csv",
            )
    else:
        st.info("按「更新 CSP 供應鏈股價」後會顯示走勢與統計。")

with st.expander("總經儀表板", expanded=False):
    st.caption("使用 yfinance 抓市場型總經指標；FedWatch 以 CME 連結為準，Fed Funds futures 僅作隱含利率參考。")
    macro_col1, macro_col2 = st.columns(2)
    macro_start = macro_col1.date_input("總經資料開始日", date(2023, 1, 1))
    macro_end = macro_col2.date_input("總經資料結束日", date.today())
    st.link_button("CME FedWatch", "https://www.cmegroup.com/markets/interest-rates/cme-fedwatch-tool.html")

    if st.button("更新總經指標"):
        try:
            macro = download_macro_series(str(macro_start), str(macro_end), MACRO_CACHE_VERSION)
            macro = add_yield_spreads(macro)
            st.session_state.macro_data = macro
            st.success("已更新總經指標。")
        except Exception as exc:
            st.error(f"總經指標更新失敗：{exc}")

    macro_data = st.session_state.get("macro_data", pd.DataFrame())
    if not macro_data.empty:
        world_index_labels = [
                "美股 S&P 500",
                "NASDAQ 100",
                "台灣加權指數",
                "日本 Nikkei 225",
            "歐洲 Euro Stoxx 50",
            "英國 FTSE 100",
            "新加坡 STI",
            "中國上證指數",
            "中國A股 ASHR ETF",
            "香港恆生指數",
            "韓國 KOSPI",
            "費城半導體",
        ]
        fx_labels = [
            "USD/TWD",
            "EUR/USD",
            "GBP/USD",
            "USD/JPY",
            "USD/CNY",
            "USD/HKD",
            "USD/SGD",
            "USD/KRW",
            "AUD/USD",
        ]
        core_macro_cols = [
            col
            for col in macro_data.columns
            if col not in world_index_labels
            and col not in fx_labels
            and col != "Fed Funds Futures"
        ]

        if core_macro_cols:
            summary = macro_summary(macro_data[core_macro_cols])
            st.dataframe(format_macro_table(summary), use_container_width=True, hide_index=True)
        else:
            st.info("這次沒有抓到核心總經指標。")
        available = pd.DataFrame(
            {
                "已抓到指標": list(macro_data.columns),
                "資料筆數": [int(macro_data[col].dropna().shape[0]) for col in macro_data.columns],
            }
        )
        st.dataframe(available, use_container_width=True, hide_index=True)

        preferred_macro = [
            "VIX 恐慌指數",
            "美元指數 DXY",
            "黃金期貨",
            "WTI 原油",
            "銅期貨",
            "天然氣",
            "小麥",
            "Fed Funds 隱含利率",
        ]
        selected_macro = st.multiselect(
            "主要指標",
            core_macro_cols,
            default=[col for col in preferred_macro if col in macro_data.columns],
        )
        if selected_macro:
            normalized_macro = macro_data[selected_macro].dropna(how="all")
            normalized_macro = normalized_macro / normalized_macro.ffill().bfill().iloc[0]
            fig = go.Figure()
            for col in normalized_macro.columns:
                fig.add_trace(go.Scatter(x=normalized_macro.index, y=normalized_macro[col], mode="lines", name=col))
            fig.update_layout(title="總經指標走勢，起點=1.00", yaxis_title="Growth of 1.00", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

        world_index_cols = [
            col
            for col in world_index_labels
            if col in macro_data.columns
        ]
        missing_world_indices = [col for col in world_index_labels if col not in macro_data.columns]
        if missing_world_indices:
            st.caption("未抓到的世界股指：" + "、".join(missing_world_indices))
        if world_index_cols:
            world_indices = macro_data[world_index_cols].dropna(how="all")
            world_indices = world_indices / world_indices.ffill().bfill().iloc[0]
            world_fig = go.Figure()
            for col in world_indices.columns:
                world_fig.add_trace(go.Scatter(x=world_indices.index, y=world_indices[col], mode="lines", name=col))
            world_fig.update_layout(title="世界主要股指，起點=1.00", yaxis_title="Growth of 1.00", hovermode="x unified")
            st.plotly_chart(world_fig, use_container_width=True)

        fx_cols = [
            col
            for col in fx_labels
            if col in macro_data.columns
        ]
        if fx_cols:
            fx_data = macro_data[fx_cols].dropna(how="all")
            fx_norm = fx_data / fx_data.ffill().bfill().iloc[0]
            fx_fig = go.Figure()
            for col in fx_norm.columns:
                fx_fig.add_trace(go.Scatter(x=fx_norm.index, y=fx_norm[col], mode="lines", name=col))
            fx_fig.update_layout(title="世界主要匯率，起點=1.00", yaxis_title="Growth of 1.00", hovermode="x unified")
            st.plotly_chart(fx_fig, use_container_width=True)

        spread_cols = [col for col in ["10Y-2Y 利差", "10Y-3M 利差", "30Y-10Y 利差"] if col in macro_data.columns]
        if spread_cols:
            spread_fig = go.Figure()
            for col in spread_cols:
                spread_fig.add_trace(go.Scatter(x=macro_data.index, y=macro_data[col], mode="lines", name=col))
            spread_fig.add_hline(y=0, line_dash="dash", line_color="#9ca3af")
            spread_fig.update_layout(title="美債長短天期利差", yaxis_title="百分點", hovermode="x unified")
            st.plotly_chart(spread_fig, use_container_width=True)

        st.download_button(
            "下載總經指標 CSV",
            macro_data.to_csv().encode("utf-8-sig"),
            "macro_indicators.csv",
            "text/csv",
        )
    else:
        st.info("按「更新總經指標」後會顯示資料。")

with st.expander("個股股價資料", expanded=False):
    price_col1, price_col2 = st.columns(2)
    price_start = price_col1.date_input("股價資料開始日", date(2021, 1, 1))
    price_end = price_col2.date_input("股價資料結束日", date.today())
    selected_tickers = st.multiselect("股票", list(current_tickers), default=list(current_tickers))
    use_adjusted_price = st.checkbox("使用含息調整價格", value=True)

    if st.button("顯示個股股價資料"):
        if not selected_tickers:
            st.warning("請至少選一檔股票。")
        else:
            try:
                if use_adjusted_price:
                    close = download_adjusted_prices(tuple(dict.fromkeys(selected_tickers)), str(price_start), str(price_end))
                else:
                    close = download_close_prices(tuple(dict.fromkeys(selected_tickers)), str(price_start), str(price_end))
                close = close[[ticker for ticker in selected_tickers if ticker in close.columns]].dropna(how="all")
                if close.empty:
                    st.warning("這個區間沒有可用股價資料。")
                else:
                    fig = go.Figure()
                    for ticker in close.columns:
                        fig.add_trace(go.Scatter(x=close.index, y=close[ticker], mode="lines", name=ticker))
                    fig.update_layout(title="個股股價走勢", yaxis_title="價格", hovermode="x unified")
                    st.plotly_chart(fig, use_container_width=True)

                    normalized = close / close.ffill().bfill().iloc[0]
                    norm_fig = go.Figure()
                    for ticker in normalized.columns:
                        norm_fig.add_trace(go.Scatter(x=normalized.index, y=normalized[ticker], mode="lines", name=ticker))
                    norm_fig.update_layout(title="個股報酬走勢，起點=1.00", yaxis_title="Growth of 1.00", hovermode="x unified")
                    st.plotly_chart(norm_fig, use_container_width=True)

                    returns = close.pct_change(fill_method=None)
                    rows = []
                    for ticker in close.columns:
                        series = close[ticker].dropna()
                        ret = returns[ticker].dropna()
                        if series.empty:
                            continue
                        years = max((series.index[-1] - series.index[0]).days / 365.25, 1 / 365.25)
                        total_return = series.iloc[-1] / series.iloc[0] - 1 if series.iloc[0] else np.nan
                        cagr = (series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1 if series.iloc[0] > 0 else np.nan
                        drawdown = series / series.cummax() - 1
                        rows.append(
                            {
                                "Ticker": ticker,
                                "起始日": series.index[0].strftime("%Y-%m-%d"),
                                "結束日": series.index[-1].strftime("%Y-%m-%d"),
                                "起始價": series.iloc[0],
                                "最新價": series.iloc[-1],
                                "區間報酬": total_return,
                                "年化報酬": cagr,
                                "年化波動": ret.std(ddof=1) * math.sqrt(TRADING_DAYS) if len(ret) > 1 else np.nan,
                                "最大回撤": drawdown.min(),
                            }
                        )

                    price_summary = pd.DataFrame(rows)
                    for col in ["起始價", "最新價"]:
                        price_summary[col] = price_summary[col].map(lambda x: "" if pd.isna(x) else f"{x:,.2f}")
                    for col in ["區間報酬", "年化報酬", "年化波動", "最大回撤"]:
                        price_summary[col] = price_summary[col].map(lambda x: "" if pd.isna(x) else f"{x:.2%}")
                    st.dataframe(price_summary, use_container_width=True)
                    st.download_button("下載個股股價 CSV", close.to_csv().encode("utf-8"), "stock_prices.csv", "text/csv")
            except Exception as exc:
                st.error(f"個股股價資料產生失敗：{exc}")

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
