import numpy as np
import streamlit as st
from PIL import Image
from datetime import datetime
import csv
import json
from io import StringIO
import pandas as pd
import altair as alt
import time
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler

from src.core.business import compute_business_metrics
from src.core.config import AppConfig
from src.core.inference import InferenceEngine


MIN_ANALYZE_LOADER_SECONDS = 5.0


@st.cache_resource
def get_engine() -> InferenceEngine:
    return InferenceEngine(AppConfig())


def history_to_csv(rows: list[dict]) -> str:
    if not rows:
        return ""

    # Union keys makes exports robust when older rows are missing newly added columns.
    all_keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in all_keys:
                all_keys.append(key)

    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=all_keys)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def format_money(value: float) -> str:
    return f"{value:,.2f}"


st.set_page_config(
        page_title="Supermarket Vision Studio",
        page_icon="SV",
        layout="wide",
)

with st.sidebar:
    ui_theme = st.radio(
        "UI Theme",
        ["Auto", "Light", "Dark"],
        horizontal=True,
        index=0,
        help="Use this switch if Streamlit menu theme does not update custom cards.",
    )

theme_base = (st.get_option("theme.base") or "light").lower()
is_dark_theme = (
    ui_theme == "Dark"
    or (ui_theme == "Auto" and theme_base == "dark")
)

st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@400;500;600;700&family=Syne:wght@700&display=swap');

        :root {
            --luxury-dark: #182230;
            --luxury-charcoal: #243447;
            --luxury-gold: #b8891a;
            --luxury-gold-light: #dfbf72;
            --luxury-accent: #0a9d6b;
            --luxury-accent-light: #13b88d;
            --luxury-white: #fafbfc;
            --luxury-gray: #5d6b7b;
            --luxury-border: rgba(184, 137, 26, 0.2);
            --luxury-shadow: 0 20px 60px rgba(19, 30, 45, 0.12);
            --luxury-shadow-lg: 0 30px 90px rgba(19, 30, 45, 0.16);
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        .stApp {
            background:
                radial-gradient(950px 480px at 10% -5%, rgba(178, 216, 255, 0.35) 0%, transparent 60%),
                radial-gradient(760px 420px at 100% 0%, rgba(245, 220, 167, 0.35) 0%, transparent 55%),
                linear-gradient(180deg, #f7f4ec 0%, #f8fafc 45%, #eef5f2 100%);
            color: var(--luxury-dark);
            font-family: 'Inter', sans-serif;
            overflow-x: hidden;
        }

        .block-container {
            max-width: 1280px !important;
            padding: 3rem 2rem !important;
        }

        .luxury-hero {
            position: relative;
            padding: 3.5rem 3rem;
            border-radius: 28px;
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.88), rgba(245, 250, 255, 0.9)),
                        radial-gradient(circle at top right, rgba(184, 137, 26, 0.16), transparent 72%);
            border: 1px solid var(--luxury-border);
            box-shadow: var(--luxury-shadow-lg);
            overflow: hidden;
            animation: slideDownIn 800ms cubic-bezier(0.34, 1.56, 0.64, 1);
            margin-bottom: 2.5rem;
        }

        .luxury-hero::before {
            content: '';
            position: absolute;
            top: -50%;
            right: -50%;
            width: 600px;
            height: 600px;
            background: radial-gradient(circle, rgba(184, 137, 26, 0.12), transparent);
            border-radius: 50%;
            pointer-events: none;
        }

        .luxury-hero-content {
            position: relative;
            z-index: 2;
        }

        .luxury-hero h1 {
            font-family: 'Playfair Display', serif;
            font-size: 3.2rem;
            font-weight: 700;
            line-height: 1.15;
            margin-bottom: 1.2rem;
            color: var(--luxury-dark);
            letter-spacing: -0.02em;
            background: linear-gradient(135deg, #1a2a3a 0%, #b8891a 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .luxury-hero p {
            font-size: 1.1rem;
            line-height: 1.6;
            color: rgba(36, 52, 71, 0.9);
            max-width: 720px;
            font-weight: 400;
        }

        .luxury-panel {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.92), rgba(248, 252, 255, 0.9));
            border: 1px solid var(--luxury-border);
            border-radius: 24px;
            padding: 2.2rem;
            box-shadow: 0 16px 48px rgba(31, 46, 64, 0.12);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            animation: fadeInUp 900ms cubic-bezier(0.23, 1, 0.320, 1);
            transition: all 400ms cubic-bezier(0.23, 1, 0.320, 1);
        }

        .luxury-panel:hover {
            border-color: rgba(184, 137, 26, 0.4);
            box-shadow: 0 24px 72px rgba(184, 137, 26, 0.14);
            transform: translateY(-4px);
        }

        .luxury-chip {
            display: inline-block;
            border-radius: 50px;
            padding: 0.5rem 1.1rem;
            font-size: 0.8rem;
            font-weight: 600;
            letter-spacing: 0.06em;
            background: linear-gradient(135deg, rgba(184, 137, 26, 0.14), rgba(10, 157, 107, 0.1));
            color: #6a4e13;
            border: 1px solid rgba(184, 137, 26, 0.3);
            text-transform: uppercase;
            margin-bottom: 1.5rem;
            animation: popIn 600ms cubic-bezier(0.34, 1.56, 0.64, 1) 100ms both;
        }

        .luxury-metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.2rem;
            margin-top: 1.5rem;
        }

        .luxury-metric-card {
            position: relative;
            border-radius: 18px;
            border: 1px solid rgba(184, 137, 26, 0.2);
            background: linear-gradient(135deg, rgba(184, 137, 26, 0.05), rgba(10, 157, 107, 0.04));
            padding: 1.8rem;
            overflow: hidden;
            transition: all 500ms cubic-bezier(0.23, 1, 0.320, 1);
            animation: slideInCard 700ms cubic-bezier(0.23, 1, 0.320, 1);
        }

        .luxury-metric-card::before {
            content: '';
            position: absolute;
            top: -1px;
            left: -1px;
            right: -1px;
            bottom: -1px;
            background: linear-gradient(135deg, rgba(184, 137, 26, 0.24), transparent);
            border-radius: 18px;
            opacity: 0;
            transition: opacity 400ms;
            z-index: -1;
        }

        .luxury-metric-card:hover::before {
            opacity: 1;
        }

        .luxury-metric-card:hover {
            transform: translateY(-6px);
            border-color: rgba(184, 137, 26, 0.42);
        }

        .luxury-metric-label {
            font-size: 0.75rem;
            color: rgba(36, 52, 71, 0.72);
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-weight: 700;
            margin-bottom: 0.8rem;
        }

        .luxury-metric-value {
            font-family: 'Syne', sans-serif;
            font-size: 2rem;
            font-weight: 700;
            color: #2a3b50;
            letter-spacing: -0.01em;
        }

        .luxury-bar-section {
            margin: 2rem 0;
        }

        .luxury-bar-section h3 {
            font-family: 'Playfair Display', serif;
            font-size: 1.4rem;
            color: #1e3044;
            margin-bottom: 1.5rem;
            font-weight: 600;
            letter-spacing: -0.01em;
        }

        .luxury-bar-row {
            margin-bottom: 1.3rem;
            animation: slideInLeft 600ms ease-out;
        }

        .luxury-bar-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.6rem;
            font-size: 0.95rem;
        }

        .luxury-bar-label {
            color: #2c3f55;
            font-weight: 600;
        }

        .luxury-bar-value {
            color: #8d6a19;
            font-weight: 700;
            font-family: 'Syne', sans-serif;
        }

        .luxury-bar-track {
            width: 100%;
            height: 8px;
            background: rgba(50, 70, 92, 0.12);
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid rgba(184, 137, 26, 0.16);
        }

        .luxury-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #c1982e 0%, #0ea479 100%);
            border-radius: 10px;
            animation: slideInToRight 1s cubic-bezier(0.23, 1, 0.320, 1);
            box-shadow: 0 0 14px rgba(184, 137, 26, 0.24);
        }

        .luxury-footnote {
            font-size: 0.8rem;
            color: rgba(39, 55, 74, 0.72);
            margin-top: 1.2rem;
            padding-top: 1.2rem;
            border-top: 1px solid rgba(184, 137, 26, 0.16);
            font-weight: 500;
        }

        .biz-metric-grid {
            margin-top: 1.1rem;
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.85rem;
        }

        .biz-metric-card {
            border: 1px solid rgba(184, 137, 26, 0.18);
            border-radius: 12px;
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.88), rgba(247, 251, 255, 0.9));
            padding: 0.95rem 1rem;
        }

        .biz-metric-label {
            color: rgba(36, 52, 71, 0.72);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        .biz-metric-value {
            color: #22374d;
            font-family: 'Syne', sans-serif;
            font-size: clamp(1rem, 2.4vw, 1.55rem);
            line-height: 1.2;
            font-weight: 700;
            white-space: nowrap;
            overflow: visible;
            text-overflow: clip;
        }

        .luxury-status-badge {
            display: inline-block;
            padding: 0.6rem 1.2rem;
            border-radius: 50px;
            font-size: 0.85rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            animation: popIn 600ms cubic-bezier(0.34, 1.56, 0.64, 1);
        }

        .luxury-status-high {
            background: linear-gradient(135deg, rgba(10, 157, 107, 0.2), rgba(19, 184, 141, 0.12));
            color: #0b7e5a;
            border: 1px solid rgba(19, 184, 141, 0.35);
        }

        .luxury-status-uncertain {
            background: linear-gradient(135deg, rgba(184, 137, 26, 0.22), rgba(223, 191, 114, 0.12));
            color: #735412;
            border: 1px solid rgba(184, 137, 26, 0.34);
        }

        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            color: rgba(36, 52, 71, 0.72) !important;
            font-weight: 600 !important;
        }

        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] [data-testid="stMarkdownContainer"] p {
            color: #1d2f44 !important;
            font-weight: 700 !important;
        }

        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"],
        [data-testid="stMetricDelta"] {
            color: #1f3044 !important;
        }

        [data-testid="stMetricValue"] {
            font-weight: 700 !important;
        }

        .stSelectbox label,
        .stNumberInput label,
        .stSlider label,
        .stDownloadButton label,
        .stButton label {
            color: #2a3d52 !important;
            font-weight: 600 !important;
        }

        .pl-input-title {
            font-family: 'Syne', sans-serif;
            font-size: 1.05rem;
            font-weight: 700;
            color: #1f3044;
            margin-bottom: 0.9rem;
            letter-spacing: 0.01em;
        }

        [data-testid="stNumberInput"] input {
            color: #1f3044 !important;
            font-weight: 600 !important;
        }

        [data-testid="stNumberInput"] div[data-baseweb="input"] {
            background: rgba(255, 255, 255, 0.92) !important;
            border: 1px solid rgba(184, 137, 26, 0.26) !important;
            border-radius: 12px !important;
        }

        [data-testid="stAlert"] {
            background: rgba(255, 242, 191, 0.7) !important;
            border: 1px solid rgba(184, 137, 26, 0.4) !important;
            color: #5b4310 !important;
            border-radius: 12px !important;
        }

        [data-testid="stAlert"] p {
            color: #5b4310 !important;
            font-weight: 600 !important;
        }

        .stButton button,
        .stDownloadButton button {
            background: linear-gradient(135deg, #f8fbff, #f0f4fa) !important;
            border: 1px solid rgba(184, 137, 26, 0.32) !important;
            color: #1f3044 !important;
            font-weight: 700 !important;
            border-radius: 10px !important;
        }

        .stButton button:hover,
        .stDownloadButton button:hover {
            border-color: rgba(184, 137, 26, 0.55) !important;
            background: linear-gradient(135deg, #ffffff, #f6f9ff) !important;
        }

        .premium-loader {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 0.8rem;
            padding: 2.2rem 1rem 1.8rem 1rem;
        }

        .premium-loader-dots {
            display: flex;
            gap: 0.45rem;
        }

        .premium-loader-dots span {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: linear-gradient(135deg, #c1982e, #0ea479);
            animation: pulseDot 1.1s infinite ease-in-out;
        }

        .premium-loader-dots span:nth-child(2) { animation-delay: 0.12s; }
        .premium-loader-dots span:nth-child(3) { animation-delay: 0.24s; }

        .premium-loader-text {
            font-size: 0.92rem;
            font-weight: 600;
            letter-spacing: 0.03em;
            color: rgba(39, 55, 74, 0.82);
        }

        .dashboard-strip {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 0.9rem;
            margin: 0.7rem 0 1.3rem 0;
        }

        .dash-kpi-card {
            border: 1px solid rgba(184, 137, 26, 0.24);
            border-radius: 14px;
            background: linear-gradient(140deg, rgba(255, 255, 255, 0.9), rgba(248, 252, 255, 0.92));
            padding: 0.95rem 1rem;
            box-shadow: 0 10px 26px rgba(31, 46, 64, 0.08);
        }

        .dash-kpi-label {
            color: rgba(36, 52, 71, 0.74);
            font-size: 0.72rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        .dash-kpi-value {
            color: #22374d;
            font-family: 'Syne', sans-serif;
            font-size: 1.35rem;
            font-weight: 700;
            line-height: 1.2;
        }

        .dash-kpi-positive {
            color: #0f815d;
        }

        .dash-kpi-negative {
            color: #b33f2d;
        }

        .dashboard-chart-shell {
            border: 1px solid rgba(184, 137, 26, 0.2);
            border-radius: 16px;
            padding: 0.9rem 0.9rem 0.3rem 0.9rem;
            background: linear-gradient(140deg, rgba(255, 255, 255, 0.88), rgba(245, 250, 255, 0.9));
            margin-bottom: 0.9rem;
        }

        .stFileUploader {
            font-size: 0 !important;
        }

        .stFileUploader label {
            font-size: 0.95rem !important;
        }

        .stFileUploader > div > button {
            border-radius: 16px !important;
            border: 2px dashed rgba(184, 137, 26, 0.32) !important;
            background: rgba(255, 255, 255, 0.7) !important;
            padding: 2rem !important;
            transition: all 400ms !important;
        }

        .stFileUploader > div > button:hover {
            border-color: rgba(184, 137, 26, 0.58) !important;
            background: rgba(184, 137, 26, 0.08) !important;
        }

        [data-testid="stFileUploadDropzone"] button,
        [data-testid="stFileUploadDropzone"] button *,
        .stFileUploader button,
        .stFileUploader button * {
            color: #ffffff !important;
            fill: #ffffff !important;
            stroke: #ffffff !important;
            opacity: 1 !important;
        }

        [data-testid="stFileUploadDropzone"] {
            border-radius: 16px;
            border: 2px dashed rgba(184, 137, 26, 0.32);
            background: rgba(255, 255, 255, 0.72);
            transition: all 400ms;
        }

        [data-testid="stFileUploadDropzone"]:hover {
            border-color: rgba(184, 137, 26, 0.58);
            background: rgba(184, 137, 26, 0.08);
        }

        @keyframes slideDownIn {
            from {
                opacity: 0;
                transform: translateY(-30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(40px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes popIn {
            from {
                opacity: 0;
                transform: scale(0.8);
            }
            to {
                opacity: 1;
                transform: scale(1);
            }
        }

        @keyframes slideInCard {
            from {
                opacity: 0;
                transform: translateX(-20px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }

        @keyframes slideInLeft {
            from {
                opacity: 0;
                transform: translateX(-30px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }

        @keyframes slideInToRight {
            from {
                transform: scaleX(0);
                transform-origin: left;
            }
            to {
                transform: scaleX(1);
                transform-origin: left;
            }
        }

        @keyframes pulseDot {
            0%, 80%, 100% {
                opacity: 0.45;
                transform: translateY(0) scale(0.9);
            }
            40% {
                opacity: 1;
                transform: translateY(-4px) scale(1.05);
            }
        }

        html[data-theme="dark"] .stApp,
        body[data-theme="dark"] .stApp {
            background: linear-gradient(135deg, #0f1419 0%, #1a2332 50%, #0d1821 100%);
            color: #f3f6fa;
        }

        html[data-theme="dark"] .luxury-hero,
        body[data-theme="dark"] .luxury-hero {
            background: linear-gradient(135deg, rgba(13, 17, 23, 0.8), rgba(26, 35, 50, 0.86)),
                        radial-gradient(circle at top right, rgba(184, 137, 26, 0.18), transparent 72%);
            border-color: rgba(184, 137, 26, 0.28);
            box-shadow: 0 30px 90px rgba(0, 0, 0, 0.38);
        }

        html[data-theme="dark"] .luxury-hero h1,
        body[data-theme="dark"] .luxury-hero h1 {
            background: linear-gradient(135deg, #f3f6fa 0%, #dfbf72 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        html[data-theme="dark"] .luxury-hero p,
        body[data-theme="dark"] .luxury-hero p {
            color: rgba(238, 244, 251, 0.88);
        }

        html[data-theme="dark"] .luxury-panel,
        body[data-theme="dark"] .luxury-panel {
            background: linear-gradient(135deg, rgba(18, 25, 36, 0.9), rgba(12, 18, 28, 0.9));
            border-color: rgba(184, 137, 26, 0.25);
            box-shadow: 0 18px 50px rgba(0, 0, 0, 0.35);
        }

        html[data-theme="dark"] .dash-kpi-card,
        body[data-theme="dark"] .dash-kpi-card {
            border-color: rgba(184, 137, 26, 0.3);
            background: linear-gradient(140deg, rgba(15, 22, 33, 0.88), rgba(24, 34, 49, 0.9));
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.26);
        }

        html[data-theme="dark"] .dash-kpi-label,
        body[data-theme="dark"] .dash-kpi-label {
            color: rgba(231, 237, 245, 0.72);
        }

        html[data-theme="dark"] .dash-kpi-value,
        body[data-theme="dark"] .dash-kpi-value {
            color: #eaf2fc;
        }

        html[data-theme="dark"] .dash-kpi-positive,
        body[data-theme="dark"] .dash-kpi-positive {
            color: #42c29a;
        }

        html[data-theme="dark"] .dash-kpi-negative,
        body[data-theme="dark"] .dash-kpi-negative {
            color: #f08b7d;
        }

        html[data-theme="dark"] .dashboard-chart-shell,
        body[data-theme="dark"] .dashboard-chart-shell {
            border-color: rgba(184, 137, 26, 0.25);
            background: linear-gradient(140deg, rgba(15, 22, 33, 0.86), rgba(24, 34, 49, 0.88));
        }

        html[data-theme="dark"] .luxury-panel:hover,
        body[data-theme="dark"] .luxury-panel:hover {
            border-color: rgba(184, 137, 26, 0.42);
            box-shadow: 0 24px 72px rgba(184, 137, 26, 0.18);
        }

        html[data-theme="dark"] .luxury-chip,
        body[data-theme="dark"] .luxury-chip {
            background: linear-gradient(135deg, rgba(184, 137, 26, 0.22), rgba(10, 157, 107, 0.15));
            color: #dfbf72;
            border-color: rgba(184, 137, 26, 0.4);
        }

        html[data-theme="dark"] .luxury-metric-card,
        body[data-theme="dark"] .luxury-metric-card {
            border-color: rgba(184, 137, 26, 0.24);
            background: linear-gradient(135deg, rgba(184, 137, 26, 0.08), rgba(10, 157, 107, 0.06));
        }

        html[data-theme="dark"] .luxury-metric-label,
        body[data-theme="dark"] .luxury-metric-label {
            color: rgba(235, 242, 250, 0.72);
        }

        html[data-theme="dark"] .luxury-metric-value,
        body[data-theme="dark"] .luxury-metric-value {
            color: #e6cb8f;
        }

        html[data-theme="dark"] .luxury-bar-section h3,
        body[data-theme="dark"] .luxury-bar-section h3 {
            color: #f0f5fb;
        }

        html[data-theme="dark"] .luxury-bar-label,
        body[data-theme="dark"] .luxury-bar-label {
            color: rgba(240, 245, 252, 0.88);
        }

        html[data-theme="dark"] .luxury-bar-value,
        body[data-theme="dark"] .luxury-bar-value {
            color: #e2c883;
        }

        html[data-theme="dark"] .luxury-bar-track,
        body[data-theme="dark"] .luxury-bar-track {
            background: rgba(255, 255, 255, 0.09);
            border-color: rgba(184, 137, 26, 0.2);
        }

        html[data-theme="dark"] .luxury-footnote,
        body[data-theme="dark"] .luxury-footnote {
            color: rgba(223, 233, 245, 0.66);
            border-top-color: rgba(184, 137, 26, 0.16);
        }

        html[data-theme="dark"] [data-testid="stFileUploadDropzone"],
        body[data-theme="dark"] [data-testid="stFileUploadDropzone"],
        html[data-theme="dark"] .stFileUploader > div > button,
        body[data-theme="dark"] .stFileUploader > div > button {
            background: rgba(15, 22, 33, 0.82) !important;
            border-color: rgba(184, 137, 26, 0.38) !important;
            color: #e7edf5 !important;
        }

        html[data-theme="dark"] [data-testid="stFileUploadDropzone"]:hover,
        body[data-theme="dark"] [data-testid="stFileUploadDropzone"]:hover,
        html[data-theme="dark"] .stFileUploader > div > button:hover,
        body[data-theme="dark"] .stFileUploader > div > button:hover {
            background: rgba(184, 137, 26, 0.14) !important;
            border-color: rgba(184, 137, 26, 0.62) !important;
        }

        @media (max-width: 768px) {
            .luxury-hero {
                padding: 2.5rem 1.8rem;
                border-radius: 20px;
            }
            .luxury-hero h1 {
                font-size: 2.2rem;
            }
            .luxury-hero p {
                font-size: 1rem;
            }
            .luxury-panel {
                padding: 1.6rem;
                border-radius: 18px;
            }
            .luxury-metric-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
)

if is_dark_theme:
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #0f1419 0%, #1a2332 50%, #0d1821 100%) !important;
            color: #f3f6fa !important;
        }

        .luxury-hero {
            background: linear-gradient(135deg, rgba(13, 17, 23, 0.8), rgba(26, 35, 50, 0.86)),
                        radial-gradient(circle at top right, rgba(184, 137, 26, 0.18), transparent 72%) !important;
            border-color: rgba(184, 137, 26, 0.28) !important;
            box-shadow: 0 30px 90px rgba(0, 0, 0, 0.38) !important;
        }

        .luxury-hero h1 {
            background: linear-gradient(135deg, #f3f6fa 0%, #dfbf72 100%) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            background-clip: text !important;
        }

        .luxury-hero p {
            color: rgba(238, 244, 251, 0.88) !important;
        }

        .luxury-panel {
            background: linear-gradient(135deg, rgba(18, 25, 36, 0.9), rgba(12, 18, 28, 0.9)) !important;
            border-color: rgba(184, 137, 26, 0.25) !important;
            box-shadow: 0 18px 50px rgba(0, 0, 0, 0.35) !important;
        }

        .luxury-chip {
            background: linear-gradient(135deg, rgba(184, 137, 26, 0.22), rgba(10, 157, 107, 0.15)) !important;
            color: #dfbf72 !important;
            border-color: rgba(184, 137, 26, 0.4) !important;
        }

        .luxury-metric-card {
            border-color: rgba(184, 137, 26, 0.24) !important;
            background: linear-gradient(135deg, rgba(184, 137, 26, 0.08), rgba(10, 157, 107, 0.06)) !important;
        }

        .luxury-metric-label {
            color: rgba(235, 242, 250, 0.72) !important;
        }

        .luxury-metric-value {
            color: #e6cb8f !important;
        }

        .luxury-bar-section h3 {
            color: #f0f5fb !important;
        }

        .luxury-bar-label {
            color: rgba(240, 245, 252, 0.88) !important;
        }

        .luxury-bar-value {
            color: #e2c883 !important;
        }

        .luxury-bar-track {
            background: rgba(255, 255, 255, 0.09) !important;
            border-color: rgba(184, 137, 26, 0.2) !important;
        }

        .luxury-footnote {
            color: rgba(223, 233, 245, 0.66) !important;
            border-top-color: rgba(184, 137, 26, 0.16) !important;
        }

        .biz-metric-card {
            border-color: rgba(184, 137, 26, 0.28) !important;
            background: linear-gradient(135deg, rgba(15, 22, 33, 0.88), rgba(24, 34, 49, 0.88)) !important;
        }

        .biz-metric-label {
            color: rgba(235, 242, 250, 0.72) !important;
        }

        .biz-metric-value {
            color: #eaf2fc !important;
        }

        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            color: rgba(223, 233, 245, 0.78) !important;
        }

        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] [data-testid="stMarkdownContainer"] p {
            color: #f4e3b1 !important;
        }

        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"],
        [data-testid="stMetricDelta"] {
            color: #eaf2fc !important;
        }

        .stSelectbox label,
        .stNumberInput label,
        .stSlider label,
        .stDownloadButton label,
        .stButton label {
            color: rgba(231, 237, 245, 0.92) !important;
        }

        .pl-input-title {
            color: #e8eef6 !important;
        }

        [data-testid="stNumberInput"] input {
            color: #f2f7ff !important;
        }

        [data-testid="stNumberInput"] div[data-baseweb="input"] {
            background: rgba(15, 22, 33, 0.82) !important;
            border: 1px solid rgba(184, 137, 26, 0.35) !important;
            border-radius: 12px !important;
        }

        [data-testid="stAlert"] {
            background: rgba(95, 74, 15, 0.28) !important;
            border: 1px solid rgba(223, 191, 114, 0.42) !important;
            color: #f3e3b1 !important;
        }

        [data-testid="stAlert"] p {
            color: #f3e3b1 !important;
        }

        .stButton button,
        .stDownloadButton button {
            background: linear-gradient(135deg, rgba(15, 22, 33, 0.92), rgba(21, 31, 46, 0.92)) !important;
            border: 1px solid rgba(184, 137, 26, 0.38) !important;
            color: #f2f7ff !important;
            font-weight: 700 !important;
            border-radius: 10px !important;
        }

        .stButton button:hover,
        .stDownloadButton button:hover {
            background: linear-gradient(135deg, rgba(184, 137, 26, 0.2), rgba(10, 157, 107, 0.16)) !important;
            border-color: rgba(184, 137, 26, 0.62) !important;
        }

        .premium-loader-text {
            color: rgba(231, 237, 245, 0.86) !important;
        }

        [data-testid="stFileUploadDropzone"],
        .stFileUploader > div > button {
            background: rgba(15, 22, 33, 0.82) !important;
            border-color: rgba(184, 137, 26, 0.38) !important;
            color: #e7edf5 !important;
        }

        [data-testid="stFileUploadDropzone"] button,
        [data-testid="stFileUploadDropzone"] button *,
        .stFileUploader button,
        .stFileUploader button * {
            color: #f2f7ff !important;
            fill: #f2f7ff !important;
            stroke: #f2f7ff !important;
            opacity: 1 !important;
        }

        [data-testid="stFileUploadDropzone"] button {
            background: linear-gradient(135deg, rgba(184, 137, 26, 0.22), rgba(10, 157, 107, 0.2)) !important;
            border: 1px solid rgba(184, 137, 26, 0.65) !important;
            border-radius: 10px !important;
        }

        [data-testid="stFileUploadDropzone"] small,
        [data-testid="stFileUploadDropzone"] p,
        [data-testid="stFileUploadDropzone"] span {
            color: rgba(231, 237, 245, 0.92) !important;
        }

        [data-testid="stFileUploadDropzone"]:hover,
        .stFileUploader > div > button:hover {
            background: rgba(184, 137, 26, 0.14) !important;
            border-color: rgba(184, 137, 26, 0.62) !important;
        }

        /* Dark Theme: Confidence Cards */
        .confidence-card {
            border-radius: 12px !important;
            padding: 16px 20px !important;
            margin-bottom: 16px !important;
            display: flex !important;
            gap: 16px !important;
            align-items: flex-start !important;
        }

        .confidence-card-success {
            background: linear-gradient(135deg, rgba(10, 157, 107, 0.16), rgba(10, 157, 107, 0.08)) !important;
            border: 1px solid rgba(10, 157, 107, 0.35) !important;
        }

        .confidence-card-warning {
            background: linear-gradient(135deg, rgba(184, 137, 26, 0.14), rgba(184, 137, 26, 0.08)) !important;
            border: 1px solid rgba(184, 137, 26, 0.38) !important;
        }

        .confidence-icon {
            font-size: 1.5rem !important;
            font-weight: 700 !important;
            min-width: 24px !important;
        }

        .confidence-content {
            flex: 1 !important;
        }

        .confidence-title {
            color: #f2f7ff !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
            margin-bottom: 4px !important;
        }

        .confidence-text {
            color: rgba(238, 244, 251, 0.88) !important;
            font-size: 0.9rem !important;
            line-height: 1.4 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Light theme CSS overrides for Profit/Loss and form controls
if not is_dark_theme:
    st.markdown(
        """
        <style>
        /* Light Theme: Profit/Loss Input Title */
        .pl-input-title {
            color: #1b2c3f !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
        }

        /* Light Theme: Number Input Fields */
        [data-testid="stNumberInput"] input {
            background-color: rgba(242, 247, 255, 0.6) !important;
            color: #1b2c3f !important;
            border: 1px solid rgba(184, 137, 26, 0.3) !important;
            border-radius: 8px !important;
            transition: all 0.2s ease !important;
        }

        [data-testid="stNumberInput"] input:hover {
            border-color: rgba(184, 137, 26, 0.4) !important;
        }

        [data-testid="stNumberInput"] input:focus {
            border-color: rgba(184, 137, 26, 0.6) !important;
            box-shadow: 0 0 0 3px rgba(184, 137, 26, 0.08) !important;
            outline: none !important;
        }

        /* Light Theme: Number Input +/- Buttons */
        [data-testid="stNumberInput"] button {
            background-color: rgba(242, 247, 255, 0.6) !important;
            color: #1b2c3f !important;
            border: 1px solid rgba(184, 137, 26, 0.3) !important;
            transition: all 0.2s ease !important;
            border-radius: 6px !important;
        }

        [data-testid="stNumberInput"] button:hover {
            background-color: rgba(220, 210, 185, 0.25) !important;
            border-color: rgba(184, 137, 26, 0.5) !important;
        }

        [data-testid="stNumberInput"] button:active {
            background-color: rgba(184, 137, 26, 0.2) !important;
            border-color: rgba(184, 137, 26, 0.6) !important;
        }

        /* Light Theme: Text Input Fields */
        [data-testid="stTextInput"] input {
            background-color: rgba(242, 247, 255, 0.6) !important;
            color: #1b2c3f !important;
            border: 1px solid rgba(184, 137, 26, 0.3) !important;
            border-radius: 8px !important;
            transition: all 0.2s ease !important;
        }

        [data-testid="stTextInput"] input:hover {
            border-color: rgba(184, 137, 26, 0.4) !important;
        }

        [data-testid="stTextInput"] input:focus {
            border-color: rgba(184, 137, 26, 0.6) !important;
            box-shadow: 0 0 0 3px rgba(184, 137, 26, 0.08) !important;
            outline: none !important;
        }

        /* Light Theme: General Input Labels */
        [data-testid="stNumberInput"] label,
        [data-testid="stTextInput"] label,
        .stNumberInput label,
        .stTextInput label,
        label {
            color: #1b2c3f !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
        }

        /* Light Theme: Help Text & Hints (matching border color) */
        [data-testid="stNumberInput"] small,
        [data-testid="stTextInput"] small,
        .stNumberInput small,
        .stTextInput small,
        .stHelp,
        [data-testid="stHelp"] {
            color: #5a718a !important;
            font-size: 0.85rem !important;
        }

        /* Light Theme: Tooltip Icons - VISIBLE */
        [data-testid="stNumberInput"] svg,
        [data-testid="stTextInput"] svg,
        [data-testid="stSelectbox"] svg {
            stroke: rgba(184, 137, 26, 0.5) !important;
            fill: none !important;
            color: rgba(184, 137, 26, 0.5) !important;
        }

        /* Tooltip/Info icon styling */
        button[data-testid*="tooltip"],
        [aria-label*="tooltip"],
        [aria-label*="info"] {
            color: rgba(184, 137, 26, 0.5) !important;
        }

        /* SVG icons in input containers */
        [data-testid="stNumberInput"] [role="img"],
        [data-testid="stTextInput"] [role="img"],
        [data-testid="stSelectbox"] [role="img"] {
            color: rgba(184, 137, 26, 0.5) !important;
            fill: rgba(184, 137, 26, 0.5) !important;
            stroke: rgba(184, 137, 26, 0.5) !important;
        }

        /* Question mark / help icon */
        svg[stroke="#7fa8d1"],
        svg[fill="#7fa8d1"],
        [class*="help"] svg,
        [class*="tooltip"] svg,
        [class*="info"] svg {
            stroke: rgba(184, 137, 26, 0.5) !important;
            fill: rgba(184, 137, 26, 0.5) !important;
            color: rgba(184, 137, 26, 0.5) !important;
        }

        /* Light Theme: File Uploader Labels */
        [data-testid="stFileUploader"] label,
        .stFileUploader label,
        [data-testid="stFileUploader"] p,
        .stFileUploader p {
            color: #1b2c3f !important;
            font-weight: 600 !important;
        }

        /* Light Theme: Keep upload button text/icons white */
        [data-testid="stFileUploader"] [data-testid="stFileUploadDropzone"] button,
        [data-testid="stFileUploader"] [data-testid="stFileUploadDropzone"] button *,
        [data-testid="stFileUploader"] [data-testid="stFileUploadDropzone"] button p,
        [data-testid="stFileUploader"] [data-testid="stFileUploadDropzone"] button div,
        [data-testid="stFileUploader"] [data-testid="stFileUploadDropzone"] button [data-testid="stMarkdownContainer"] p,
        [data-testid="stFileUploader"] [data-testid="stFileUploadDropzone"] button span,
        [data-testid="stFileUploader"] [data-testid="stFileUploadDropzone"] button svg,
        .stFileUploader [data-testid="stFileUploadDropzone"] button,
        .stFileUploader [data-testid="stFileUploadDropzone"] button *,
        .stFileUploader [data-testid="stFileUploadDropzone"] button p {
            color: #ffffff !important;
            fill: #ffffff !important;
            stroke: #ffffff !important;
            opacity: 1 !important;
        }

        /* Light Theme: General Text Labels */
        .stLabel label {
            color: #1b2c3f !important;
            font-weight: 600 !important;
        }

        /* Light Theme: Checkbox, Radio Labels */
        [data-testid="stCheckbox"] label,
        [data-testid="stRadio"] label {
            color: #1b2c3f !important;
            font-weight: 500 !important;
        }

        /* Light Theme: Tooltips & Popovers */
        .stTooltipExtraContent,
        [data-testid="stTooltip"],
        .tooltip-content {
            background-color: rgba(242, 247, 255, 0.95) !important;
            border: 1px solid rgba(184, 137, 26, 0.3) !important;
            border-radius: 6px !important;
            color: #1b2c3f !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06) !important;
            padding: 8px 12px !important;
        }

        /* Light Theme: Hover Tooltips */
        .stTooltip:hover .tooltip-content {
            border-color: rgba(184, 137, 26, 0.5) !important;
            background-color: rgba(255, 255, 255, 0.98) !important;
            box-shadow: 0 4px 12px rgba(184, 137, 26, 0.1) !important;
        }

        /* Light Theme: Alert/Warning Messages */
        [data-testid="stAlert"] {
            background: rgba(242, 239, 230, 0.6) !important;
            border: 1px solid rgba(224, 189, 89, 0.5) !important;
            color: rgba(90, 67, 15, 0.9) !important;
        }

        [data-testid="stAlert"] p,
        [data-testid="stAlert"] div,
        [data-testid="stAlert"] span {
            color: rgba(90, 67, 15, 0.9) !important;
        }

        /* Light Theme: Success Alert */
        [data-testid="stAlert"] svg {
            color: rgba(212, 165, 116, 0.8) !important;
        }

        /* Light Theme: Expander Headers (for Profit/Loss section) */
        [data-testid="stExpander"] button {
            color: rgba(231, 237, 245, 0.92) !important;
            border: 1px solid rgba(184, 137, 26, 0.2) !important;
            border-radius: 8px !important;
            transition: all 0.2s ease !important;
        }

        [data-testid="stExpander"] button:hover {
            background-color: rgba(184, 137, 26, 0.06) !important;
            border-color: rgba(184, 137, 26, 0.3) !important;
        }

        /* Light Theme: Expander Content Border */
        [data-testid="stExpander"] {
            border: 1px solid rgba(184, 137, 26, 0.2) !important;
            border-radius: 8px !important;
            padding: 12px !important;
        }

        /* Light Theme: Custom Card Styling */
        .profit-loss-card {
            background-color: rgba(242, 247, 255, 0.4) !important;
            border: 1px solid rgba(184, 137, 26, 0.15) !important;
            border-radius: 8px !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02) !important;
        }

        /* Light Theme: Metric Text */
        .metric-label {
            color: rgba(235, 242, 250, 0.72) !important;
            font-size: 0.85rem !important;
            font-weight: 500 !important;
        }

        .metric-value {
            color: rgba(231, 237, 245, 0.92) !important;
            font-weight: 600 !important;
        }

        /* Light Theme: Divider */
        hr {
            border-color: rgba(184, 137, 26, 0.15) !important;
            border-width: 1px !important;
        }

        /* Light Theme: Chart Text */
        [data-testid="stPlotlyChart"] text {
            fill: rgba(231, 237, 245, 0.92) !important;
        }

        /* Light Theme: Select & Dropdown */
        [data-testid="stSelectbox"] select,
        [role="combobox"] {
            background-color: rgba(242, 247, 255, 0.6) !important;
            color: #1b2c3f !important;
            border: 1px solid rgba(184, 137, 26, 0.3) !important;
            border-radius: 8px !important;
        }

        [data-testid="stSelectbox"] select:hover,
        [role="combobox"]:hover {
            border-color: rgba(184, 137, 26, 0.4) !important;
        }

        [data-testid="stSelectbox"] select:focus,
        [role="combobox"]:focus {
            border-color: rgba(184, 137, 26, 0.6) !important;
            box-shadow: 0 0 0 3px rgba(184, 137, 26, 0.08) !important;
        }

        /* Light Theme: Multi-select and BaseWeb Select Controls */
        [data-testid="stMultiSelect"] [data-baseweb="select"],
        [data-testid="stMultiSelect"] div[data-baseweb="select"] > div,
        [data-testid="stSelectbox"] [data-baseweb="select"],
        [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            background: rgba(242, 247, 255, 0.78) !important;
            border: 1px solid rgba(184, 137, 26, 0.36) !important;
            border-radius: 10px !important;
            color: #1b2c3f !important;
        }

        [data-testid="stMultiSelect"] [data-baseweb="tag"],
        [data-testid="stSelectbox"] [data-baseweb="tag"] {
            background: rgba(184, 137, 26, 0.18) !important;
            color: #5a420f !important;
            border: 1px solid rgba(184, 137, 26, 0.36) !important;
        }

        [data-testid="stMultiSelect"] [data-baseweb="select"] svg,
        [data-testid="stSelectbox"] [data-baseweb="select"] svg {
            color: rgba(96, 73, 19, 0.8) !important;
            fill: rgba(96, 73, 19, 0.8) !important;
            stroke: rgba(96, 73, 19, 0.8) !important;
        }

        /* Light Theme: Radio controls readability */
        [data-testid="stRadio"] div[role="radiogroup"] label,
        [data-testid="stRadio"] div[role="radiogroup"] label p,
        [data-testid="stRadio"] div[role="radiogroup"] span {
            color: #1b2c3f !important;
            font-weight: 600 !important;
        }

        [data-testid="stRadio"] div[role="radiogroup"] input + div {
            border-color: rgba(184, 137, 26, 0.46) !important;
        }

        [data-testid="stRadio"] div[role="radiogroup"] input:checked + div {
            border-color: #d46f2e !important;
            background-color: rgba(212, 111, 46, 0.12) !important;
        }

        /* Light Theme: Slider readability */
        [data-testid="stSlider"] [data-baseweb="slider"] > div > div {
            background: rgba(184, 137, 26, 0.24) !important;
        }

        [data-testid="stSlider"] [role="slider"] {
            background: #d46f2e !important;
            border: 2px solid #ffffff !important;
            box-shadow: 0 0 0 2px rgba(212, 111, 46, 0.28) !important;
        }

        [data-testid="stSlider"] [data-testid="stTickBarMin"],
        [data-testid="stSlider"] [data-testid="stTickBarMax"],
        [data-testid="stSlider"] [data-testid="stSliderTickBarMin"],
        [data-testid="stSlider"] [data-testid="stSliderTickBarMax"] {
            color: #1b2c3f !important;
        }

        /* Light Theme: Consistent Transition Effects */
        input, button, select, textarea {
            transition: border-color 0.2s ease, box-shadow 0.2s ease, background-color 0.2s ease !important;
        }

        /* Light Theme: Confidence Cards */
        .confidence-card {
            border-radius: 12px !important;
            padding: 16px 20px !important;
            margin-bottom: 16px !important;
            display: flex !important;
            gap: 16px !important;
            align-items: flex-start !important;
        }

        .confidence-card-success {
            background: rgba(198, 239, 206, 0.25) !important;
            border: 1.5px solid rgba(85, 186, 124, 0.4) !important;
        }

        .confidence-card-warning {
            background: rgba(255, 237, 185, 0.3) !important;
            border: 1.5px solid rgba(230, 180, 70, 0.5) !important;
        }

        .confidence-icon {
            font-size: 1.5rem !important;
            font-weight: 700 !important;
            min-width: 24px !important;
        }

        .confidence-content {
            flex: 1 !important;
        }

        .confidence-title {
            color: #1b2c3f !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
            margin-bottom: 4px !important;
        }

        .confidence-text {
            color: #3a5a7a !important;
            font-size: 0.9rem !important;
            line-height: 1.4 !important;
        }

        /* Light Theme: Premium Profit/Loss Section */
        .pl-input-title {
            color: #1b2c3f !important;
        }

        /* Light Theme: Number Input Fields */
        [data-testid="stNumberInput"] input {
            color: #1b2c3f !important;
        }

        /* Light Theme: Warning Alert Enhanced */
        [data-testid="stAlert"] {
            background: linear-gradient(135deg, #fffbf0 0%, #fff6d8 100%) !important;
            border: 2px solid #e0bd59 !important;
            border-radius: 8px !important;
            padding: 12px 16px !important;
            margin-top: 16px !important;
        }

        [data-testid="stAlert"] p {
            color: #6b4c0f !important;
            font-weight: 500 !important;
            margin: 0 !important;
        }

        /* Light Theme: Number Input Buttons Enhanced */
        [data-testid="stNumberInput"] button {
            border-radius: 6px !important;
            font-size: 0.9rem !important;
            padding: 8px 12px !important;
            font-weight: 600 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
        """
        <section class="luxury-hero">
            <div class="luxury-hero-content">
                <h1>Supermarket Vision Studio</h1>
                <p>
                    Upload one product image to classify category and audience age range with AI precision.
                    Predictions are confidence-aware, ensuring reliable insights.
                </p>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
)

engine = get_engine()
runtime = engine.runtime
class_names = runtime.class_names
confidence_threshold = runtime.confidence_threshold
hybrid_alpha = runtime.hybrid_alpha

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

try:
    engine.get_model()
except FileNotFoundError:
    st.error("No model file found (model.keras/model.h5). Run 'make train' first.")
    st.stop()

tab_analyzer, tab_dashboard = st.tabs(["AI Analyzer", "Interactive Dashboard"])

with tab_analyzer:
    left, right = st.columns([1.05, 1.2], gap="large")

    with left:
        has_uploaded = st.session_state.get("product_uploader") is not None
        if not has_uploaded:
            st.markdown("<div class='luxury-panel'><span class='luxury-chip'>Image Input</span></div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload product image",
            type=["jpg", "jpeg", "png", "webp"],
            key="product_uploader",
        )

        with st.container(border=True):
            st.markdown('<div class="pl-input-title">Profit/Loss Inputs</div>', unsafe_allow_html=True)
            unit_cost = st.number_input(
                "Unit Cost",
                value=0.0,
                step=1.0,
                help="Cost price per product unit",
            )
            sale_price = st.number_input(
                "Sale Price",
                value=0.0,
                step=1.0,
                help="Selling price per product unit",
            )
            quantity_sold = st.number_input(
                "Quantity Sold",
                value=0,
                step=1,
                help="Units sold for this predicted item",
            )

            if sale_price < unit_cost and (sale_price > 0 or unit_cost > 0):
                st.warning("Sale price is below unit cost, this transaction will produce a loss.")

            analyze_clicked = st.button(
                "Analyze Product",
                type="primary",
                use_container_width=True,
                disabled=uploaded_file is None,
            )

        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded image", use_container_width=True)

    should_analyze = uploaded_file is not None and analyze_clicked

    if should_analyze:
        img = Image.open(uploaded_file)

        with right:
            loading_slot = st.empty()
            loading_slot.markdown(
                """
                <div class="luxury-panel">
                  <div class="premium-loader">
                    <div class="premium-loader-dots"><span></span><span></span><span></span></div>
                    <div class="premium-loader-text">Analyzing image...</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        started_at = time.perf_counter()
        try:
            result = engine.predict_pil(img)
        except ValueError:
            loading_slot.empty()
            st.error("Model and class metadata mismatch. Please run 'make train' again.")
            st.stop()

        elapsed = time.perf_counter() - started_at
        if elapsed < MIN_ANALYZE_LOADER_SECONDS:
            time.sleep(MIN_ANALYZE_LOADER_SECONDS - elapsed)

        loading_slot.empty()

        centroid_counts = engine.centroid_counts

        probs = result.probs
        second_idx = result.second_idx
        confidence = result.confidence
        margin = result.margin
        sorted_idx = np.argsort(probs)[::-1]
        is_uncertain = result.uncertain
        business = compute_business_metrics(
            category=result.top_label,
            confidence=confidence,
            uncertain=is_uncertain,
            unit_cost=unit_cost,
            sale_price=sale_price,
            quantity=quantity_sold,
        )

        st.session_state.prediction_history.append(
            {
                "time": datetime.now().strftime("%H:%M:%S"),
                "category": result.top_label,
                "shown_category": result.category,
                "confidence": round(confidence, 4),
                "margin": round(margin, 4),
                "uncertain": is_uncertain,
                "unit_cost": round(business.unit_cost, 2),
                "sale_price": round(business.sale_price, 2),
                "quantity": business.quantity,
                "revenue": round(business.revenue, 2),
                "total_cost": round(business.total_cost, 2),
                "profit": round(business.profit, 2),
                "profit_margin_percent": round(business.profit_margin_percent, 2),
                "expected_margin_rate": round(business.expected_margin_rate, 4),
                "expected_profit": round(business.expected_profit, 2),
            }
        )
        st.session_state.prediction_history = st.session_state.prediction_history[-50:]

        with right:

            if is_uncertain:
                category_display = result.category
                age_display = result.age_group
                reliability = "Low"
                status_class = "luxury-status-uncertain"
            else:
                category_display = result.category
                age_display = result.age_group
                reliability = "High"
                status_class = "luxury-status-high"

            st.markdown(
                f"""
                <div class="luxury-metric-grid">
                  <div class="luxury-metric-card">
                    <div class="luxury-metric-label">Category</div>
                    <div class="luxury-metric-value">{category_display}</div>
                  </div>
                  <div class="luxury-metric-card">
                    <div class="luxury-metric-label">Age Group</div>
                    <div class="luxury-metric-value">{age_display}</div>
                  </div>
                  <div class="luxury-metric-card">
                                        <div class="luxury-metric-label">Calibrated Confidence</div>
                    <div class="luxury-metric-value">{confidence:.1%}</div>
                  </div>
                  <div class="luxury-metric-card">
                    <div class="luxury-metric-label">Reliability</div>
                    <div style="margin-top: 0.5rem;"><span class="{status_class}">{reliability}</span></div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

            if is_uncertain:
                st.markdown(
                    f"""
                    <div class="confidence-card confidence-card-warning">
                        <div class="confidence-icon">⚠️</div>
                        <div class="confidence-content">
                            <div class="confidence-title">Multiple Predictions Close</div>
                            <div class="confidence-text">Likely {result.top_label} ({confidence:.1%}) | Second likely {result.second_label} ({float(probs[second_idx]):.1%})</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div class="confidence-card confidence-card-success">
                        <div class="confidence-icon">✓</div>
                        <div class="confidence-content">
                            <div class="confidence-title">Strong Prediction Confidence</div>
                            <div class="confidence-text">Reliable classification for this image</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown(
                """
                <div class="luxury-bar-section">
                    <h3>Classification Probabilities</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )

            for idx in sorted_idx:
                class_name = class_names[int(idx)]
                prob = float(probs[int(idx)])
                st.markdown(
                    f"""
                    <div class="luxury-bar-row">
                      <div class="luxury-bar-header">
                        <span class="luxury-bar-label">{class_name}</span>
                        <span class="luxury-bar-value">{prob:.1%}</span>
                      </div>
                      <div class="luxury-bar-track">
                        <div class="luxury-bar-fill" style="width:{max(2.0, prob * 100):.1f}%;"></div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown(
                f"""
                <div class='luxury-footnote'>
                    Confidence Gate: {confidence_threshold:.0%} |
                    Top-2 Margin: {margin:.1%} |
                    Hybrid Blend: {100*hybrid_alpha:.0f}% Model + {100*(1-hybrid_alpha):.0f}% Similarity
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="biz-metric-grid">
                  <div class="biz-metric-card">
                    <div class="biz-metric-label">Revenue</div>
                    <div class="biz-metric-value">{format_money(business.revenue)}</div>
                  </div>
                  <div class="biz-metric-card">
                    <div class="biz-metric-label">Cost</div>
                    <div class="biz-metric-value">{format_money(business.total_cost)}</div>
                  </div>
                  <div class="biz-metric-card">
                    <div class="biz-metric-label">Actual Profit/Loss</div>
                    <div class="biz-metric-value">{format_money(business.profit)}</div>
                  </div>
                  <div class="biz-metric-card">
                    <div class="biz-metric-label">Expected Profit</div>
                    <div class="biz-metric-value">{format_money(business.expected_profit)}</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.caption(
                f"Actual margin: {business.profit_margin_percent:.1f}% | "
                f"Expected margin rule: {business.expected_margin_rate*100:.1f}%"
            )

            if centroid_counts:
                st.markdown(
                    f"""
                    <div class='luxury-footnote'>
                        Reference Dataset: {', '.join(f'{name}: {centroid_counts.get(name, 0)} images' for name in class_names)}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        with right:
            if uploaded_file is None:
                st.markdown(
                    """
                    <div class="luxury-panel">
                      <span class="luxury-chip">Analysis Output</span>
                      <p style="margin-top:1.2rem; color:rgba(39, 55, 74, 0.82); line-height:1.6; font-size:1rem;">
                        <strong>Ready to analyze.</strong> Upload a clear, front-facing product image with good lighting for best results.
                      </p>
                      <p style="margin-top:1rem; color:rgba(39, 55, 74, 0.68); line-height:1.5; font-size:0.9rem;">
                        Our AI model blends trained pattern recognition with real dataset similarity matching for highly accurate classifications.
                      </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div class="luxury-panel">
                      <span class="luxury-chip">Analysis Output</span>
                      <p style="margin-top:1.2rem; color:rgba(39, 55, 74, 0.82); line-height:1.6; font-size:1rem;">
                        <strong>Image uploaded.</strong> Set pricing values and click <strong>Analyze Product</strong> to run inference.
                      </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

with tab_dashboard:
    history = st.session_state.prediction_history

    if not history:
        st.info("No predictions yet. Run at least one image through AI Analyzer to populate live dashboard insights.")
    else:
        history_df = pd.DataFrame(history)
        total_preds = len(history_df)
        avg_conf = float(history_df["confidence"].mean())
        uncertain_count = int(history_df["uncertain"].sum())
        uncertain_rate = uncertain_count / max(total_preds, 1)

        category_counts = history_df["category"].value_counts().to_dict()
        top_category = max(category_counts, key=category_counts.get)

        total_revenue = float(history_df.get("revenue", pd.Series(dtype=float)).sum())
        total_cost = float(history_df.get("total_cost", pd.Series(dtype=float)).sum())
        total_profit = float(history_df.get("profit", pd.Series(dtype=float)).sum())
        total_expected_profit = float(history_df.get("expected_profit", pd.Series(dtype=float)).sum())
        loss_rows = int((history_df.get("profit", pd.Series(dtype=float)) < 0.0).sum())
        overall_margin = (total_profit / total_revenue * 100.0) if total_revenue > 0 else 0.0

        st.markdown("<div class='luxury-chip'>Interactive Dashboard</div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="dashboard-strip">
              <div class="dash-kpi-card">
                <div class="dash-kpi-label">Total Analyses</div>
                <div class="dash-kpi-value">{total_preds}</div>
              </div>
              <div class="dash-kpi-card">
                <div class="dash-kpi-label">Avg Confidence</div>
                <div class="dash-kpi-value">{avg_conf:.1%}</div>
              </div>
              <div class="dash-kpi-card">
                <div class="dash-kpi-label">Uncertain Rate</div>
                <div class="dash-kpi-value">{uncertain_rate:.1%}</div>
              </div>
              <div class="dash-kpi-card">
                <div class="dash-kpi-label">Top Predicted</div>
                <div class="dash-kpi-value">{top_category}</div>
              </div>
              <div class="dash-kpi-card">
                <div class="dash-kpi-label">Session Profit/Loss</div>
                <div class="dash-kpi-value {'dash-kpi-positive' if total_profit >= 0 else 'dash-kpi-negative'}">{format_money(total_profit)}</div>
              </div>
              <div class="dash-kpi-card">
                <div class="dash-kpi-label">Expected Profit</div>
                <div class="dash-kpi-value {'dash-kpi-positive' if total_expected_profit >= 0 else 'dash-kpi-negative'}">{format_money(total_expected_profit)}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        ctrl1, ctrl2, ctrl3 = st.columns([1.6, 1.1, 1.1], gap="large")
        category_options = sorted(history_df["category"].dropna().unique().tolist())
        with ctrl1:
            selected_categories = st.multiselect(
                "Categories",
                options=category_options,
                default=category_options,
                help="Show only selected predicted categories in charts and table.",
            )
        with ctrl2:
            reliability_filter = st.radio(
                "Reliability",
                ["All", "High", "Uncertain"],
                horizontal=True,
            )
        with ctrl3:
            show_rows = st.slider(
                "Rows to display",
                min_value=5,
                max_value=50,
                value=min(25, total_preds),
                step=5,
            )

        filtered_df = history_df.copy()
        if selected_categories:
            filtered_df = filtered_df[filtered_df["category"].isin(selected_categories)]
        else:
            filtered_df = filtered_df.iloc[0:0]

        if reliability_filter == "High":
            filtered_df = filtered_df[~filtered_df["uncertain"]]
        elif reliability_filter == "Uncertain":
            filtered_df = filtered_df[filtered_df["uncertain"]]

        filtered_df = filtered_df.tail(show_rows).copy()

        if filtered_df.empty:
            st.warning("No rows match current dashboard filters. Showing latest session rows instead.")
            filtered_df = history_df.tail(show_rows).copy()

        filtered_records = filtered_df.to_dict(orient="records")

        export_col1, export_col2, export_col3 = st.columns([1, 1, 1])
        with export_col1:
            st.download_button(
                "Export CSV",
                data=history_to_csv(filtered_records),
                file_name="prediction_history.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with export_col2:
            st.download_button(
                "Export JSON",
                data=json.dumps(filtered_records, indent=2),
                file_name="prediction_history.json",
                mime="application/json",
                use_container_width=True,
            )
        with export_col3:
            if st.button("Clear History", use_container_width=True):
                st.session_state.prediction_history = []
                st.rerun()

        st.markdown("### Prediction History")
        st.dataframe(filtered_df, use_container_width=True)

        chart_col1, chart_col2 = st.columns(2, gap="large")

        with chart_col1:
            st.markdown("<div class='dashboard-chart-shell'>", unsafe_allow_html=True)
            st.markdown("### Confidence Trend")
            trend_df = filtered_df.copy()
            trend_df["step"] = np.arange(1, len(trend_df) + 1)
            trend_df["confidence_percent"] = trend_df["confidence"] * 100
            if len(trend_df) < 2:
                st.info("Need at least 2 predictions to draw a trend. Add more analyses.")
                st.metric("Latest Confidence", f"{float(trend_df['confidence_percent'].iloc[-1]):.1f}%")
            else:
                trend_chart = (
                    alt.Chart(trend_df)
                    .mark_line(point=True, strokeWidth=3, color="#b8891a")
                    .encode(
                        x=alt.X("step:Q", title="Prediction Order"),
                        y=alt.Y("confidence_percent:Q", title="Confidence (%)", scale=alt.Scale(domain=[0, 100])),
                        tooltip=[
                            alt.Tooltip("step:Q", title="Order"),
                            alt.Tooltip("confidence_percent:Q", title="Confidence (%)", format=".1f"),
                            alt.Tooltip("category:N", title="Category"),
                        ],
                    )
                    .properties(height=250)
                )
                st.altair_chart(trend_chart, use_container_width=True)
            st.caption("Confidence shown as percentage for each recent prediction.")
            st.markdown("</div>", unsafe_allow_html=True)

        with chart_col2:
            st.markdown("<div class='dashboard-chart-shell'>", unsafe_allow_html=True)
            st.markdown("### Category Distribution")
            dist_df = (
                filtered_df["category"]
                .value_counts()
                .rename_axis("category")
                .reset_index(name="count")
                .sort_values("count", ascending=False)
            )
            dist_df["percent"] = (dist_df["count"] / dist_df["count"].sum()) * 100
            cat_chart = (
                alt.Chart(dist_df)
                .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
                .encode(
                    x=alt.X("count:Q", title="Count"),
                    y=alt.Y("category:N", title=None, sort="-x"),
                    color=alt.Color("category:N", legend=None),
                    tooltip=[
                        alt.Tooltip("category:N", title="Category"),
                        alt.Tooltip("count:Q", title="Count"),
                        alt.Tooltip("percent:Q", title="Percent", format=".1f"),
                    ],
                )
                .properties(height=250)
            )
            cat_text = cat_chart.mark_text(align="left", baseline="middle", dx=5).encode(text=alt.Text("percent:Q", format=".1f"))
            st.altair_chart(cat_chart + cat_text, use_container_width=True)
            st.caption("Category share for filtered session rows.")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='dashboard-chart-shell'>", unsafe_allow_html=True)
        st.markdown("### Reliability Mix")
        reliability_df = pd.DataFrame(
            {
                "status": ["High", "Uncertain"],
                "count": [int((~filtered_df["uncertain"]).sum()), int(filtered_df["uncertain"].sum())],
            }
        )
        reliability_df["percent"] = (reliability_df["count"] / max(len(filtered_df), 1)) * 100
        reliability_chart = (
            alt.Chart(reliability_df)
            .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
            .encode(
                x=alt.X("count:Q", title="Count"),
                y=alt.Y("status:N", title=None, sort=["High", "Uncertain"]),
                color=alt.Color(
                    "status:N",
                    title="Status",
                    scale=alt.Scale(domain=["High", "Uncertain"], range=["#1ea672", "#d48c2b"]),
                ),
                tooltip=[
                    alt.Tooltip("status:N", title="Status"),
                    alt.Tooltip("count:Q", title="Count"),
                    alt.Tooltip("percent:Q", title="Percent", format=".1f"),
                ],
            )
            .properties(height=180)
        )
        reliability_text = reliability_chart.mark_text(align="left", baseline="middle", dx=5).encode(text=alt.Text("percent:Q", format=".1f"))
        st.altair_chart(reliability_chart + reliability_text, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if "profit" in filtered_df.columns:
            st.markdown("### Profit/Loss Analytics")
            pl_col1, pl_col2 = st.columns(2, gap="large")

            with pl_col1:
                st.markdown("<div class='dashboard-chart-shell'>", unsafe_allow_html=True)
                trend_pl_df = filtered_df.copy()
                trend_pl_df["step"] = np.arange(1, len(trend_pl_df) + 1)
                trend_pl_df["cumulative_profit"] = trend_pl_df["profit"].cumsum()

                hover = alt.selection_point(fields=["step"], nearest=True, on="mouseover", empty=True)
                zero_rule = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#8793a0", strokeDash=[5, 4]).encode(y="y:Q")

                pl_area = (
                    alt.Chart(trend_pl_df)
                    .mark_area(opacity=0.18)
                    .encode(
                        x=alt.X("step:Q", title="Prediction Order"),
                        y=alt.Y("cumulative_profit:Q", title="Cumulative Profit/Loss"),
                        color=alt.condition("datum.cumulative_profit >= 0", alt.value("#1ea672"), alt.value("#d16b5a")),
                    )
                )
                txn_bars = (
                    alt.Chart(trend_pl_df)
                    .mark_bar(size=10, opacity=0.55)
                    .encode(
                        x=alt.X("step:Q", title="Prediction Order"),
                        y=alt.Y("profit:Q", title="Transaction P/L"),
                        color=alt.condition("datum.profit >= 0", alt.value("#1ea672"), alt.value("#d16b5a")),
                        tooltip=[
                            alt.Tooltip("step:Q", title="Order"),
                            alt.Tooltip("profit:Q", title="Txn P/L", format=".2f"),
                            alt.Tooltip("category:N", title="Category"),
                        ],
                    )
                )
                cum_line = (
                    alt.Chart(trend_pl_df)
                    .mark_line(strokeWidth=3, color="#b8891a")
                    .encode(
                        x=alt.X("step:Q", title="Prediction Order"),
                        y=alt.Y("cumulative_profit:Q", title="Cumulative Profit/Loss"),
                        tooltip=[
                            alt.Tooltip("step:Q", title="Order"),
                            alt.Tooltip("cumulative_profit:Q", title="Cumulative P/L", format=".2f"),
                            alt.Tooltip("profit:Q", title="Txn P/L", format=".2f"),
                            alt.Tooltip("category:N", title="Category"),
                        ],
                    )
                )
                cum_points = (
                    alt.Chart(trend_pl_df)
                    .mark_circle(size=90, color="#1d2f44")
                    .encode(
                        x=alt.X("step:Q", title="Prediction Order"),
                        y=alt.Y("cumulative_profit:Q", title="Cumulative Profit/Loss"),
                        opacity=alt.condition(hover, alt.value(1), alt.value(0.25)),
                    )
                    .add_params(hover)
                )

                pl_trend_chart = (zero_rule + pl_area + txn_bars + cum_line + cum_points).properties(height=300)
                st.altair_chart(pl_trend_chart, use_container_width=True)
                st.caption("Green bars are profitable transactions, red bars are loss transactions; gold line tracks cumulative outcome.")
                st.markdown("</div>", unsafe_allow_html=True)

            with pl_col2:
                st.markdown("<div class='dashboard-chart-shell'>", unsafe_allow_html=True)
                by_cat_df = (
                    filtered_df.groupby("category", as_index=False)
                    .agg(total_profit=("profit", "sum"))
                    .sort_values("total_profit", ascending=False)
                )
                profit_zero_rule = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#8793a0", strokeDash=[5, 4]).encode(x="x:Q")
                by_cat_chart = (
                    alt.Chart(by_cat_df)
                    .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
                    .encode(
                        x=alt.X("total_profit:Q", title="Total Profit/Loss"),
                        y=alt.Y("category:N", title=None, sort="-x"),
                        color=alt.condition("datum.total_profit >= 0", alt.value("#1ea672"), alt.value("#d16b5a")),
                        tooltip=[
                            alt.Tooltip("category:N", title="Category"),
                            alt.Tooltip("total_profit:Q", title="Total P/L", format=".2f"),
                        ],
                    )
                    .properties(height=300)
                )
                st.altair_chart(profit_zero_rule + by_cat_chart, use_container_width=True)
                st.caption("Category contribution view with diverging axis around break-even (0).")
                st.markdown("</div>", unsafe_allow_html=True)

            if "expected_profit" in filtered_df.columns:
                st.markdown("<div class='dashboard-chart-shell'>", unsafe_allow_html=True)
                compare_df = filtered_df.copy()
                compare_df["step"] = np.arange(1, len(compare_df) + 1)
                compare_long_df = compare_df.melt(
                    id_vars=["step"],
                    value_vars=["profit", "expected_profit"],
                    var_name="metric",
                    value_name="value",
                )
                compare_chart = (
                    alt.Chart(compare_long_df)
                    .mark_line(point=True, strokeWidth=2.5)
                    .encode(
                        x=alt.X("step:Q", title="Prediction Order"),
                        y=alt.Y("value:Q", title="Profit/Loss"),
                        color=alt.Color(
                            "metric:N",
                            title="Metric",
                            scale=alt.Scale(
                                domain=["profit", "expected_profit"],
                                range=["#1f7a8c", "#b8891a"],
                            ),
                        ),
                        tooltip=[
                            alt.Tooltip("step:Q", title="Order"),
                            alt.Tooltip("metric:N", title="Metric"),
                            alt.Tooltip("value:Q", title="Value", format=".2f"),
                        ],
                    )
                    .properties(height=240)
                )
                st.altair_chart(compare_chart, use_container_width=True)
                st.caption("Actual profit/loss versus expected profit (rule-based forecast).")
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### Behavioral Segmentation (Hierarchical Clustering)")
        st.caption(
            "Applies Agglomerative Hierarchical Clustering on prediction-session signals "
            "(confidence, top-2 margin, uncertainty) to identify operational behavior segments."
        )

        if len(filtered_df) < 3:
            st.info("Need at least 3 prediction rows for clustering view.")
        else:
            cluster_df = filtered_df.copy()
            cluster_df["uncertain_num"] = cluster_df["uncertain"].astype(int)

            features = cluster_df[["confidence", "margin", "uncertain_num"]].astype(float)
            scaler = MinMaxScaler()
            X = scaler.fit_transform(features)

            n_clusters = min(3, len(cluster_df))
            clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
            cluster_df["cluster"] = clusterer.fit_predict(X)
            cluster_df["cluster"] = cluster_df["cluster"].astype(str)

            cluster_chart = (
                alt.Chart(cluster_df)
                .mark_circle(size=120, opacity=0.85)
                .encode(
                    x=alt.X("confidence:Q", title="Confidence"),
                    y=alt.Y("margin:Q", title="Top-2 Margin"),
                    color=alt.Color("cluster:N", title="Cluster"),
                    tooltip=[
                        alt.Tooltip("time:N", title="Time"),
                        alt.Tooltip("category:N", title="Category"),
                        alt.Tooltip("confidence:Q", format=".3f", title="Confidence"),
                        alt.Tooltip("margin:Q", format=".3f", title="Margin"),
                        alt.Tooltip("uncertain:N", title="Uncertain"),
                        alt.Tooltip("cluster:N", title="Cluster"),
                    ],
                )
                .properties(height=280)
            )
            st.altair_chart(cluster_chart, use_container_width=True)

            cluster_summary = (
                cluster_df.groupby("cluster", as_index=False)
                .agg(
                    count=("cluster", "count"),
                    avg_confidence=("confidence", "mean"),
                    avg_margin=("margin", "mean"),
                    uncertain_rate=("uncertain_num", "mean"),
                )
                .sort_values("cluster")
            )
            cluster_summary["avg_confidence"] = (cluster_summary["avg_confidence"] * 100).round(1)
            cluster_summary["avg_margin"] = (cluster_summary["avg_margin"] * 100).round(1)
            cluster_summary["uncertain_rate"] = (cluster_summary["uncertain_rate"] * 100).round(1)

            st.dataframe(
                cluster_summary.rename(
                    columns={
                        "cluster": "Cluster",
                        "count": "Rows",
                        "avg_confidence": "Avg Confidence (%)",
                        "avg_margin": "Avg Margin (%)",
                        "uncertain_rate": "Uncertain Rate (%)",
                    }
                ),
                use_container_width=True,
            )
