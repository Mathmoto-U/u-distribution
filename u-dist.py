"""
サイコロ(一様分布)の標本平均が正規分布に収束するシミュレーション (Streamlit版)
====================================================================
実行方法:
    pip install streamlit numpy plotly
    streamlit run dice_app.py
"""

import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ============================================================================
# 1. ページ設定 & カスタムCSS（横画面固定・上部見切れ防止・スクロール対応済）
# ============================================================================
st.set_page_config(
    page_title="標本平均の正規分布収束シミュレーター（サイコロ・一様分布）",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
    /* 1. Streamlit標準の上部ヘッダーとサイドバーを非表示にする（上部の見切れ防止） */
    header[data-testid="stHeader"],
    [data-testid="stDecoration"],
    [data-testid="stSidebar"],
    [data-testid="collapsedControl"] {
        display: none !important;
    }

    /* 2. ページ全体の縦・横スクロールを確実に有効化する */
    html, body, .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    section.main {
        overflow-y: auto !important;
        overflow-x: auto !important;
        height: auto !important;
        min-height: 100vh !important;
    }

    /* 3. 上下の余白調整（上が見切れないよう padding-top を確保） */
    .block-container {
        padding-top: 2.2rem !important;
        padding-bottom: 3rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 1400px !important;
    }

    /* 4. PCの横画面（左：標本、右：ヒストグラム）を強制維持 */
    [data-testid="stHorizontalBlock"] {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        align-items: flex-start !important;
    }
    [data-testid="column"] {
        min-width: 0 !important;
    }

    /* ヘッダーバナー */
    .sim-header {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 14px 20px;
        margin-bottom: 14px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }
    .sim-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0;
        line-height: 1.3;
    }
    .sim-subtitle {
        font-size: 0.78rem;
        color: #64748b;
        margin: 4px 0 0 0;
    }

    /* 左カラム：最新結果バナー（「標本平均」を中央揃え） */
    .result-banner {
        background: linear-gradient(90deg, #ecfdf5 0%, #f0fdfa 55%, #f8fafc 100%);
        border: 1px solid #a7f3d0;
        border-radius: 14px;
        padding: 12px 16px;
        margin-bottom: 14px;
        text-align: center;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .result-label {
        font-size: 0.85rem;
        font-weight: 700;
        color: #475569;
        margin-bottom: 2px;
    }
    .result-val-emerald {
        font-size: 1.95rem;
        font-weight: 800;
        color: #059669;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        line-height: 1.1;
    }

    /* 10x10 サイコロ・グリッド */
    .sampling-grid-wrapper {
        background: #f1f5f9;
        padding: 10px;
        border-radius: 14px;
        border: 1px solid #cbd5e1;
        max-width: 390px;
        margin: 0 auto 6px auto;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.03);
    }
    .sampling-grid {
        display: grid;
        grid-template-columns: repeat(10, 1fr);
        gap: 5px;
        aspect-ratio: 1 / 1;
    }
    .dice-cell {
        background: #ffffff;
        color: #0f172a;
        font-weight: 800;
        font-size: 0.95rem;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        border: 1px solid #cbd5e1;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
        user-select: none;
        aspect-ratio: 1 / 1;
    }
    /* 1の目だけサイコロらしく赤アクセント、6の目はエメラルドアクセント */
    .dice-1 {
        color: #e11d48;
        background: #fff1f2;
        border-color: #fecdd3;
    }
    .dice-6 {
        color: #059669;
        background: #ecfdf5;
        border-color: #a7f3d0;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================================
# 2. 定数・統計パラメータ・セッション状態の初期化
# ============================================================================
N_CELLS = 100  # 10x10 マス = サイコロ100個
MIN_SUM = 100  # 100個すべて「1」のときの合計 (標本平均 1.00)
MAX_SUM = 600  # 100個すべて「6」のときの合計 (標本平均 6.00)

# サイコロ1個の一様分布: μ = 3.5, σ^2 = (6^2 - 1)/12 = 35/12 ≒ 2.9167
# 100個の標本平均 X_bar: μ = 3.5, σ^2 = 35/1200 ≒ 0.029167, σ ≒ 0.17078
THEO_MEAN = 3.5
THEO_VAR = 35.0 / 1200.0
THEO_STD = math.sqrt(THEO_VAR)

# 連続実行のデフォルト速度（テンポよく進むよう設定）
SLEEP_INTERVAL = 0.06
STEP_BATCH_SIZE = 2


def init_session_state() -> None:
    """Streamlitのセッション状態を初期化する"""
    if "bin_step" not in st.session_state:
        # 合計値 S (100〜600) の束ね幅:
        # 10 -> 標本平均の階級幅 0.10 (±2σを約7等分・推奨)
        # 5  -> 標本平均の階級幅 0.05
        # 2  -> 標本平均の階級幅 0.02
        st.session_state.bin_step = 10
    if "exact_sum_counts" not in st.session_state:
        # 100個のサイコロの合計値 S (0〜600) の出現回数を厳密に記録
        st.session_state.exact_sum_counts = np.zeros(MAX_SUM + 1, dtype=int)
    if "total_trials" not in st.session_state:
        st.session_state.total_trials = 0
    if "latest_sum" not in st.session_state:
        st.session_state.latest_sum = None
    if "current_grid" not in st.session_state:
        st.session_state.current_grid = np.random.randint(1, 7, size=N_CELLS)
    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    if "show_normal_curve" not in st.session_state:
        st.session_state.show_normal_curve = True


init_session_state()


# ============================================================================
# 3. サンプリング & 階級（ビン）構築ロジック
# ============================================================================
def reset_simulation() -> None:
    """累積データをリセットして初期状態に戻す"""
    st.session_state.is_running = False
    st.session_state.exact_sum_counts = np.zeros(MAX_SUM + 1, dtype=int)
    st.session_state.total_trials = 0
    st.session_state.latest_sum = None
    st.session_state.current_grid = np.random.randint(1, 7, size=N_CELLS)


def run_trials(num_samples: int = 1) -> None:
    """
    指定回数のサンプリング（サイコロ100個振り）を実行し、
    10×10グリッドおよび標本平均の度数分布を更新する。
    """
    if num_samples <= 0:
        return

    # 最後の1回分は10x10マスの個別サイコロ目を生成して左カラムに表示
    last_grid = np.random.randint(1, 7, size=N_CELLS)
    last_sum = int(np.sum(last_grid))

    if num_samples == 1:
        st.session_state.exact_sum_counts[last_sum] += 1
        st.session_state.total_trials += 1
    else:
        # (num_samples - 1) 回分は2次元乱数で一括高速計算
        batch_rolls = np.random.randint(1, 7, size=(num_samples - 1, N_CELLS))
        batch_sums = np.sum(batch_rolls, axis=1)
        all_sums = np.append(batch_sums, last_sum)

        bincount = np.bincount(all_sums, minlength=MAX_SUM + 1)
        st.session_state.exact_sum_counts += bincount
        st.session_state.total_trials += num_samples

    st.session_state.current_grid = last_grid
    st.session_state.latest_sum = last_sum


def build_bins(
    bin_step: int,
    exact_sum_counts: np.ndarray,
    latest_sum: Optional[int],
) -> List[Dict]:
    """
    母平均 μ = 3.50 (合計 S = 350) が中央階級の中心付近になるよう対称に階級を構築する。
    bin_step: 合計値 S の束ね幅
      - 10 (階級幅 0.10): S=345〜354 (平均 3.45〜3.54) が中央階級
      -  5 (階級幅 0.05): S=348〜352 (平均 3.48〜3.52) が中央階級
      -  2 (階級幅 0.02): S=349〜350 (平均 3.49〜3.50) が中央階級
    """
    bins = []
    mean_sum = 350  # μ = 3.50 に対応する合計値

    if bin_step == 10:
        center_start = mean_sum - 5  # 345〜354
    elif bin_step == 5:
        center_start = mean_sum - 2  # 348〜352
    else:
        center_start = mean_sum - 1  # 349〜350

    offset = (center_start - MIN_SUM) % bin_step
    intervals: List[Tuple[int, int]] = []

    if offset > 0:
        intervals.append((MIN_SUM, MIN_SUM + offset - 1))

    for start_s in range(MIN_SUM + offset, MAX_SUM + 1, bin_step):
        end_s = min(MAX_SUM, start_s + bin_step - 1)
        intervals.append((start_s, end_s))

    for min_s, max_s in intervals:
        actual_step = max_s - min_s + 1
        actual_w = actual_step / 100.0  # 標本平均軸での階級幅
        c = int(np.sum(exact_sum_counts[min_s : max_s + 1]))

        min_mean = min_s / 100.0
        max_mean = max_s / 100.0
        center_mean = (min_mean + max_mean) / 2.0

        lbl = f"{min_mean:.2f}" if min_s == max_s else f"{min_mean:.2f}〜{max_mean:.2f}"
        bins.append({
            "min_s": min_s,
            "max_s": max_s,
            "actual_width": actual_w,
            "center": center_mean,
            "cont_start": (min_s - 0.5) / 100.0,
            "cont_end": (max_s + 0.5) / 100.0,
            "count": c,
            "label": lbl,
            "range_label": lbl,
            "contains_latest": (latest_sum is not None and min_s <= latest_sum <= max_s),
        })

    return bins


# ============================================================================
# 4. HTML描画ヘルパー（10×10 サイコログリッド ＆ 統計カード）
# ============================================================================
def render_10x10_dice_grid_html(grid_vals: np.ndarray) -> str:
    """10×10のサイコロの目（1〜6）グリッドをHTMLとして生成する"""
    cells_html = []
    for val in grid_vals:
        v = int(val)
        extra_cls = " dice-1" if v == 1 else (" dice-6" if v == 6 else "")
        cells_html.append(f'<div class="dice-cell{extra_cls}">{v}</div>')
    return f'<div class="sampling-grid-wrapper"><div class="sampling-grid">{"".join(cells_html)}</div></div>'


def render_left_result_banner_html(latest_sum: Optional[int]) -> str:
    """左カラム上部の「標本平均」バナーHTML（中央揃え）"""
    if latest_sum is None:
        mean_str = "-.--"
    else:
        mean_str = f"{latest_sum / 100.0:.2f}"

    return f"""
    <div class="result-banner">
        <div class="result-label">標本平均</div>
        <div class="result-val-emerald">{mean_str}</div>
    </div>
    """


# ============================================================================
# 5. Plotlyによる相対度数ヒストグラム＋正規分布曲線の描画
# ============================================================================
def create_histogram_figure(bins: List[Dict]) -> go.Figure:
    """標本平均の相対度数ヒストグラムと正規分布 N(3.5, 35/1200) を描画する"""
    N = st.session_state.total_trials
    w = st.session_state.bin_step / 100.0  # 標本平均の階級幅 (0.10, 0.05, 0.02)

    # X軸の表示範囲（サイコロの目 1.0 〜 6.0 全体を表示）
    x_min, x_max = 1.0, 6.0

    visible_bins = [b for b in bins if b["cont_end"] >= x_min and b["cont_start"] <= x_max]

    # 各階級の相対度数 (度数 / N)
    rel_freqs = [(b["count"] / N) if N > 0 else 0.0 for b in visible_bins]
    max_bar_rel = max(rel_freqs, default=0.0)

    # 正規分布の理論上の相対度数ピーク (階級幅 w × 確率密度ピーク)
    theo_peak = w / (THEO_STD * math.sqrt(2 * math.pi))
    raw_max_y = max(max_bar_rel, theo_peak)
    y_max = raw_max_y * 1.20

    fig = go.Figure()

    # 1. ヒストグラムの棒グラフ（相対度数）
    centers = [b["center"] for b in visible_bins]
    counts = [b["count"] for b in visible_bins]
    widths = [b["actual_width"] * 0.88 for b in visible_bins]
    range_labels = [b["range_label"] for b in visible_bins]

    bar_colors = [
        "#0d9488" if b["contains_latest"] else "rgba(16, 185, 129, 0.82)"
        for b in visible_bins
    ]
    border_colors = [
        "#065f46" if b["contains_latest"] else "#059669"
        for b in visible_bins
    ]
    border_widths = [2.2 if b["contains_latest"] else 1.0 for b in visible_bins]

    expected_rel_freqs = []
    for b in visible_bins:
        c_val = b["center"]
        z = (c_val - THEO_MEAN) / THEO_STD
        pdf = (1.0 / (THEO_STD * math.sqrt(2 * math.pi))) * math.exp(-0.5 * z * z)
        exp_rel = b["actual_width"] * pdf
        expected_rel_freqs.append(exp_rel)

    custom_data = np.column_stack([range_labels, counts, expected_rel_freqs])

    fig.add_trace(
        go.Bar(
            x=centers,
            y=rel_freqs,
            width=widths,
            name="相対度数",
            marker=dict(
                color=bar_colors,
                line=dict(color=border_colors, width=border_widths),
            ),
            customdata=custom_data,
            hovertemplate=(
                "<b>標本平均: %{customdata[0]}</b><br>"
                "相対度数: <b>%{y:.3f}</b> (%{customdata[1]:,}回)<br>"
                "理論相対度数: 約 %{customdata[2]:.3f}"
                "<extra></extra>"
            ),
        )
    )

    # 2. 正規分布の理論曲線オーバーレイ（相対度数スケール）
    if st.session_state.show_normal_curve:
        x_curve = np.linspace(x_min, x_max, 400)
        z_curve = (x_curve - THEO_MEAN) / THEO_STD
        pdf_curve = (1.0 / (THEO_STD * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * z_curve**2)

        y_curve = w * pdf_curve
        line_dash = "solid" if N > 0 else "dash"

        fig.add_trace(
            go.Scatter(
                x=x_curve,
                y=y_curve,
                mode="lines",
                name="正規分布曲線",
                line=dict(color="#f59e0b", width=3.2, dash=line_dash),
                hoverinfo="skip",
            )
        )

    xaxis_config = dict(
        range=[1.0, 6.0],
        dtick=0.5,
        gridcolor="#f1f5f9",
        zeroline=False,
        linecolor="#cbd5e1",
        tickfont=dict(size=11, color="#475569"),
    )

    fig.update_layout(
        height=440,
        margin=dict(l=52, r=20, t=20, b=42),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        xaxis=xaxis_config,
        yaxis=dict(
            range=[0, y_max],
            gridcolor="#f1f5f9",
            zeroline=True,
            zerolinecolor="#cbd5e1",
            tickformat=".2f",
            tickfont=dict(size=11, color="#64748b"),
        ),
        showlegend=False,
        bargap=0.08,
        hovermode="closest",
    )

    return fig


# ============================================================================
# 6. ヘッダー & トップコントロールバー
# ============================================================================
st.markdown(
    """
    <div class="sim-header">
        <div class="sim-title">
            標本平均が正規分布に収束するシミュレーション（一様分布）
        </div>
        <p class="sim-subtitle">
            サイコロを100個振り、出た目の標本平均を計算します。これを何度も繰り返し、ヒストグラムに記録していきます。
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# 操作ボタンバー
ctrl_cols = st.columns([1.2, 1.6, 1.0, 1.0, 1.0, 1.0])

with ctrl_cols[0]:
    if st.button("1回実行", use_container_width=True):
        st.session_state.is_running = False
        run_trials(1)

with ctrl_cols[1]:
    run_btn_label = "一時停止" if st.session_state.is_running else "連続実行"
    run_btn_type = "secondary" if st.session_state.is_running else "primary"
    if st.button(run_btn_label, type=run_btn_type, use_container_width=True):
        st.session_state.is_running = not st.session_state.is_running
        if st.session_state.is_running:
            run_trials(STEP_BATCH_SIZE)

with ctrl_cols[2]:
    if st.button("+10回", use_container_width=True):
        st.session_state.is_running = False
        run_trials(10)

with ctrl_cols[3]:
    if st.button("+100回", use_container_width=True):
        st.session_state.is_running = False
        run_trials(100)

with ctrl_cols[4]:
    if st.button("+1000回", use_container_width=True):
        st.session_state.is_running = False
        run_trials(1000)

with ctrl_cols[5]:
    if st.button("リセット", use_container_width=True):
        reset_simulation()
        st.rerun()


# ============================================================================
# 7. メイン2カラムレイアウト（左: 10×10 サイコロ / 右: 累計回数 & ヒストグラム）
# ============================================================================
all_bins = build_bins(
    st.session_state.bin_step,
    st.session_state.exact_sum_counts,
    st.session_state.latest_sum,
)

col_left, col_right = st.columns([4.2, 6.8], gap="large")

# ---------------- 左カラム: 10×10 サイコロ・グリッド ----------------
with col_left:
    with st.container(border=True):
        # 今回の標本平均バナー（中央揃え）
        st.markdown(
            render_left_result_banner_html(st.session_state.latest_sum),
            unsafe_allow_html=True,
        )

        # 10x10 サイコロの目グリッド本体
        st.markdown(
            render_10x10_dice_grid_html(st.session_state.current_grid),
            unsafe_allow_html=True,
        )


# ---------------- 右カラム: 相対度数ヒストグラム ----------------
with col_right:
    with st.container(border=True):
        # 階級幅セレクター（3段階）と正規分布曲線トグルを横並びに配置
        top_bar_cols = st.columns([6.5, 3.5])

        BIN_STEP_LABELS = {
            10: "幅 0.10",
            5: "幅 0.05",
            2: "幅 0.02",
        }

        with top_bar_cols[0]:
            selected_step = st.radio(
                "階級幅の選択",
                options=[10, 5, 2],
                format_func=lambda x: BIN_STEP_LABELS[x],
                index=[10, 5, 2].index(st.session_state.bin_step),
                horizontal=True,
            )
            if selected_step != st.session_state.bin_step:
                st.session_state.bin_step = selected_step
                st.rerun()

        with top_bar_cols[1]:
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            st.session_state.show_normal_curve = st.toggle(
                "正規分布の理論曲線を表示",
                value=st.session_state.show_normal_curve,
            )

        # Plotly ヒストグラム描画
        fig = create_histogram_figure(all_bins)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ============================================================================
# 8. 連続実行モードの自動更新ループ
# ============================================================================
if st.session_state.is_running:
    time.sleep(SLEEP_INTERVAL)
    run_trials(STEP_BATCH_SIZE)
    st.rerun()
