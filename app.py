import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from streamlit_gsheets import GSheetsConnection

# =========================================================
# UI
# =========================================================
st.set_page_config(page_title="Macro Analytical Dashboard For Bank - Author: Le Hoang Quan", layout="wide")
st.title("🏦 Macro Analytical Dashboard For Bank - Author: Le Hoang Quan")
st.caption("Dashboard kết hợp dự báo, giải thích các yếu tố tác động, backtest, regime, chiến lược thị trường 1 và thị trường 2")
mobile_mode = st.toggle("📱 Mobile-friendly mode", value=False, help="Tối ưu bố cục cho màn hình điện thoại bằng cách chuyển các khối nhiều cột thành dạng xếp dọc.")
st.markdown("---")

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("⚙️ Trung tâm dữ liệu")
data_mode = st.sidebar.radio(
    "Nguồn dữ liệu",
    ["Tự động từ Google Sheets", "Upload file Excel"],
    index=0
)
forecast_horizon = st.sidebar.selectbox("Forecast horizon (months)", [1, 3, 6], index=1)
backtest_points = st.sidebar.slider("Backtest points", 6, 36, 12)
show_tail = st.sidebar.slider("Rows to preview", 5, 30, 10)
if st.sidebar.button("🔄 Làm mới dữ liệu"):
    st.cache_data.clear()
    st.rerun()

if mobile_mode:
    st.info("Chế độ mobile-friendly đang bật. Các khối số liệu và biểu đồ sẽ được xếp dọc để phù hợp hơn với màn hình điện thoại.")

@st.cache_data(ttl=600)
def load_data_from_gsheet():
    conn = st.connection("gsheets", type=GSheetsConnection)
    return conn.read()

raw = None
if data_mode == "Tự động từ Google Sheets":
    try:
        raw = load_data_from_gsheet()
        if raw is not None and len(raw) > 0:
            st.sidebar.success("Đã kết nối Google Sheets")
        else:
            st.sidebar.warning("Google Sheets không trả về dữ liệu")
    except Exception as e:
        st.sidebar.error(f"Lỗi kết nối Google Sheets: {e}")
else:
    uploaded_file = st.sidebar.file_uploader("Upload macro_data.xlsx", type=["xlsx"])
    if uploaded_file is not None:
        raw = pd.read_excel(uploaded_file)
        st.sidebar.success("Đã nạp file Excel")

# =========================================================
# HELPERS
# =========================================================
def responsive_cols(n):
    if mobile_mode:
        return [st.container() for _ in range(n)]
    return st.columns(n)

def responsive_ratio(spec):
    if mobile_mode:
        return [st.container() for _ in range(len(spec))]
    return st.columns(spec)

def preprocess(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    for c in df.columns:
        if c != 'date':
            df[c] = pd.to_numeric(df[c], errors='coerce')

    df = df.ffill().bfill()

    if 'vibor_on' not in df.columns:
        raise ValueError("Thiếu cột vibor_on")
    if 'usd_vnd' not in df.columns:
        raise ValueError("Thiếu cột usd_vnd")

    # Dùng EMA để tạo trend ổn định, tránh phụ thuộc statsmodels
    df['ir_trend'] = df['vibor_on'].ewm(span=6, adjust=False).mean()
    df['fx_trend'] = df['usd_vnd'].ewm(span=6, adjust=False).mean()

    # Một số feature kỹ thuật đơn giản để dashboard phong phú hơn
    df['fx_mom_1m'] = df['usd_vnd'].pct_change(1) * 100
    df['fx_mom_3m'] = df['usd_vnd'].pct_change(3) * 100
    df['ir_chg_1m'] = df['vibor_on'].diff(1)
    df['ir_chg_3m'] = df['vibor_on'].diff(3)

    return df.ffill().bfill()


def detect_features(df):
    excluded = ['date', 'usd_vnd', 'vibor_on', 'fx_trend', 'ir_trend']
    features = [c for c in df.columns if c not in excluded]
    return features


def build_data(df, target, features, horizon):
    work = df[['date', target] + features].copy()

    for f in features:
        work[f + '_lag1'] = work[f].shift(1)
        work[f + '_chg1'] = work[f].diff(1)

    work['target_future'] = work[target].shift(-horizon)
    work = work.dropna().reset_index(drop=True)

    model_cols = [c for c in work.columns if c not in ['date', target, 'target_future']]
    X = work[model_cols]
    y = work['target_future']
    meta = work[['date', target, 'target_future']].copy()
    return X, y, meta, model_cols


def train_forecast(df, target, features, horizon):
    X, y, meta, model_cols = build_data(df, target, features, horizon)
    if len(X) < 24:
        return None

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = Ridge(alpha=1.0)
    model.fit(Xs, y)

    X_last = X.iloc[[-1]]
    X_last_s = scaler.transform(X_last)
    forecast = model.predict(X_last_s)[0]

    contrib_raw = pd.Series(model.coef_ * X_last_s[0], index=model_cols)
    grouped = {}
    for idx, val in contrib_raw.items():
        base = idx.replace('_lag1', '').replace('_chg1', '')
        grouped[base] = grouped.get(base, 0.0) + float(val)
    contrib = pd.Series(grouped).sort_values(key=lambda s: np.abs(s), ascending=False)

    coef = pd.Series(model.coef_, index=model_cols).sort_values(key=lambda s: np.abs(s), ascending=False)
    last_value = float(df[target].iloc[-1])

    return {
        'forecast': float(forecast),
        'last_value': last_value,
        'delta': float(forecast - last_value),
        'contrib': contrib,
        'coef': coef,
        'model': model,
        'scaler': scaler,
        'X': X,
        'y': y,
        'meta': meta,
        'model_cols': model_cols
    }


def rolling_backtest(df, target, features, horizon, n_points=12):
    X, y, meta, model_cols = build_data(df, target, features, horizon)
    if len(X) < max(24, n_points + 6):
        return None

    preds, actuals, dates = [], [], []
    start = max(24, len(X) - n_points)

    for i in range(start, len(X)):
        X_train = X.iloc[:i]
        y_train = y.iloc[:i]
        X_test = X.iloc[[i]]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = Ridge(alpha=1.0)
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)[0]

        preds.append(pred)
        actuals.append(float(y.iloc[i]))
        dates.append(meta.iloc[i]['date'])

    bt = pd.DataFrame({'date': dates, 'actual': actuals, 'pred': preds})
    bt['error'] = bt['pred'] - bt['actual']
    bt['abs_pct_error'] = np.where(bt['actual'] != 0, np.abs(bt['error'] / bt['actual']) * 100, np.nan)
    return bt


def summarize_backtest(bt):
    mae = mean_absolute_error(bt['actual'], bt['pred'])
    rmse = np.sqrt(mean_squared_error(bt['actual'], bt['pred']))
    mape = np.nanmean(bt['abs_pct_error'])
    hit = (np.sign(bt['actual'].diff().fillna(0)) == np.sign(bt['pred'].diff().fillna(0))).mean() * 100
    return mae, rmse, mape, hit


def regime_signal(series):
    s = series.dropna()
    latest = float(s.iloc[-1])
    mean = float(s.mean())
    std = float(s.std()) if float(s.std()) != 0 else 1e-9
    z = (latest - mean) / std
    pct = float(s.rank(pct=True).iloc[-1] * 100)
    return latest, mean, z, pct


def top_driver_text(contrib, label):
    pos = contrib[contrib > 0].head(3)
    neg = contrib[contrib < 0].head(3)
    out = []
    if len(pos) > 0:
        out.append(f"Các biến đang kéo {label} tăng: " + ", ".join([f"{k} ({v:+.2f})" for k, v in pos.items()]) + ".")
    if len(neg) > 0:
        out.append(f"Các biến đang ghìm {label} xuống: " + ", ".join([f"{k} ({v:+.2f})" for k, v in neg.items()]) + ".")
    if not out:
        out.append(f"Chưa có driver đủ nổi bật cho {label}.")
    return " ".join(out)


def plot_series_with_forecast(df, raw_col, trend_col, forecast_value, horizon, title):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    hist = df.tail(18)
    ax.plot(hist['date'], hist[raw_col], marker='o', linewidth=2, label='Actual')
    ax.plot(hist['date'], hist[trend_col], linestyle='--', linewidth=2, label='Trend')
    future_date = hist['date'].iloc[-1] + pd.DateOffset(months=horizon)
    ax.plot([hist['date'].iloc[-1], future_date], [hist[trend_col].iloc[-1], forecast_value], linestyle=':', marker='o', linewidth=2, label='Forecast')
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    return fig


def plot_contrib(contrib, title, n=10):
    show = contrib.head(n).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(show.index, show.values)
    ax.set_title(title)
    ax.grid(axis='x', alpha=0.3)
    return fig


def plot_backtest(bt, title):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(bt['date'], bt['actual'], marker='o', linewidth=2, label='Actual')
    ax.plot(bt['date'], bt['pred'], marker='o', linestyle='--', linewidth=2, label='Pred')
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    return fig


def safe_get(df, col):
    return col in df.columns


def scenario_analysis(base_df, target, features, horizon, shocks):
    temp = base_df.copy()
    last_idx = temp.index[-1]
    for col, shock in shocks.items():
        if col in temp.columns:
            temp.loc[last_idx, col] = temp.loc[last_idx, col] + shock
    res = train_forecast(temp, target, features, horizon)
    return res['forecast'] if res is not None else np.nan


def estimate_price_change_from_duration(duration_years, rate_shock_pct):
    return -duration_years * rate_shock_pct


def estimate_mtm_loss(portfolio_value_billion_vnd, duration_years, rate_shock_pct):
    price_change_pct = estimate_price_change_from_duration(duration_years, rate_shock_pct)
    pnl_billion_vnd = portfolio_value_billion_vnd * price_change_pct / 100
    return price_change_pct, pnl_billion_vnd


def duration_risk_table(portfolio_value_billion_vnd, durations, shocks_bps):
    rows = []
    for duration in durations:
        row = {'Duration (years)': duration}
        for shock_bps in shocks_bps:
            shock_pct = shock_bps / 100.0
            price_change_pct, pnl = estimate_mtm_loss(portfolio_value_billion_vnd, duration, shock_pct)
            row[f'{shock_bps}bps price change %'] = price_change_pct
            row[f'{shock_bps}bps P&L (bn VND)'] = pnl
        rows.append(row)
    return pd.DataFrame(rows)


def plot_duration_price_sensitivity(durations, shock_bps):
    shock_pct = shock_bps / 100.0
    changes = [estimate_price_change_from_duration(d, shock_pct) for d in durations]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([str(d) for d in durations], changes)
    ax.set_title(f'Ước lượng thay đổi giá khi lãi suất tăng {shock_bps}bps')
    ax.set_xlabel('Duration (năm)')
    ax.set_ylabel('Price change (%)')
    ax.grid(axis='y', alpha=0.3)
    return fig


def estimate_dv01_billion_vnd(portfolio_value_billion_vnd, duration_years):
    market_value_vnd = portfolio_value_billion_vnd * 1_000_000_000
    dv01_vnd = duration_years * market_value_vnd * 0.0001
    return dv01_vnd / 1_000_000_000


def compute_bucket_metrics(total_portfolio_billion_vnd, bucket_weights, bucket_durations, shock_bps):
    rows = []
    for bucket, weight in bucket_weights.items():
        alloc = total_portfolio_billion_vnd * weight
        dur = bucket_durations[bucket]
        dv01 = estimate_dv01_billion_vnd(alloc, dur)
        price_change_pct, pnl = estimate_mtm_loss(alloc, dur, shock_bps / 100.0)
        rows.append({
            'Bucket': bucket,
            'Weight %': weight * 100,
            'Allocated value (bn VND)': alloc,
            'Duration (years)': dur,
            'DV01 (bn VND / 1bp)': dv01,
            f'Price change @ {shock_bps}bps (%)': price_change_pct,
            f'P&L @ {shock_bps}bps (bn VND)': pnl
        })
    return pd.DataFrame(rows)


def stress_loss_heatmap_df(total_portfolio_billion_vnd, duration_years_list, shock_bps_list):
    data = {}
    for shock_bps in shock_bps_list:
        col_vals = []
        for dur in duration_years_list:
            _, pnl = estimate_mtm_loss(total_portfolio_billion_vnd, dur, shock_bps / 100.0)
            col_vals.append(pnl)
        data[f'{shock_bps}bps'] = col_vals
    return pd.DataFrame(data, index=[f'{d}y' for d in duration_years_list])


def plot_stress_loss_heatmap(df_heatmap):
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(df_heatmap.values, aspect='auto')
    ax.set_xticks(range(len(df_heatmap.columns)))
    ax.set_xticklabels(df_heatmap.columns)
    ax.set_yticks(range(len(df_heatmap.index)))
    ax.set_yticklabels(df_heatmap.index)
    ax.set_title('Heatmap stress loss theo duration và cú sốc lãi suất')
    ax.set_xlabel('Rate shock')
    ax.set_ylabel('Duration bucket')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('P&L (bn VND)')
    for i in range(df_heatmap.shape[0]):
        for j in range(df_heatmap.shape[1]):
            ax.text(j, i, f"{df_heatmap.iloc[i, j]:,.0f}", ha='center', va='center', fontsize=8)
    return fig


def plot_bucket_pnl(df_bucket, pnl_col):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(df_bucket['Bucket'], df_bucket[pnl_col])
    ax.set_title('Stress loss theo duration bucket')
    ax.set_ylabel('P&L (bn VND)')
    ax.grid(axis='y', alpha=0.3)
    return fig


def weighted_average_duration(bucket_weights, bucket_durations):
    return sum(bucket_weights[k] * bucket_durations[k] for k in bucket_weights)


def classify_regime(percentile):
    if percentile >= 90:
        return "Rất căng"
    if percentile >= 75:
        return "Căng"
    if percentile <= 25:
        return "Thấp"
    return "Bình thường"


def classify_duration_risk(stress_loss_billion_vnd, portfolio_value_billion_vnd):
    ratio = abs(stress_loss_billion_vnd) / portfolio_value_billion_vnd * 100
    if ratio >= 5:
        return "Rủi ro cao"
    if ratio >= 2:
        return "Cảnh báo"
    return "An toàn"


def tab_hint(tab_idx, total_tabs=9):
    if mobile_mode:
        st.info("👉 Dashboard có nhiều tab. Vuốt ngang trên thanh tab để xem các phần phân tích khác.")
        st.caption(f"Tab {tab_idx}/{total_tabs} • Vuốt ngang để xem tiếp")
    else:
        st.info("👉 Dashboard có nhiều tab. Click các tab phía trên để xem các phần phân tích khác.")
        st.caption(f"Tab {tab_idx}/{total_tabs}")


def plot_missing_values(df):
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(8, 4))
    if len(miss) == 0:
        ax.text(0.5, 0.5, "Không có giá trị thiếu", ha="center", va="center")
        ax.axis("off")
    else:
        ax.barh(miss.index, miss.values)
        ax.set_title("Số lượng giá trị thiếu theo biến")
        ax.grid(axis='x', alpha=0.3)
    return fig

def plot_fx_ir_overview(df):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    hist = df.tail(24)
    ax.plot(hist['date'], hist['usd_vnd'], label='USD/VND', linewidth=2)
    ax2 = ax.twinx()
    ax2.plot(hist['date'], hist['vibor_on'], label='VIBOR ON', linewidth=2, linestyle='--')
    ax.set_title("Diễn biến USD/VND và VIBOR ON")
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    return fig

def plot_correlation_heatmap_like(df, cols):
    corr = df[cols].corr().round(2)
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(corr.values, aspect='auto')
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    ax.set_title("Tương quan giữa các biến chính")
    plt.colorbar(im, ax=ax)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha='center', va='center', fontsize=7)
    return fig

# =========================================================
# MAIN
# =========================================================
if raw is not None and len(raw) > 0:
    df = preprocess(raw)
    features = detect_features(df)

    
    st.info("Dữ liệu đã được nạp. Dashboard đang chạy realtime và sẽ tự cập nhật khi thay đổi tham số, nguồn dữ liệu hoặc chế độ hiển thị.")

    if True:
        fx_res = train_forecast(df, 'fx_trend', features, forecast_horizon)
        ir_res = train_forecast(df, 'ir_trend', features, forecast_horizon)
        bt_fx = rolling_backtest(df, 'fx_trend', features, forecast_horizon, backtest_points)
        bt_ir = rolling_backtest(df, 'ir_trend', features, forecast_horizon, backtest_points)

        if fx_res is None or ir_res is None:
            st.error("Dữ liệu hiện quá ngắn. Sau khi tạo lag và horizon, cần có tối thiểu khoảng 24 quan sát hữu ích.")
        else:
            tabs = st.tabs([
                "📊 Tổng quan dữ liệu",
                "💵 Dự báo tỷ giá",
                "📈 Dự báo lãi suất ON",
                "🧪 Backtest",
                "📊 Market regime",
                "🧭 Strategy market 1 & 2",
                "📉 Duration risk",
                "🌪️ Scenario analysis",
                "📋 CEO note"
            ])

            with tabs[0]:
                st.subheader("Tổng quan dữ liệu")
                tab_hint(1, 9)

                preview_rows = 5 if mobile_mode else show_tail
                s1, s2, s3, s4 = responsive_cols(4)
                s1.metric("Số quan sát", len(df))
                s2.metric("Ngày đầu", df['date'].min().strftime('%Y-%m-%d'))
                s3.metric("Ngày cuối", df['date'].max().strftime('%Y-%m-%d'))
                s4.metric("Số biến giải thích", len(features))

                top1, top2 = responsive_cols(2)
                with top1:
                    st.pyplot(plot_fx_ir_overview(df))
                with top2:
                    st.pyplot(plot_missing_values(raw))

                corr_cols = [c for c in ['usd_vnd', 'vibor_on', 'dxy_index', 'us_10y_yield', 'fed_rate', 'gold_price'] if c in df.columns]
                if len(corr_cols) >= 2:
                    st.pyplot(plot_correlation_heatmap_like(df, corr_cols))

                st.markdown("### Dữ liệu gần nhất")
                st.dataframe(df.tail(preview_rows), width="stretch")
                st.caption("Dữ liệu được cập nhật tự động từ Google Sheets.")

            with tabs[1]:
                st.subheader("Dự báo tỷ giá")
                tab_hint(2, 9)

                c1, c2 = responsive_cols(2)
                with c1:
                    st.pyplot(plot_series_with_forecast(df, 'usd_vnd', 'fx_trend', fx_res['forecast'], forecast_horizon, 'USD/VND actual vs trend vs forecast'))
                with c2:
                    st.pyplot(plot_contrib(fx_res['contrib'], 'Top FX drivers', 10))

                st.markdown("### Diễn giải nhanh")
                st.write(top_driver_text(fx_res['contrib'], 'tỷ giá'))
                st.write("Forecast FX không chỉ phản ánh quán tính của tỷ giá, mà còn tính đến tác động của các biến vĩ mô khác và thay đổi ngắn hạn của chúng. Vì vậy, nếu DXY, lợi suất Mỹ, vàng, thanh khoản hoặc các biến ngoại thương thay đổi mạnh, dự báo sẽ đổi theo.")

                st.markdown("### Hệ số nhạy")
                st.dataframe(fx_res["coef"].head(15).rename("coefficient").to_frame(), width="stretch")

                left, right = responsive_ratio([1, 1])
                with left:
                    st.pyplot(plot_contrib(fx_res['contrib'], 'Driver contribution into FX forecast', 12))
                with right:
                    st.markdown("### Giải thích sâu")
                    st.write(top_driver_text(fx_res['contrib'], 'tỷ giá'))
                    st.write("Cột dương là các biến đóng góp làm tăng forecast tỷ giá; cột âm là các biến đóng góp làm giảm forecast. Độ lớn thể hiện mức độ chi phối của từng biến ở thời điểm hiện tại.")

                st.markdown("### Lưu ý khi ra quyết định")
                st.write("- Nếu tỷ giá dự báo tăng nhưng đang ở vùng rất cao so với lịch sử, cần cảnh giác với rủi ro phản ứng quá mức.")
                st.write("- Nếu contribution tập trung vào 1–2 biến, dự báo sẽ nhạy hơn với nhiễu dữ liệu ở các biến đó.")
                st.write("- Nếu backtest gần đây suy giảm, nên dùng forecast như tín hiệu định hướng thay vì tín hiệu giao dịch cứng.")

            with tabs[2]:
                st.subheader("Dự báo lãi suất ON")
                tab_hint(3, 9)

                c1, c2 = responsive_cols(2)
                with c1:
                    st.pyplot(plot_series_with_forecast(df, 'vibor_on', 'ir_trend', ir_res['forecast'], forecast_horizon, 'VIBOR ON actual vs trend vs forecast'))
                with c2:
                    st.pyplot(plot_contrib(ir_res['contrib'], 'Top IR drivers', 10))

                st.markdown("### Diễn giải nhanh")
                st.write(top_driver_text(ir_res['contrib'], 'lãi suất'))
                st.write("Lãi suất ON thường nhạy hơn với thanh khoản hệ thống, OMO, áp lực tỷ giá, chênh lệch lãi suất quốc tế và lạm phát.")

                st.markdown("### Hệ số nhạy")
                st.dataframe(ir_res["coef"].head(15).rename("coefficient").to_frame(), width="stretch")

                left, right = responsive_ratio([1, 1])
                with left:
                    st.pyplot(plot_contrib(ir_res['contrib'], 'Driver contribution into IR forecast', 12))
                with right:
                    st.markdown("### Giải thích sâu")
                    st.write(top_driver_text(ir_res['contrib'], 'lãi suất'))
                    st.write("Khi phân bố nghiêng mạnh về các biến thanh khoản và lãi suất quốc tế, lãi suất trong nước đang được dẫn dắt nhiều hơn bởi áp lực funding và lãi suất bên ngoài.")
                    st.write("Ngược lại, khi CPI, hoạt động kinh tế hoặc seasonal dummy nổi bật hơn, kết quả phản ánh nhiều hơn xu hướng nội tại của nền kinh tế.")

                st.markdown("### Cách sử dụng cho Treasury")
                st.write("- Forecast IR tăng: hạn chế kéo duration quá dài và cần quản trị cost of fund thận trọng hơn.")
                st.write("- Forecast IR giảm hoặc ổn định: có thể cân nhắc mở duration chọn lọc, tối ưu carry trên danh mục đầu tư.")

            with tabs[3]:
                st.subheader("Backtest và độ tin cậy")
                tab_hint(4, 9)
                c1, c2 = responsive_cols(2)
                with c1:
                    if bt_fx is not None:
                        st.pyplot(plot_backtest(bt_fx, 'FX backtest'))
                        mae, rmse, mape, hit = summarize_backtest(bt_fx)
                        r1, r2, r3, r4 = responsive_cols(4)
                        r1.metric('MAE FX', f'{mae:,.2f}')
                        r2.metric('RMSE FX', f'{rmse:,.2f}')
                        r3.metric('MAPE FX', f'{mape:.2f}%')
                        r4.metric('Hit Ratio FX', f'{hit:.1f}%')
                with c2:
                    if bt_ir is not None:
                        st.pyplot(plot_backtest(bt_ir, 'IR backtest'))
                        mae, rmse, mape, hit = summarize_backtest(bt_ir)
                        r1, r2, r3, r4 = responsive_cols(4)
                        r1.metric('MAE IR', f'{mae:,.2f}')
                        r2.metric('RMSE IR', f'{rmse:,.2f}')
                        r3.metric('MAPE IR', f'{mape:.2f}%')
                        r4.metric('Hit Ratio IR', f'{hit:.1f}%')

                st.warning("Ghi chú: Backtest giúp đánh giá mức độ chính xác của mô hình khi so kết quả dự báo với dự liệu thật trong các kỳ gần đây.")

            with tabs[4]:
                st.subheader("Market regime")
                tab_hint(5, 9)
                fx_latest, fx_mean, fx_z, fx_pct = regime_signal(df['usd_vnd'])
                ir_latest, ir_mean, ir_z, ir_pct = regime_signal(df['vibor_on'])
                a, b, c, d = responsive_cols(4)
                a.metric('FX vs mean', f'{fx_latest:,.0f}', f'z={fx_z:.2f}')
                b.metric('FX percentile', f'{fx_pct:.1f}%')
                c.metric('IR vs mean', f'{ir_latest:.2f}%', f'z={ir_z:.2f}')
                d.metric('IR percentile', f'{ir_pct:.1f}%')

                st.markdown("### Cách đọc")
                st.write("- Percentile cao nghĩa là biến đang ở vùng lịch sử tương đối cao; rủi ro mean reversion tăng lên.")
                st.write("- Z-score giúp bạn biết thị trường đang chỉ hơi căng, căng vừa hay căng mạnh.")
                st.write("- Khi kết hợp forecast với regime, quyết định sẽ thực tế hơn nhiều so với chỉ nhìn hướng tăng/giảm đơn thuần.")

            with tabs[5]:
                st.subheader("Hàm ý chiến lược cho thị trường 1 và thị trường 2")
                tab_hint(6, 9)
                m1, m2 = responsive_cols(2)
                with m1:
                    st.info("**Thị trường 1 – Lending / Funding / Balance Sheet**")
                    if ir_res['delta'] > 0.15:
                        st.write("- Lãi suất có xu hướng tăng: ưu tiên bảo vệ NIM, kiểm soát repricing gap và kéo dài kỳ hạn huy động chọn lọc.")
                    else:
                        st.write("- Lãi suất chưa tăng mạnh: có thể linh hoạt hơn trong pricing tín dụng và tối ưu tăng trưởng tài sản sinh lãi.")
                    if fx_res['delta'] > 80:
                        st.write("- Tỷ giá dự báo tăng: rà soát khách hàng nhập khẩu, khách hàng vay ngoại tệ và áp lực chuyển dịch sang VND funding.")
                    else:
                        st.write("- Tỷ giá không quá căng: áp lực lên khách hàng ngoại tệ và nhu cầu hedge có thể ở mức vừa phải hơn.")

                with m2:
                    st.warning("**Thị trường 2 – Treasury / Investment Book / FX**")
                    if fx_res['delta'] > 80:
                        st.write("- Nghiêng về trạng thái FX phòng thủ hơn; ưu tiên hedge ngắn hạn thay vì mở vị thế directional lớn.")
                    else:
                        st.write("- FX chưa quá căng: tập trung vào tối ưu carry và vốn VND.")
                    if ir_res['delta'] > 0.15:
                        st.write("- Hạn chế kéo duration quá dài; cẩn trọng với P&L mark-to-market trên sổ đầu tư.")
                    else:
                        st.write("- Có thể cân nhắc mở duration chọn lọc khi lợi suất ổn định hơn và funding bớt áp lực.")

                st.markdown("### Góc nhìn điều hành")
                st.write("Tab này khôi phục đúng tinh thần dashboard trước của bạn: forecast phải chuyển hóa thành hành động cụ thể cho market 1 và market 2, chứ không dừng ở việc dự báo thuần túy.")

            with tabs[6]:
                st.subheader("Duration risk module")
                tab_hint(7, 9)
                if mobile_mode:
                    st.caption("Chế độ mobile hiển thị theo dạng xếp dọc để dễ theo dõi trên điện thoại.")
                st.write("Module này lượng hóa mức độ nhạy cảm của danh mục đầu tư đối với biến động lãi suất, bao gồm modified duration, DV01, stress loss, phân bổ duration bucket và tác động lên danh mục chuẩn 50.000 tỷ VND.")

                dcol1, dcol2, dcol3 = responsive_cols(3)
                with dcol1:
                    portfolio_value = st.number_input("Quy mô danh mục (tỷ VND)", min_value=100.0, value=50000.0, step=100.0)
                with dcol2:
                    assumed_duration = st.number_input("Modified duration giả định (năm)", min_value=0.5, value=3.0, step=0.5)
                with dcol3:
                    shock_bps = st.selectbox("Cú sốc lãi suất", [25, 50, 100, 150, 200], index=2)

                price_change_pct, pnl = estimate_mtm_loss(portfolio_value, assumed_duration, shock_bps / 100.0)
                dv01_total = estimate_dv01_billion_vnd(portfolio_value, assumed_duration)

                m1, m2, m3, m4 = responsive_cols(4)
                m1.metric("Quy mô danh mục", f"{portfolio_value:,.0f} tỷ")
                m2.metric("Modified duration", f"{assumed_duration:.1f} năm")
                m3.metric("DV01 ước tính", f"{dv01_total:,.2f} tỷ / 1bp")
                m4.metric("Stress P&L", f"{pnl:,.0f} tỷ")

                l1, l2 = responsive_cols(2)
                with l1:
                    st.pyplot(plot_duration_price_sensitivity([1, 2, 3, 5, 7, 10], shock_bps))
                with l2:
                    duration_grid = [1, 2, 3, 5, 7, 10]
                    shock_grid = [25, 50, 100, 150, 200]
                    heatmap_df = stress_loss_heatmap_df(portfolio_value, duration_grid, shock_grid)
                    st.pyplot(plot_stress_loss_heatmap(heatmap_df))

                st.markdown("### Duration bucket analysis")
                st.write("Phần này mô phỏng cơ cấu danh mục 50.000 tỷ VND theo các bucket duration. Tỷ trọng có thể điều chỉnh để đánh giá mức độ tập trung rủi ro theo kỳ hạn.")

                b1, b2, b3, b4 = responsive_cols(4)
                with b1:
                    w_short = st.slider("0–1 năm (%)", 0, 100, 25, 5) / 100
                with b2:
                    w_1_3 = st.slider("1–3 năm (%)", 0, 100, 35, 5) / 100
                with b3:
                    w_3_5 = st.slider("3–5 năm (%)", 0, 100, 25, 5) / 100
                with b4:
                    w_5_plus = st.slider("5+ năm (%)", 0, 100, 15, 5) / 100

                weight_sum = w_short + w_1_3 + w_3_5 + w_5_plus
                if abs(weight_sum - 1.0) > 1e-9:
                    st.warning(f"Tổng tỷ trọng hiện là {weight_sum*100:.1f}%. Cần điều chỉnh về 100% để phân tích chính xác.")
                else:
                    bucket_weights = {
                        '0-1Y': w_short,
                        '1-3Y': w_1_3,
                        '3-5Y': w_3_5,
                        '5Y+': w_5_plus
                    }
                    bucket_durations = {
                        '0-1Y': 0.5,
                        '1-3Y': 2.0,
                        '3-5Y': 4.0,
                        '5Y+': 7.0
                    }
                    wad = weighted_average_duration(bucket_weights, bucket_durations)
                    bucket_df = compute_bucket_metrics(portfolio_value, bucket_weights, bucket_durations, shock_bps)
                    pnl_col = f'P&L @ {shock_bps}bps (bn VND)'

                    k1, k2, k3 = responsive_cols(3)
                    k1.metric("Weighted avg duration", f"{wad:.2f} năm")
                    k2.metric("Tổng DV01 bucket", f"{bucket_df['DV01 (bn VND / 1bp)'].sum():,.2f} tỷ / 1bp")
                    k3.metric("Tổng stress loss", f"{bucket_df[pnl_col].sum():,.0f} tỷ")

                    c1, c2 = responsive_cols(2)
                    with c1:
                        st.dataframe(bucket_df, width="stretch")
                    with c2:
                        st.pyplot(plot_bucket_pnl(bucket_df, pnl_col))

                st.markdown("### Ý nghĩa điều hành")
                st.write("- DV01 cho biết danh mục biến động bao nhiêu tỷ VND khi lợi suất dịch chuyển 1 điểm cơ bản; đây là thước đo rất hữu ích để đặt hạn mức rủi ro.")
                st.write("- Heatmap stress loss cho thấy tổn thất mark-to-market thay đổi như thế nào khi duration dài hơn hoặc cú sốc lãi suất lớn hơn.")
                st.write("- Duration bucket analysis giúp nhận diện cụ thể bucket nào đang đóng góp lớn nhất vào rủi ro định giá.")
                st.write("- Với danh mục chuẩn 50.000 tỷ VND, việc tăng tỷ trọng bucket dài sẽ làm DV01 và stress loss tăng lên rõ rệt.")
                st.write("- Trong bối cảnh Forecast IR tăng, nên ưu tiên kiểm soát bucket dài, giới hạn DV01 và theo dõi đồng thời tác động lên cost of fund.")

            with tabs[7]:
                st.subheader("Scenario analysis")
                tab_hint(6, 9)
                shock_sets = {
                    'Base': {},
                    'DXY +1': {'dxy_index': 1.0} if safe_get(df, 'dxy_index') else {},
                    'US10Y +0.25': {'us_10y_yield': 0.25} if safe_get(df, 'us_10y_yield') else {},
                    'Fed +0.25': {'fed_rate': 0.25} if safe_get(df, 'fed_rate') else {},
                    'OMO +1000': {'omo_outstanding': 1000.0} if safe_get(df, 'omo_outstanding') else {},
                    'Gold +50': {'gold_price': 50.0} if safe_get(df, 'gold_price') else {}
                }

                rows = []
                for name, shocks in shock_sets.items():
                    fx_val = scenario_analysis(df, 'fx_trend', features, forecast_horizon, shocks)
                    ir_val = scenario_analysis(df, 'ir_trend', features, forecast_horizon, shocks)
                    rows.append({'Scenario': name, 'FX Forecast': fx_val, 'IR Forecast': ir_val})
                scen = pd.DataFrame(rows)
                st.dataframe(scen, width="stretch", height=260 if mobile_mode else "auto")
                st.caption("Scenario analysis giúp dashboard dài hơn và hữu ích hơn: không chỉ thể hiện forecast cơ sở mà còn cho thấy mức độ nhạy của forecast khi các driver chính bị shock.")

            with tabs[8]:
                st.subheader("CEO narrative")
                tab_hint(7, 9)
                fx_text = top_driver_text(fx_res['contrib'], 'tỷ giá')
                ir_text = top_driver_text(ir_res['contrib'], 'lãi suất')
                note = f"""
**1. Kết quả chính**
- Sau {forecast_horizon} tháng, hệ thống dự báo xu hướng tỷ giá ở mức **{fx_res['forecast']:,.0f}** và xu hướng lãi suất qua đêm ở mức **{ir_res['forecast']:.2f}%**.
- So với trend hiện tại, tỷ giá thay đổi **{fx_res['delta']:+,.0f}** và lãi suất thay đổi **{ir_res['delta']:+.2f} điểm**.

**2. Vì sao có kết quả này**
- Với tỷ giá: {fx_text}
- Với lãi suất: {ir_text}

**3. Mức độ tin cậy**
- Dashboard đi kèm backtest để đánh giá độ phù hợp của mô hình trong các kỳ gần đây. Khi sai số cao hoặc hit ratio thấp, forecast nên được dùng như tín hiệu định hướng thay vì tín hiệu giao dịch cứng.

**4. Hàm ý điều hành**
- Nếu lãi suất tăng, cần ưu tiên bảo vệ NIM, quản trị funding và duration thận trọng hơn.
- Nếu tỷ giá tăng, cần tăng cường quản trị trạng thái ngoại tệ và rà soát nhóm khách hàng nhạy cảm với nhập khẩu / vay ngoại tệ.
- Nếu regime đã ở vùng lịch sử rất cao, cần thận trọng với khả năng đảo chiều ngắn hạn.

**5. Cách đọc dashboard**
- Executive summary: nhìn nhanh mức hiện tại, forecast và biến động gần đây.
- FX / IR overview: nhìn từng thị trường riêng biệt.
- Explain tabs: xem driver nào đang kéo forecast lên / xuống.
- Backtest: kiểm tra forecast có đáng tin không.
- Regime: hiểu thị trường đang căng tới mức nào so với lịch sử.
- Strategy tab: chuyển forecast thành hành động cho market 1 và market 2.
- Scenario tab: xem forecast nhạy thế nào khi biến chính bị shock.
"""
                st.markdown(note)

else:
    st.info("Không có dữ liệu đầu vào. Có thể chọn chế độ tự động từ Google Sheets hoặc upload file Excel thủ công.")
