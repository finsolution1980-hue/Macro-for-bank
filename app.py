import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
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
st.sidebar.caption("Secrets cần có: gsheets_input cho dữ liệu đầu vào và gsheets_history cho lịch sử forecast.")
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
    conn = st.connection("gsheets_input", type=GSheetsConnection)
    return conn.read()

def get_history_conn():
    return st.connection("gsheets_history", type=GSheetsConnection)

st.sidebar.caption("Dữ liệu đầu vào và lịch sử forecast đang dùng 2 Google Sheet tách biệt. Sheet lịch sử có thể để trống ban đầu; app sẽ tự append các dòng forecast mới.")

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


def forecast_confidence_band(model_name, forecast_value, backtest_df):
    if backtest_df is None or len(backtest_df) == 0:
        return forecast_value, forecast_value, np.nan
    rmse = float(np.sqrt(mean_squared_error(backtest_df['actual'], backtest_df['pred'])))
    if model_name == "Random Forest":
        width = 1.15 * rmse
    elif model_name == "Linear Regression":
        width = 1.00 * rmse
    else:
        width = 1.05 * rmse
    return forecast_value - width, forecast_value + width, rmse

def model_candidates():
    return {
        "Ridge": lambda: Ridge(alpha=1.0),
        "Linear Regression": lambda: LinearRegression(),
        "Random Forest": lambda: RandomForestRegressor(
            n_estimators=200, max_depth=5, random_state=42, min_samples_leaf=2
        ),
    }

def rolling_backtest_for_model(df, target, features, horizon, n_points, model_name):
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
        model = model_candidates()[model_name]()
        model.fit(X_train_s, y_train)
        pred = float(model.predict(X_test_s)[0])
        preds.append(pred)
        actuals.append(float(y.iloc[i]))
        dates.append(meta.iloc[i]['date'])
    bt = pd.DataFrame({'date': dates, 'actual': actuals, 'pred': preds})
    bt['error'] = bt['pred'] - bt['actual']
    bt['abs_pct_error'] = np.where(bt['actual'] != 0, np.abs(bt['error'] / bt['actual']) * 100, np.nan)
    return bt

def evaluate_model_table(df, target, features, horizon, n_points):
    rows = []
    for model_name in model_candidates().keys():
        bt = rolling_backtest_for_model(df, target, features, horizon, n_points, model_name)
        if bt is None or len(bt) == 0:
            continue
        mae, rmse, mape, hit = summarize_backtest(bt)
        rows.append({'Model': model_name, 'MAE': mae, 'RMSE': rmse, 'MAPE %': mape, 'Hit Ratio %': hit})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(['RMSE', 'MAPE %']).reset_index(drop=True)

def plot_model_comparison(df_model, title):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(df_model['Model'], df_model['RMSE'])
    ax.set_title(title)
    ax.set_ylabel('RMSE')
    ax.grid(axis='y', alpha=0.3)
    return fig

def classify_risk_light(value, green_thres, yellow_thres):
    v = abs(value)
    if v <= green_thres:
        return "🟢 An toàn"
    if v <= yellow_thres:
        return "🟡 Cảnh báo"
    return "🔴 Rủi ro cao"

def previous_run_forecast_proxy(df, target, features, horizon):
    if len(df) < 30:
        return np.nan
    prev_df = df.iloc[:-1].copy()
    res = train_forecast(prev_df, target, features, horizon)
    return float(res['forecast']) if res is not None else np.nan


def dynamic_duration_limit(ir_delta):
    if ir_delta > 0.15:
        return 2.5
    elif ir_delta < -0.15:
        return 5.0
    return 3.5

def classify_risk_light_by_limit(value, limit):
    if limit == 0:
        return "N/A"
    ratio = abs(value) / limit
    if ratio <= 0.7:
        return "🟢 An toàn"
    elif ratio <= 1.0:
        return "🟡 Cảnh báo"
    return "🔴 Vượt hạn mức"

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
        model_table_fx = evaluate_model_table(df, 'fx_trend', features, forecast_horizon, backtest_points)
        model_table_ir = evaluate_model_table(df, 'ir_trend', features, forecast_horizon, backtest_points)
        fx_model_name = model_table_fx.iloc[0]['Model'] if len(model_table_fx) > 0 else "Ridge"
        ir_model_name = model_table_ir.iloc[0]['Model'] if len(model_table_ir) > 0 else "Ridge"
        fx_low, fx_high, fx_rmse = forecast_confidence_band(fx_model_name, fx_res['forecast'] if fx_res else np.nan, bt_fx)
        ir_low, ir_high, ir_rmse = forecast_confidence_band(ir_model_name, ir_res['forecast'] if ir_res else np.nan, bt_ir)
        prev_fx_forecast = previous_run_forecast_proxy(df, 'fx_trend', features, forecast_horizon)
        prev_ir_forecast = previous_run_forecast_proxy(df, 'ir_trend', features, forecast_horizon)

        if fx_res is None or ir_res is None:
            st.error("Dữ liệu hiện quá ngắn. Sau khi tạo lag và horizon, cần có tối thiểu khoảng 24 quan sát hữu ích.")
        else:
            tabs = st.tabs([
                "📊 Tổng quan dữ liệu",
                "💵 Dự báo tỷ giá",
                "📈 Dự báo lãi suất ON",
                "🧪 Backtest",
                "📉 Rủi ro kỳ hạn",
                "📊 Phân tích thị trường",
                "🧭 Khuyến nghị chiến lược",
                "📋 Tổng kết"
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

                    st.markdown("**1. Cấu trúc kết quả dự báo**")
                    st.write(
                        "Forecast tỷ giá được hình thành từ hai phần: (i) quán tính của xu hướng lịch sử (trend) "
                        "và (ii) tác động tại thời điểm hiện tại của các biến vĩ mô."
                    )

                    st.markdown("**2. Driver nào đang quyết định kết quả**")
                    st.write(
                        "Các biến có contribution lớn nhất đang chi phối hướng forecast. "
                        "Nếu contribution tập trung vào 1–2 biến, kết quả dự báo sẽ nhạy hơn với biến động của chính các biến đó."
                    )

                    st.markdown("**3. Độ ổn định của tín hiệu**")
                    st.write(
                        "Nếu các driver chính là các biến mang tính chu kỳ như DXY hoặc lợi suất Mỹ, "
                        "forecast có thể thay đổi nhanh khi điều kiện thị trường đảo chiều. "
                        "Ngược lại, nếu driver nổi bật là các biến nội tại như CPI hoặc hoạt động kinh tế, tín hiệu thường ổn định hơn."
                    )

                    st.markdown("**4. Cách sử dụng trong điều hành**")
                    st.write(
                        "Forecast nên được sử dụng như tín hiệu định hướng. "
                        "Khi kết hợp với phân tích regime và scenario, có thể đánh giá tốt hơn biên độ rủi ro "
                        "và tránh quyết định dựa trên một kịch bản đơn lẻ."
                    )

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

                    st.markdown("**1. Cấu trúc dự báo lãi suất**")
                    st.write(
                        "Forecast lãi suất ON phản ánh sự kết hợp giữa xu hướng thanh khoản hệ thống "
                        "và tác động tức thời của các biến như OMO, lãi suất quốc tế, tỷ giá và lạm phát."
                    )

                    st.markdown("**2. Nhóm driver chi phối**")
                    st.write(
                        "Nếu contribution tập trung vào các biến thanh khoản như OMO hoặc các chỉ báo thị trường tiền tệ, "
                        "lãi suất ON chủ yếu bị dẫn dắt bởi cung – cầu vốn ngắn hạn. "
                        "Nếu contribution đến nhiều từ yếu tố quốc tế, áp lực bên ngoài đang chi phối rõ hơn."
                    )

                    st.markdown("**3. Độ tin cậy của forecast**")
                    st.write(
                        "Lãi suất ON thường biến động mạnh trong ngắn hạn, do đó forecast có thể thay đổi nhanh khi điều kiện thanh khoản đổi hướng. "
                        "Cần kết hợp với backtest và regime để đánh giá mức độ tin cậy trước khi sử dụng cho quyết định vị thế lớn."
                    )

                    st.markdown("**4. Hàm ý cho Treasury**")
                    st.write(
                        "Kết quả forecast cần được chuyển hóa thành quản trị duration, DV01 và cost of fund. "
                        "Không nên chỉ nhìn vào hướng tăng/giảm mà cần đánh giá đồng thời cả mức độ nhạy của danh mục."
                    )

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
                st.subheader("Rủi ro kỳ hạn")
                tab_hint(5, 9)
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

                st.markdown("### Phân tích bucket kỳ hạn")
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
                    bucket_weights = {'0-1Y': w_short, '1-3Y': w_1_3, '3-5Y': w_3_5, '5Y+': w_5_plus}
                    bucket_durations = {'0-1Y': 0.5, '1-3Y': 2.0, '3-5Y': 4.0, '5Y+': 7.0}
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
                st.write("- DV01 cho biết danh mục biến động bao nhiêu tỷ VND khi lợi suất dịch chuyển 1 điểm cơ bản; đây là thước đo hữu ích để đặt hạn mức rủi ro.")
                st.write("- Heatmap stress loss cho thấy tổn thất mark-to-market thay đổi như thế nào khi duration dài hơn hoặc cú sốc lãi suất lớn hơn.")
                st.write("- Duration bucket analysis giúp nhận diện cụ thể bucket nào đang đóng góp lớn nhất vào rủi ro định giá.")
                st.write("- Với danh mục chuẩn 50.000 tỷ VND, việc tăng tỷ trọng bucket dài sẽ làm DV01 và stress loss tăng lên rõ rệt.")
                st.write("- Khi Forecast IR tăng, nên ưu tiên kiểm soát bucket dài, giới hạn DV01 và theo dõi đồng thời tác động lên cost of fund.")

            with tabs[5]:
                st.subheader("Hạn mức rủi ro")
                tab_hint(6, 9)

                st.markdown("### Khung thiết lập hạn mức")
                rl1, rl2, rl3 = responsive_cols(3)
                with rl1:
                    capital = st.number_input("Quy mô vốn Treasury (tỷ VND)", min_value=1000.0, value=20000.0, step=500.0)
                with rl2:
                    risk_appetite = st.slider("Risk appetite (% vốn)", 1, 10, 5) / 100
                with rl3:
                    stress_appetite = st.slider("Max stress loss (% vốn)", 1, 15, 7) / 100

                dv01_limit = capital * risk_appetite / 100
                stress_loss_limit = capital * stress_appetite
                duration_limit = dynamic_duration_limit(ir_res['delta'])

                st.markdown("### Hạn mức động được hệ thống tính toán")
                lm1, lm2, lm3 = responsive_cols(3)
                lm1.metric("DV01 limit", f"{dv01_limit:,.2f} tỷ / 1bp")
                lm2.metric("Stress loss limit", f"{stress_loss_limit:,.0f} tỷ")
                lm3.metric("Duration limit", f"{duration_limit:.1f} năm")

                st.markdown("### Mức sử dụng hạn mức hiện tại")
                util1, util2, util3 = responsive_cols(3)
                util1.metric("DV01 utilization", f"{(abs(dv01_total) / dv01_limit):.1%}" if dv01_limit else "N/A")
                util2.metric("Stress loss utilization", f"{(abs(pnl) / stress_loss_limit):.1%}" if stress_loss_limit else "N/A")
                util3.metric("Duration utilization", f"{(assumed_duration / duration_limit):.1%}" if duration_limit else "N/A")

                st.markdown("### Traffic light")
                t1, t2, t3 = responsive_cols(3)
                t1.metric("DV01 status", classify_risk_light_by_limit(dv01_total, dv01_limit), f"limit {dv01_limit:,.1f}")
                t2.metric("Stress loss status", classify_risk_light_by_limit(pnl, stress_loss_limit), f"limit {stress_loss_limit:,.0f}")
                t3.metric("Duration status", classify_risk_light_by_limit(assumed_duration, duration_limit), f"limit {duration_limit:,.1f}")

                st.markdown("### Cảnh báo tự động")
                alert_count = 0
                if abs(dv01_total) > dv01_limit:
                    st.error("DV01 đang vượt hạn mức. Cần giảm duration hoặc giảm quy mô rủi ro lãi suất.")
                    alert_count += 1
                elif abs(dv01_total) > dv01_limit * 0.7:
                    st.warning("DV01 đang tiến gần hạn mức. Cần theo dõi sát trước khi mở thêm vị thế.")

                if abs(pnl) > stress_loss_limit:
                    st.error("Stress loss đang vượt hạn mức. Cần hedge hoặc tái cơ cấu danh mục.")
                    alert_count += 1
                elif abs(pnl) > stress_loss_limit * 0.7:
                    st.warning("Stress loss đang tiến gần hạn mức. Nên hạn chế gia tăng rủi ro ở bucket dài.")

                if assumed_duration > duration_limit:
                    st.error("Duration đang vượt hạn mức động theo bối cảnh thị trường hiện tại.")
                    alert_count += 1
                elif assumed_duration > duration_limit * 0.7:
                    st.warning("Duration đang tiến gần hạn mức động. Cần thận trọng khi mở thêm vị thế dài.")

                if alert_count == 0:
                    st.success("Các chỉ tiêu rủi ro chính vẫn đang nằm trong vùng chấp nhận được theo hạn mức hiện tại.")

                st.markdown("### Ý nghĩa điều hành")
                st.write("- DV01 limit phản ánh mức độ nhạy cảm lãi suất tối đa có thể chấp nhận trên mỗi 1bp dịch chuyển lợi suất.")
                st.write("- Stress loss limit phản ánh ngưỡng tổn thất tối đa chấp nhận được trong kịch bản cú sốc lãi suất đã chọn.")
                st.write("- Duration limit được điều chỉnh động theo forecast lãi suất ON: khi lãi suất có xu hướng tăng, duration limit sẽ tự động thấp hơn.")
                st.write("- Tab này giúp chuyển phân tích forecast thành hệ thống kiểm soát rủi ro cụ thể cho ALM và Treasury.")

            with tabs[6]:
                st.subheader("Phân tích thị trường")
                tab_hint(9, 9)
                fx_latest, fx_mean, fx_z, fx_pct = regime_signal(df['usd_vnd'])
                ir_latest, ir_mean, ir_z, ir_pct = regime_signal(df['vibor_on'])
                a, b, c, d = responsive_cols(4)
                a.metric('FX vs mean', f'{fx_latest:,.0f}', f'z={fx_z:.2f}')
                b.metric('FX percentile', f'{fx_pct:.1f}%')
                c.metric('IR vs mean', f'{ir_latest:.2f}%', f'z={ir_z:.2f}')
                d.metric('IR percentile', f'{ir_pct:.1f}%')

                st.markdown("### Vị thế thị trường trong lịch sử")
                st.write("- Percentile cao cho thấy biến đang ở vùng lịch sử tương đối cao; rủi ro mean reversion tăng lên.")
                st.write("- Z-score giúp lượng hóa mức độ căng của thị trường.")
                st.write("- Khi kết hợp hướng forecast với regime, đánh giá thị trường sẽ thực tế hơn so với chỉ nhìn hướng tăng/giảm đơn thuần.")

                st.markdown("### Điều gì thay đổi so với lần chạy trước")
                ch1, ch2 = responsive_cols(2)
                ch1.metric("Thay đổi forecast FX", f"{fx_res['forecast']:,.0f}", f"{fx_res['forecast'] - prev_fx_forecast:+,.0f}" if not np.isnan(prev_fx_forecast) else "N/A")
                ch2.metric("Thay đổi forecast IR", f"{ir_res['forecast']:.2f}%", f"{ir_res['forecast'] - prev_ir_forecast:+.2f}" if not np.isnan(prev_ir_forecast) else "N/A")
                st.caption("So sánh với lần chạy trước được ước lượng bằng cách loại bỏ điểm dữ liệu mới nhất và chạy lại mô hình trên chuỗi lịch sử ngay trước đó.")

                st.markdown("### Phân tích kịch bản")
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
                    rows.append({'Kịch bản': name, 'Dự báo FX': fx_val, 'Dự báo IR': ir_val})
                scen = pd.DataFrame(rows)
                st.dataframe(scen, width="stretch", height=260 if mobile_mode else "auto")
                st.caption("Bảng kịch bản cho thấy mức độ nhạy của forecast khi các driver chính bị shock và hỗ trợ đánh giá biên độ rủi ro của nhận định cơ sở.")

            with tabs[7]:
                st.subheader("Khuyến nghị chiến lược")
                tab_hint(7, 9)

                fx_up = fx_res['delta'] > 80
                ir_up = ir_res['delta'] > 0.15
                fx_hot = fx_pct >= 90
                ir_hot = ir_pct >= 90

                st.markdown("### Định hướng trọng tâm")
                if ir_up and fx_up:
                    st.error("Kịch bản trung tâm hiện tại là lãi suất ON tăng cùng với áp lực tỷ giá đi lên. Đây là tổ hợp bất lợi cho cả cost of fund, trạng thái ngoại tệ và định giá danh mục đầu tư.")
                elif ir_up:
                    st.warning("Kịch bản trung tâm hiện tại là lãi suất ON tăng. Trọng tâm điều hành nên nghiêng về bảo vệ NIM, quản trị funding và rủi ro định giá danh mục.")
                elif fx_up:
                    st.warning("Kịch bản trung tâm hiện tại là áp lực tỷ giá tăng rõ hơn lãi suất. Trọng tâm điều hành nên nghiêng về quản trị trạng thái ngoại tệ và khách hàng nhạy cảm với ngoại tệ.")
                else:
                    st.success("Kịch bản trung tâm hiện tại chưa cho thấy áp lực tăng mạnh đồng thời ở cả lãi suất và tỷ giá. Có thêm dư địa để tối ưu cấu trúc bảng cân đối và hiệu quả danh mục.")

                m1, m2 = responsive_cols(2)
                with m1:
                    st.info("**Khuyến nghị cho thị trường 1**")
                    if ir_up:
                        st.write("- Chủ động rà soát repricing gap theo kỳ hạn, ưu tiên tăng tốc độ điều chỉnh lãi suất đầu ra ở các nhóm khách hàng có thể repricing sớm.")
                        st.write("- Điều chỉnh cơ cấu huy động theo hướng tăng tính ổn định của vốn, tránh phụ thuộc quá lớn vào nguồn vốn nhạy với lãi suất ngắn hạn.")
                    else:
                        st.write("- Có thể linh hoạt hơn trong định giá tín dụng và lựa chọn phân khúc tăng trưởng, nhưng vẫn cần duy trì kỷ luật margin theo ngành và theo kỳ hạn.")
                    if fx_up:
                        st.write("- Rà soát danh mục khách hàng nhập khẩu, khách hàng vay ngoại tệ hoặc doanh nghiệp có nghĩa vụ thanh toán ngoại tệ lớn trong 3–6 tháng tới.")
                        st.write("- Tăng cường khuyến nghị hedge cho nhóm khách hàng có dòng tiền ngoại tệ âm hoặc biên lợi nhuận mỏng.")
                    if fx_hot or ir_hot:
                        st.write("- Do các biến thị trường đang ở vùng cao trong lịch sử, nên thận trọng hơn với các giả định tăng trưởng và tránh mở rộng khẩu vị rủi ro quá nhanh.")

                with m2:
                    st.warning("**Khuyến nghị cho thị trường 2 / Treasury**")
                    if ir_up:
                        st.write("- Giảm khuynh hướng mở thêm duration dài; ưu tiên cấu trúc danh mục có khả năng phòng thủ tốt hơn trước cú sốc lợi suất.")
                        st.write("- Thiết lập hoặc siết chặt hạn mức DV01 và stress loss cho các bucket dài, đặc biệt nếu danh mục đang tập trung nhiều vào 3–5 năm và 5Y+.")
                    else:
                        st.write("- Có thể cân nhắc mở duration chọn lọc ở các điểm lợi suất hấp dẫn, nhưng nên gắn với hạn mức DV01 rõ ràng.")
                    if fx_up:
                        st.write("- Giữ trạng thái ngoại tệ thận trọng hơn; ưu tiên hedge ngắn hạn hơn là mở vị thế directional lớn.")
                        st.write("- Theo dõi sát độ lệch giữa tín hiệu forecast và regime để tránh mua đuổi khi tỷ giá đã ở vùng quá cao.")
                    if not ir_up and not fx_up:
                        st.write("- Trọng tâm có thể chuyển sang tối ưu carry, lựa chọn điểm vào danh mục và cải thiện hiệu quả sử dụng vốn VND.")

                st.markdown("### Các hành động ưu tiên trong 30 ngày tới")
                actions = []
                if ir_up:
                    actions.append("Rà soát lại hạn mức DV01, stress loss và phân bổ duration bucket của danh mục đầu tư.")
                    actions.append("Cập nhật lại giả định cost of fund trong kế hoạch NIM và kế hoạch kinh doanh quý tới.")
                if fx_up:
                    actions.append("Đánh giá lại mức độ nhạy cảm của danh mục khách hàng ngoại tệ và cơ chế hedge hiện hành.")
                if fx_hot or ir_hot:
                    actions.append("Bổ sung kịch bản đảo chiều ngắn hạn vào quy trình ra quyết định vị thế do thị trường đang ở vùng lịch sử cao.")
                if not actions:
                    actions.append("Tiếp tục theo dõi các driver chính và duy trì kỷ luật hạn mức danh mục, sẵn sàng tăng trạng thái khi risk-reward thuận lợi hơn.")
                for act in actions:
                    st.write(f"- {act}")

            with tabs[7]:
                st.subheader("Tổng kết")
                tab_hint(8, 9)

                fx_text = top_driver_text(fx_res['contrib'], 'tỷ giá')
                ir_text = top_driver_text(ir_res['contrib'], 'lãi suất')

                st.markdown("### 1. Tóm tắt dự báo")
                c1, c2 = responsive_cols(2)
                c1.metric(
                    "Tỷ giá (USD/VND)",
                    f"{fx_res['forecast']:,.0f}",
                    f"{fx_res['delta']:+,.0f}"
                )
                c2.metric(
                    "Lãi suất ON",
                    f"{ir_res['forecast']:.2f}%",
                    f"{ir_res['delta']:+.2f}đ"
                )

                st.markdown("### 2. Động lực chính")
                st.write(f"- Tỷ giá: {fx_text}")
                st.write(f"- Lãi suất: {ir_text}")

                st.markdown("### 2A. Độ tin cậy và thay đổi so với lần chạy trước")
                st.write(f"- Biên độ dự báo FX tham chiếu: {fx_low:,.0f} đến {fx_high:,.0f}; mô hình có RMSE tốt nhất hiện tại: {fx_model_name}.")
                st.write(f"- Biên độ dự báo IR tham chiếu: {ir_low:.2f}% đến {ir_high:.2f}%; mô hình có RMSE tốt nhất hiện tại: {ir_model_name}.")
                if not np.isnan(prev_fx_forecast):
                    st.write(f"- Forecast FX thay đổi {fx_res['forecast'] - prev_fx_forecast:+,.0f} so với lần chạy trước.")
                if not np.isnan(prev_ir_forecast):
                    st.write(f"- Forecast IR thay đổi {ir_res['forecast'] - prev_ir_forecast:+.2f} điểm so với lần chạy trước.")

                st.markdown("### 3. Đánh giá bối cảnh thị trường")
                fx_latest, fx_mean, fx_z, fx_pct = regime_signal(df['usd_vnd'])
                ir_latest, ir_mean, ir_z, ir_pct = regime_signal(df['vibor_on'])

                if fx_pct >= 90 or ir_pct >= 90:
                    st.warning("Các biến thị trường đang ở vùng cao trong lịch sử, do đó cần tính đến rủi ro đảo chiều ngắn hạn khi sử dụng forecast cho quyết định vị thế.")
                elif fx_pct <= 25 or ir_pct <= 25:
                    st.info("Một số biến thị trường đang ở vùng thấp so với lịch sử, cho thấy khả năng xuất hiện nhịp hồi hoặc điều chỉnh theo hướng ngược lại trong giai đoạn tới.")
                else:
                    st.success("Các biến thị trường đang ở vùng trung tính hơn so với lịch sử, giúp tín hiệu forecast thuận lợi hơn cho việc sử dụng làm cơ sở định hướng điều hành.")

                st.markdown("### 4. Rủi ro chính")
                if fx_res['delta'] > 80:
                    st.write("- Áp lực tỷ giá tăng có thể làm gia tăng rủi ro đối với khách hàng nhập khẩu, khách hàng vay ngoại tệ và trạng thái ngoại tệ của ngân hàng.")
                if ir_res['delta'] > 0.15:
                    st.write("- Lãi suất ON tăng có thể gây áp lực lên cost of fund, NIM và định giá danh mục đầu tư lãi suất cố định.")
                if abs(ir_res['delta']) <= 0.15 and abs(fx_res['delta']) <= 80:
                    st.write("- Chưa xuất hiện rủi ro nổi trội ở mức cao trong ngắn hạn, tuy nhiên vẫn cần tiếp tục theo dõi sát các driver chính của forecast.")

                st.markdown("### 5. Khuyến nghị hành động")
                if ir_res['delta'] > 0.15:
                    st.write("- Kiểm soát duration danh mục đầu tư ở mức chặt chẽ hơn, hạn chế mở thêm vị thế dài nếu không có biện pháp hedge phù hợp.")
                    st.write("- Rà soát lại giả định cost of fund, repricing gap và kế hoạch bảo vệ NIM trong các kỳ điều hành tiếp theo.")
                if fx_res['delta'] > 80:
                    st.write("- Tăng cường quản trị trạng thái ngoại tệ, ưu tiên hedge ngắn hạn và rà soát nhóm khách hàng có độ nhạy cao với tỷ giá.")
                if ir_res['delta'] <= 0.15 and fx_res['delta'] <= 80:
                    st.write("- Có thể tiếp tục tối ưu carry và cơ cấu danh mục ở mức thận trọng, đồng thời duy trì kỷ luật hạn mức rủi ro.")

                st.markdown("### 6. Kết luận điều hành")
                if ir_res['delta'] > 0.15 and fx_res['delta'] > 80:
                    st.error("Bối cảnh điều hành hiện tại nghiêng về phòng thủ: cần ưu tiên kiểm soát thanh khoản, hạn mức rủi ro và trạng thái ngoại tệ.")
                elif ir_res['delta'] > 0.15 or fx_res['delta'] > 80:
                    st.warning("Bối cảnh điều hành có áp lực tăng ở một số biến trọng yếu, do đó cần bám sát diễn biến thị trường và điều chỉnh chiến lược theo hướng thận trọng.")
                else:
                    st.success("Bối cảnh điều hành tương đối ổn định, cho phép tập trung hơn vào tối ưu hiệu quả danh mục và tăng trưởng có chọn lọc.")

else:
    st.info("Không có dữ liệu đầu vào. Có thể chọn chế độ tự động từ Google Sheets hoặc upload file Excel thủ công.")
