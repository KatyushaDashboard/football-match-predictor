import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.stats import poisson
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Football Match Predictor Dashboard",
    page_icon="⚽",
    layout="wide"
)

# --- LOAD MODELS & DATA ---
@st.cache_resource
def load_assets():
    model_p_home = joblib.load('poisson_home.joblib')
    model_p_away = joblib.load('poisson_away.joblib')
    model_xgb = joblib.load('xgboost_1x2.joblib')
    le = joblib.load('label_encoder.joblib')
    df_stats = pd.read_csv('latest_team_stats.csv')
    return model_p_home, model_p_away, model_xgb, le, df_stats

try:
    model_p_home, model_p_away, model_xgb, le, df_stats = load_assets()
except Exception as e:
    st.error(f"Gagal memuat file model/data. Pastikan seluruh file .joblib dan .csv sudah di-upload. Error: {e}")
    st.stop()

# --- HELPER FUNCTIONS ---
def calculate_poisson_probs(lh, la, max_goals=6):
    prob_matrix = np.zeros((max_goals + 1, max_goals + 1))
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            prob_matrix[h, a] = poisson.pmf(h, lh) * poisson.pmf(a, la)
            
    over_25 = prob_matrix[np.triu_indices(max_goals + 1, k=1)].sum() # Rough approx, exact below
    over_25 = sum(prob_matrix[h, a] for h in range(max_goals + 1) for a in range(max_goals + 1) if h + a > 2.5)
    btts_yes = prob_matrix[1:, 1:].sum()
    
    return prob_matrix, over_25, btts_yes

def calculate_ah_prob(lh, la, ah_line, max_goals=6):
    prob_cover = 0.0
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            prob = poisson.pmf(h, lh) * poisson.pmf(a, la)
            if (h - a) + ah_line > 0:
                prob_cover += prob
            elif (h - a) + ah_line == 0:
                prob_cover += prob * 0.5
    return prob_cover

# --- HEADER & SIDEBAR ---
st.title("⚽ Dashboard Prediksi Analisis Sepak Bola")
st.markdown("Analisis Prediksi **Moneyline (1X2)**, **Total Goal (O/U 2.5)**, **BTTS**, dan **Asian Handicap** berbasis Machine Learning & Poisson Model.")

st.sidebar.header("⚙️ Pengaturan Pertandingan")

leagues = df_stats['League'].unique()
selected_league = st.sidebar.selectbox("Pilih Liga", leagues)

teams = df_stats[df_stats['League'] == selected_league]['HomeTeam'].unique()
home_team = st.sidebar.selectbox("Tuan Rumah (Home)", teams)
away_team = st.sidebar.selectbox("Tamu (Away)", [t for t in teams if t != home_team])

st.sidebar.subheader("📊 Odds Bandar (Untuk Cek +EV)")
odds_home = st.sidebar.number_input("Odds Home Win", value=2.10, step=0.05)
odds_draw = st.sidebar.number_input("Odds Draw", value=3.30, step=0.05)
odds_away = st.sidebar.number_input("Odds Away Win", value=3.50, step=0.05)

# --- FEATURE EXTRACTION FOR MATCH ---
h_data = df_stats[df_stats['HomeTeam'] == home_team].iloc[0]
a_data = df_stats[df_stats['HomeTeam'] == away_team].iloc[0]

features_poisson_df = pd.DataFrame([{
    'Home_Elo': h_data['Home_Elo'],
    'Away_Elo': a_data['Home_Elo'], # Mengambil current Elo
    'Elo_Diff': h_data['Home_Elo'] - a_data['Home_Elo'],
    'Home_Roll5_GF': h_data['Home_Roll5_GF'],
    'Away_Roll5_GF': a_data['Home_Roll5_GF'],
    'Home_Roll5_GA': h_data['Home_Roll5_GA'],
    'Away_Roll5_GA': a_data['Home_Roll5_GA'],
    'Home_Roll5_SoT': h_data['Home_Roll5_SoT'],
    'Away_Roll5_SoT': a_data['Home_Roll5_SoT'],
    'Home_Rest_Days': h_data['Home_Rest_Days'],
    'Away_Rest_Days': a_data['Home_Rest_Days']
}])

features_xgb_df = features_poisson_df.copy()
features_xgb_df['B365H'] = odds_home
features_xgb_df['B365D'] = odds_draw
features_xgb_df['B365A'] = odds_away

# --- MODEL PREDICTIONS ---
exp_home_goals = model_p_home.predict(features_poisson_df)[0]
exp_away_goals = model_p_away.predict(features_poisson_df)[0]

prob_matrix, prob_over25, prob_btts = calculate_poisson_probs(exp_home_goals, exp_away_goals)

probs_1x2 = model_xgb.predict_proba(features_xgb_df)[0]

# Mapping otomatis berdasarkan nama class model
class_mapping = dict(zip(model_xgb.classes_, probs_1x2))

prob_h = class_mapping['H']
prob_d = class_mapping['D']
prob_a = class_mapping['A']

# --- MAIN DASHBOARD VIEW ---
st.subheader(f"📌 {home_team} vs {away_team}")
col_info1, col_info2, col_info3 = st.columns(3)

with col_info1:
    st.metric("Elo Rating Home", f"{h_data['Home_Elo']:.0f}")
    st.metric("Rata-rata Gol Home (5 Match)", f"{h_data['Home_Roll5_GF']:.2f}")

with col_info2:
    st.metric("Elo Rating Away", f"{a_data['Home_Elo']:.0f}")
    st.metric("Rata-rata Gol Away (5 Match)", f"{a_data['Home_Roll5_GF']:.2f}")

with col_info3:
    st.metric("Ekspektasi Gol Home (xG)", f"{exp_home_goals:.2f}")
    st.metric("Ekspektasi Gol Away (xG)", f"{exp_away_goals:.2f}")

st.divider()

# --- MARKET ANALYSIS TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Moneyline 1X2", "⚽ Total Goal & BTTS", "🛡️ Asian Handicap", "💰 +EV Value Bet"])

with tab1:
    st.write("### Prediksi Probabilitas Hasil Akhir (1X2)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Home Win", f"{prob_h * 100:.1f}%", help=f"Implied Odds: {1/prob_h:.2f}")
    c2.metric("Draw", f"{prob_d * 100:.1f}%", help=f"Implied Odds: {1/prob_d:.2f}")
    c3.metric("Away Win", f"{prob_a * 100:.1f}%", help=f"Implied Odds: {1/prob_a:.2f}")
    
    # Chart
    fig = go.Figure(go.Bar(
        x=[home_team, 'Draw', away_team],
        y=[prob_h * 100, prob_d * 100, prob_a * 100],
        marker_color=['#2ca02c', '#ff7f0e', '#d62728']
    ))
    fig.update_layout(title="Probabilitas Hasil Pertandingan (%)", yaxis_range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.write("### Prediksi Total Gol & Both Teams To Score")
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.write("#### Total Goal (Over/Under 2.5)")
        st.metric("Over 2.5 Goal", f"{prob_over25 * 100:.1f}%")
        st.metric("Under 2.5 Goal", f"{(1 - prob_over25) * 100:.1f}%")
        if prob_over25 > 0.55:
            st.success("💡 Rekomendasi: **OVER 2.5 GOAL**")
        elif prob_over25 < 0.45:
            st.info("💡 Rekomendasi: **UNDER 2.5 GOAL**")
        else:
            st.warning("⚠️ Laga Berimbang (Netral)")

    with col_g2:
        st.write("#### Both Teams To Score (BTTS)")
        st.metric("BTTS - YES", f"{prob_btts * 100:.1f}%")
        st.metric("BTTS - NO", f"{(1 - prob_btts) * 100:.1f}%")
        if prob_btts > 0.55:
            st.success("💡 Rekomendasi: **BTTS YES**")
        else:
            st.info("💡 Rekomendasi: **BTTS NO**")

with tab3:
    st.write("### Prediksi Asian Handicap Standar")
    ah_lines = [-1.5, -0.5, 0.5]
    for line in ah_lines:
        prob_cover = calculate_ah_prob(exp_home_goals, exp_away_goals, line)
        line_str = f"+{line}" if line > 0 else f"{line}"
        st.write(f"**Home Handicap ({line_str}):** Probabilitas Cover = **{prob_cover * 100:.1f}%**")
        st.progress(float(prob_cover))

with tab4:
    st.write("### Analisis Nilai Taruhan (+EV Value Bet)")
    
    ev_h = (prob_h * odds_home) - 1
    ev_d = (prob_d * odds_draw) - 1
    ev_a = (prob_a * odds_away) - 1
    
    ev_df = pd.DataFrame({
        'Pilihan': [f"Home ({home_team})", "Draw", f"Away ({away_team})"],
        'Probabilitas Model': [f"{prob_h*100:.1f}%", f"{prob_d*100:.1f}%", f"{prob_a*100:.1f}%"],
        'Odds Bandar': [odds_home, odds_draw, odds_away],
        'Expected Value (+EV)': [f"{ev_h*100:+.2f}%", f"{ev_d*100:+.2f}%", f"{ev_a*100:+.2f}%"],
        'Sinyal Value': [
            "✅ VALUE BET" if ev_h > 0.05 else "❌ NO VALUE",
            "✅ VALUE BET" if ev_d > 0.05 else "❌ NO VALUE",
            "✅ VALUE BET" if ev_a > 0.05 else "❌ NO VALUE"
        ]
    })
    st.table(ev_df)
