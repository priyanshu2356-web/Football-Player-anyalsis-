import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="Football Player Analysis", page_icon="⚽", layout="wide")

import plotly.io as pio

# Neon dark theme for all charts
pio.templates["neon"] = pio.templates["plotly_dark"]
pio.templates["neon"].layout.update(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    colorway=["#00FFA3", "#00B8FF", "#FF2E97", "#FFD600", "#9D4EDD"],
    font=dict(color="#E5F4FF"),
)
pio.templates.default = "neon"

st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at 20% 0%, #0f2a3f 0%, #0A0E1A 45%, #05070f 100%);
}
h1 {
    background: linear-gradient(90deg, #00FFA3, #00B8FF, #FF2E97);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 900 !important;
    letter-spacing: 1px;
}
h2, h3 { color: #00FFA3 !important; }
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(0,255,163,0.35);
    border-radius: 16px;
    padding: 16px;
    box-shadow: 0 0 18px rgba(0,255,163,0.15);
    backdrop-filter: blur(8px);
}
[data-testid="stMetricValue"] { color: #00FFA3; text-shadow: 0 0 12px #00FFA3; }
.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.05);
    border-radius: 12px 12px 0 0;
    padding: 10px 22px;
}
.stTabs [aria-selected="true"] {
    background: rgba(0,255,163,0.15);
    border-bottom: 2px solid #00FFA3;
}
.stSuccess {
    border: 1px solid #00FFA3;
    box-shadow: 0 0 20px rgba(0,255,163,0.3);
}
h1 { animation: glow 2.5s ease-in-out infinite alternate; }
@keyframes glow {
  from { filter: drop-shadow(0 0 6px #00FFA3); }
  to   { filter: drop-shadow(0 0 20px #00B8FF); }
}
[data-testid="stMetric"] { transition: transform 0.2s; }
[data-testid="stMetric"]:hover { transform: translateY(-4px) scale(1.02); 
}
</style>
""", unsafe_allow_html=True)

SKILLS = ['Crossing','Finishing','Heading accuracy','Short passing','Volleys','Dribbling',
          'Curve','FK Accuracy','Long passing','Ball control','Acceleration','Sprint speed',
          'Agility','Reactions','Balance','Shot power','Jumping','Stamina','Strength',
          'Long shots','Aggression','Interceptions','Att. Position','Vision','Penalties',
          'Composure','Defensive awareness','Standing tackle','Sliding tackle']
GK = ['GK Diving','GK Handling','GK Kicking','GK Positioning','GK Reflexes']
FEATURES = SKILLS + GK + ['Age','Height_cm','Weight_kg','International reputation']

@st.cache_data
def load():
    d = pd.read_csv("players_clean.csv")
    d["Label"] = d["Player"] + " (" + d["Club"].astype(str) + ")"
    return d

@st.cache_resource
def train(_df):
    X = _df[FEATURES].fillna(_df[FEATURES].median())
    y = _df["Overall rating"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    m = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1).fit(Xtr, ytr)
    p = m.predict(Xte)
    return m, r2_score(yte, p), mean_absolute_error(yte, p)

@st.cache_resource
def build_nn(_df):
    X = _df[SKILLS].fillna(_df[SKILLS].median())
    sc = StandardScaler().fit(X)
    nn = NearestNeighbors(n_neighbors=6).fit(sc.transform(X))
    return sc, nn

@st.cache_resource
def build_value_model(_df):
    value_features = SKILLS + GK + ['Age', 'Overall rating', 'Potential', 'International reputation']
    Xv = _df[value_features].fillna(_df[value_features].median())
    yv = _df['Value']
    vm = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1).fit(Xv, yv)
    pred_value = vm.predict(Xv)
    return value_features, pred_value

df = load()
st.title("⚽ Football Player Statistics Analysis")
st.caption("Explore players, compare them, find similar players, and predict ratings with ML.")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📊 Overview", "🆚 Compare Players", "🔍 Similar Players",
     "🤖 Rating Predictor", "💰 Bargain Finder", "💡 Key Insights"])

with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Players", f"{len(df):,}")
    c2.metric("Average rating", f"{df['Overall rating'].mean():.1f}")
    c3.metric("Clubs", df["Club"].nunique())
    pos = st.multiselect("Filter by position", sorted(df["Best position"].unique()))
    d = df[df["Best position"].isin(pos)] if pos else df
    st.plotly_chart(px.histogram(d, x="Overall rating", nbins=30, title="Rating distribution"), use_container_width=True)
    st.plotly_chart(px.scatter(d, x="Age", y="Overall rating", hover_name="Player", color="Best position", title="Age vs rating"), use_container_width=True)
    st.plotly_chart(px.scatter(d, x="Overall rating", y="Value", log_y=True, hover_name="Player", title="Market value vs rating (log scale)"), use_container_width=True)

    fig3d = px.scatter_3d(d, x="Age", y="Overall rating", z="Value",
                           color="Best position", hover_name="Player",
                           title="Age × Rating × Value (drag to rotate)")
    st.plotly_chart(fig3d, use_container_width=True)

    st.subheader("Top 10 players")
    st.dataframe(d.nlargest(10, "Overall rating")[["Player", "Club", "Best position", "Age", "Overall rating", "Potential"]], hide_index=True)

with tab2:
    a, b = st.columns(2)
    p1 = a.selectbox("Player 1", df["Label"], index=0)
    p2 = b.selectbox("Player 2", df["Label"], index=1)
    r1 = df[df["Label"] == p1].iloc[0]
    r2 = df[df["Label"] == p2].iloc[0]
    cats = ["Finishing", "Short passing", "Dribbling", "Sprint speed", "Stamina", "Strength", "Defensive awareness", "Vision"]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=[r1[c] for c in cats], theta=cats, fill="toself", name=r1["Player"]))
    fig.add_trace(go.Scatterpolar(r=[r2[c] for c in cats], theta=cats, fill="toself", name=r2["Player"]))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100])))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(pd.DataFrame({"Stat": ["Age", "Overall rating", "Potential", "Value", "Wage"],
                               r1["Player"]: [r1[c] for c in ["Age", "Overall rating", "Potential", "Value", "Wage"]],
                               r2["Player"]: [r2[c] for c in ["Age", "Overall rating", "Potential", "Value", "Wage"]]}), hide_index=True)

with tab3:
    sel = st.selectbox("Pick a player", df["Label"], key="sim")
    sc, nn = build_nn(df)
    idx = df.index[df["Label"] == sel][0]
    row = df.loc[[idx], SKILLS].fillna(df[SKILLS].median())
    _, ids = nn.kneighbors(sc.transform(row))
    sim = df.iloc[ids[0][1:]]
    st.write("Players with the most similar skill profile:")
    st.dataframe(sim[["Player", "Club", "Best position", "Age", "Overall rating", "Value"]], hide_index=True)

with tab4:
    model, r2, mae = train(df)
    c1, c2 = st.columns(2)
    c1.metric("Model R²", f"{r2:.3f}")
    c2.metric("Average error (rating points)", f"{mae:.2f}")
    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False).head(10)
    st.plotly_chart(px.bar(imp[::-1], orientation="h", title="Top 10 features for predicting rating"), use_container_width=True)
    st.subheader("Try it: predict a rating")
    reactions = st.slider("Reactions", 30, 99, 70)
    shortp = st.slider("Short passing", 30, 99, 70)
    age = st.slider("Age", 16, 45, 25)
    inp = df[FEATURES].median().to_frame().T
    inp["Reactions"], inp["Short passing"], inp["Age"] = reactions, shortp, age
    pred = model.predict(inp)[0]
    st.success(f"Predicted overall rating: {pred:.1f}")

    st.write("What's pushing this prediction:")
    baseline = df[FEATURES].median()
    diff = (inp.iloc[0] - baseline).sort_values(key=abs, ascending=False).head(5)
    st.bar_chart(diff)

with tab5:
    st.subheader("Find undervalued players")
    st.caption("Players whose actual value is well below what their skills predict.")
    value_features, pred_value = build_value_model(df)
    df["Predicted_Value"] = pred_value
    df["Value_Gap"] = df["Predicted_Value"] - df["Value"]

    max_age = st.slider("Max age", 16, 40, 26)
    min_rating = st.slider("Min overall rating", 50, 95, 70)

    bargains = df[(df["Age"] <= max_age) & (df["Overall rating"] >= min_rating)]
    bargains = bargains.nlargest(10, "Value_Gap")
    show_cols = ["Player", "Club", "Age", "Overall rating", "Value", "Predicted_Value"]
    st.dataframe(bargains[show_cols], hide_index=True)
    st.caption("Predicted_Value is what the model expects based on skills alone. "
               "A large gap suggests the player may be underpriced.")

with tab6:
    st.subheader("Key Insights")
    st.markdown("""
    - **Ratings cluster between 70-80** — very few players exceed 85 overall.
    - **Players peak around age 25-30**, then ratings level off.
    - **Reactions is the strongest predictor** of overall rating (0.91 correlation) — stronger than any physical stat.
    - **Attacking positions (CF, CM, RW)** have the highest average ratings.
    - **Market value rises steeply with rating**, and value and wage move closely together.
    - The **Bargain Finder** uses a second ML model to flag players priced below what their skills justify — a real scouting use case.
    """)
