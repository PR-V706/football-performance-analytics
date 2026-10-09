
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix

st.set_page_config(
    page_title="Football Performance Analytics",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ Football Performance Analytics")
st.subheader("Match Outcome Prediction Dashboard")

st.write(
    "Analyze football team performance and predict "
    "whether the home team wins, the match draws, "
    "or the away team wins."
)

# 1. Generate demonstration match data
@st.cache_data
def generate_data():
    np.random.seed(42)
    n = 600

    home_attack = np.random.randint(1, 11, n)
    away_attack = np.random.randint(1, 11, n)
    home_defense = np.random.randint(1, 11, n)
    away_defense = np.random.randint(1, 11, n)
    home_form = np.random.randint(1, 11, n)
    away_form = np.random.randint(1, 11, n)

    home_goals = np.random.poisson(
        np.maximum(
            0.5,
            1.4 + 0.13 * (home_attack - 5)
            - 0.08 * (away_defense - 5)
            + 0.10 * (home_form - 5)
        )
    )

    away_goals = np.random.poisson(
        np.maximum(
            0.5,
            1.1 + 0.13 * (away_attack - 5)
            - 0.08 * (home_defense - 5)
            + 0.10 * (away_form - 5)
        )
    )

    result = np.where(
        home_goals > away_goals, "Home Win",
        np.where(home_goals == away_goals, "Draw", "Away Win")
    )

    return pd.DataFrame({
        "Home_Attack": home_attack,
        "Away_Attack": away_attack,
        "Home_Defense": home_defense,
        "Away_Defense": away_defense,
        "Home_Form": home_form,
        "Away_Form": away_form,
        "Home_Goals": home_goals,
        "Away_Goals": away_goals,
        "Result": result
    })


df = generate_data()

# 2. Sidebar filters
st.sidebar.header("Dashboard Filters")

minimum_attack = st.sidebar.slider(
    "Minimum home attack rating", 1, 10, 1
)

filtered_df = df[
    df["Home_Attack"] >= minimum_attack
]

# 3. Key performance indicators
st.header("Match Statistics")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Matches", len(filtered_df))

home_win_pct = (
    filtered_df["Result"].eq("Home Win").mean() * 100
    if len(filtered_df) else 0
)
draw_pct = (
    filtered_df["Result"].eq("Draw").mean() * 100
    if len(filtered_df) else 0
)
away_win_pct = (
    filtered_df["Result"].eq("Away Win").mean() * 100
    if len(filtered_df) else 0
)

c2.metric("Home Wins", f"{home_win_pct:.1f}%")
c3.metric("Draws", f"{draw_pct:.1f}%")
c4.metric("Away Wins", f"{away_win_pct:.1f}%")

# 4. Charts
st.header("Performance Visualization")

left, right = st.columns(2)

with left:
    st.subheader("Match Outcome Distribution")
    counts = filtered_df["Result"].value_counts()

    fig, ax = plt.subplots()
    counts.reindex(
        ["Home Win", "Draw", "Away Win"], fill_value=0
    ).plot(kind="bar", ax=ax)
    ax.set_xlabel("Match Outcome")
    ax.set_ylabel("Number of Matches")
    ax.tick_params(axis="x", rotation=0)
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.subheader("Team Attack Comparison")

    fig, ax = plt.subplots()
    ax.hist(
        filtered_df["Home_Attack"],
        bins=np.arange(1, 12) - 0.5,
        alpha=0.7,
        label="Home Team"
    )
    ax.hist(
        filtered_df["Away_Attack"],
        bins=np.arange(1, 12) - 0.5,
        alpha=0.7,
        label="Away Team"
    )
    ax.set_xlabel("Attack Rating")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

st.subheader("Sample Match Data")
st.dataframe(filtered_df.head(20), use_container_width=True)

st.download_button(
    "Download Demonstration Data as CSV",
    data=filtered_df.to_csv(index=False).encode("utf-8"),
    file_name="football_match_data.csv",
    mime="text/csv"
)

# 5. Train machine-learning model
st.header("Machine Learning: Match Outcome Prediction")

features = [
    "Home_Attack",
    "Away_Attack",
    "Home_Defense",
    "Away_Defense",
    "Home_Form",
    "Away_Form"
]

X = df[features]
y = df["Result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

st.metric("Model Test Accuracy", f"{accuracy * 100:.1f}%")

st.caption(
    "This accuracy uses synthetic demonstration data "
    "and is not evidence of real-world football prediction accuracy."
)

# Confusion matrix
st.subheader("Confusion Matrix")

labels = ["Home Win", "Draw", "Away Win"]
cm = confusion_matrix(y_test, predictions, labels=labels)

fig, ax = plt.subplots()
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
    ax=ax
)
ax.set_xlabel("Predicted Outcome")
ax.set_ylabel("Actual Outcome")
st.pyplot(fig)
plt.close(fig)

# 6. Interactive prediction
st.header("Predict a New Match")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Home Team")
    h_attack = st.slider("Home attack", 1, 10, 7)
    h_defense = st.slider("Home defense", 1, 10, 7)
    h_form = st.slider("Home recent form", 1, 10, 7)

with col2:
    st.subheader("Away Team")
    a_attack = st.slider("Away attack", 1, 10, 6)
    a_defense = st.slider("Away defense", 1, 10, 6)
    a_form = st.slider("Away recent form", 1, 10, 6)

if st.button("Predict Match Outcome", type="primary"):
    new_match = pd.DataFrame([{
        "Home_Attack": h_attack,
        "Away_Attack": a_attack,
        "Home_Defense": h_defense,
        "Away_Defense": a_defense,
        "Home_Form": h_form,
        "Away_Form": a_form
    }])

    result = model.predict(new_match)[0]
    probabilities = model.predict_proba(new_match)[0]

    st.success(f"Predicted Outcome: {result}")

    probability_df = pd.DataFrame({
        "Outcome": model.classes_,
        "Probability (%)": probabilities * 100
    }).sort_values("Probability (%)", ascending=False)

    st.dataframe(probability_df, use_container_width=True)

st.info(
    "Project note: The built-in dataset is synthetic. "
    "For meaningful real-match analysis, use verified "
    "historical football match data."
)