import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load

# Step 1: Load and Prepare the Dataset
df = pd.read_csv('season-1819.csv', usecols=['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HS', 'AS', 'HST', 'AST'])

# Step 2: Feature Engineering
# Calculate the average shots and shots on target for home and away teams
df['AvgHS'] = df.groupby('HomeTeam')['HS'].transform('mean')
df['AvgAS'] = df.groupby('AwayTeam')['AS'].transform('mean')
df['AvgHST'] = df.groupby('HomeTeam')['HST'].transform('mean')
df['AvgAST'] = df.groupby('AwayTeam')['AST'].transform('mean')

# Step 3: Choose and Train Models
X = df[['AvgHS', 'AvgAS', 'AvgHST', 'AvgAST']]  # Selected features
y_home_goals = df['FTHG']  # Home team goals prediction
y_away_goals = df['FTAG']  # Away team goals prediction

# Train Random Forest models
model_home_goals = RandomForestRegressor()
model_away_goals = RandomForestRegressor()
model_home_goals.fit(X, y_home_goals)
model_away_goals.fit(X, y_away_goals)

# Save trained models
dump(model_home_goals, 'random_forest_model_home_goals.joblib')
dump(model_away_goals, 'random_forest_model_away_goals.joblib')


st.title("Football Match Predictor")

option = st.selectbox("Choose an option:", ["Specific Game", "Whole Season"])

if option == "Specific Game":
    home_team = st.text_input("Enter the name of the home team:")
    away_team = st.text_input("Enter the name of the away team:")

    if st.button("Predict"):

                if home_team not in df['HomeTeam'].values or away_team not in df['AwayTeam'].values:
                    print("Invalid team names. Please try again.")
                else:
                    # Prepare input for prediction
                    home_team_data = df[df['HomeTeam'] == home_team].tail(1)
                    away_team_data = df[df['AwayTeam'] == away_team].tail(1)

                    new_data = pd.DataFrame({'AvgHS': home_team_data['AvgHS'].values,
                                             'AvgAS': away_team_data['AvgAS'].values,
                                             'AvgHST': home_team_data['AvgHST'].values,
                                             'AvgAST': away_team_data['AvgAST'].values})

                    # Make predictions
                    home_goals = round(model_home_goals.predict(new_data)[0])
                    away_goals = round(model_away_goals.predict(new_data)[0])

                    # Step 5: Display Prediction Result
                    print("Prediction:")
                    st.write(f"Prediction: {home_team} {home_goals} : {away_goals} {away_team}")

elif option == "Whole Season":
    selected_team = st.text_input("Enter the name of the selected team:")

    if st.button("Predict Season"):
            # Function to predict the result of a match
            def predict_match_result(home_team, away_team):
                home_team_data = df[df['HomeTeam'] == home_team].tail(1)
                away_team_data = df[df['AwayTeam'] == away_team].tail(1)

                new_data = pd.DataFrame({'AvgHS': home_team_data['AvgHS'].values,
                                         'AvgAS': away_team_data['AvgAS'].values,
                                         'AvgHST': home_team_data['AvgHST'].values,
                                         'AvgAST': away_team_data['AvgAST'].values})

                home_goals = round(model_home_goals.predict(new_data)[0])
                away_goals = round(model_away_goals.predict(new_data)[0])

                if home_goals > away_goals:
                    return home_team
                elif home_goals < away_goals:
                    return away_team
                else:
                    return 'DRAW'

            # Get the selected team and the list of all teams
            all_teams = df['HomeTeam'].unique()

            # Calculate the result of every possible game for the selected team

            for opponent_team in all_teams:
                if opponent_team != selected_team:
                    result = predict_match_result(selected_team, opponent_team)
                    st.write(f"{selected_team} vs {opponent_team}: {result}")
                    result2 = predict_match_result(opponent_team, selected_team)
                    st.write(f"{opponent_team} vs {selected_team}: {result2}")


