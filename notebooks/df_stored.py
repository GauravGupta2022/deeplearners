import pandas as pd

# Raw data
country_df = pd.read_csv('../data/raw/Country.csv')
league_df = pd.read_csv('../data/raw/League.csv')
match_df = pd.read_csv('../data/raw/Match.csv')
player_attr_df = pd.read_csv('../data/raw/Player_Attributes.csv')
player_df = pd.read_csv('../data/raw/Player.csv')
sqlite_df = pd.read_csv('../data/raw/sqlite_sequence.csv')
team_attr_df = pd.read_csv('../data/raw/Team_Attributes.csv')
team_df = pd.read_csv('../data/raw/Team.csv')

# Cleaned data
player_cleaned_df = pd.read_csv('../data/cleaned/player_attributes_cleaned.csv')
