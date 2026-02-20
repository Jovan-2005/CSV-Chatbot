import pandas as pd
import numpy as np
import ast
import os

print("Starting data cleaning...")

# Load the datasets
game_data = pd.read_csv('data/game_data.csv')
additional_data = pd.read_csv('data/additional_data.csv')

print(f"Game Data shape: {game_data.shape}")
print(f"Additional Data shape: {additional_data.shape}")

# Merging
df = pd.merge(game_data, additional_data, left_on='steam_appid', right_on='appid', suffixes=('_main', '_extra'))
print(f"Merged shape: {df.shape}")

# Filter relevant columns
# price is unique to additional_data, name is in both
cols_to_keep = [
    'name_main', 'steam_appid', 'is_free', 'supported_languages', 'developers', 
    'publishers', 'categories', 'genres', 'release_date', 
    'positive', 'negative', 'price', 'tags', 'owners', 'ccu'
]
df = df[cols_to_keep].rename(columns={'name_main': 'name', 'price': 'price_usd'})

# Rating calculation
df['total_ratings'] = df['positive'] + df['negative']
df['rating'] = (df['positive'] / df['total_ratings']) * 100
df['rating'] = df['rating'].fillna(0)

# Price conversion (Roughly 1 USD = 83 INR)
df['price_inr'] = df['price_usd'] * 83

# Parse release year
def extract_year(x):
    try:
        if pd.isna(x): return np.nan
        if isinstance(x, str):
            # Clean string before literal_eval if it has weird characters
            d = ast.literal_eval(x)
        else:
            d = x
        date_str = d.get('date', '')
        return int(date_str.split(',')[-1].strip())
    except:
        return np.nan

df['release_year'] = df['release_date'].apply(extract_year)

# Q1: Do free games have higher ratings on average than paid games?
avg_rating = df.groupby('is_free')['rating'].mean()
print(f"Average Rating (Free vs Paid):\n{avg_rating}")

# Q2: How many Action games support Korean?
action_korean = df[(df['genres'].str.contains('Action', na=False)) & (df['supported_languages'].str.contains('Korean', na=False))]
print(f"Action games supporting Korean: {len(action_korean)}")

# Save cleaned data
os.makedirs('data', exist_ok=True)
df.to_csv('data/cleaned_game_data.csv', index=False)
print("Cleaned data saved to data/cleaned_game_data.csv")
