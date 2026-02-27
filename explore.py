import pandas as pd

train = pd.read_csv(r"C:\Users\pc1\OneDrive - University of Venda\Desktop\SME_Project\Train.csv")

country_health = train.groupby(['country', 'Target']).size().unstack(fill_value=0)
country_pct = country_health.div(country_health.sum(axis=1), axis=0).round(3) * 100
print(country_pct)