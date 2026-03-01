import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Data Loading
file_path = r"C:\Users\muham\Downloads\archive (8)\marketing_campaign.csv"
df = pd.read_csv(file_path, sep='\t') 

# 2. Data Preprocessing
df.dropna(inplace=True)

purchase_cols = ['NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
df['TotalPurchases'] = df[purchase_cols].sum(axis=1)

campaign_cols = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response']
df['AcceptedAnyCampaign'] = df[campaign_cols].max(axis=1)

# 3. Statistical Analysis (T-Test)
group_campaign = df[df['AcceptedAnyCampaign'] == 1]['TotalPurchases']
group_no_campaign = df[df['AcceptedAnyCampaign'] == 0]['TotalPurchases']

t_stat, p_value = stats.ttest_ind(group_campaign, group_no_campaign)

print(f"Average Purchases (Campaign Accepted): {group_campaign.mean():.2f}")
print(f"Average Purchases (Campaign Not Accepted): {group_no_campaign.mean():.2f}")
print(f"T-test p-value: {p_value:.5f}\n")

# 4. Machine Learning (Feature Importance Analysis)
features = ['Year_Birth', 'Income', 'Kidhome', 'Teenhome', 'Recency'] + campaign_cols
X = df[features]
y = df['TotalPurchases']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Model R2 Score: {r2_score(y_test, y_pred):.2f}\n")

feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
print("Feature Importances:")
print(feature_importances)