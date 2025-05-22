import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Reload and filter the dataset
df = pd.read_csv("savant_data (1).csv")
filtered_df = df[df['launch_speed'].notnull() & df['launch_angle'].notnull()
                 & df['estimated_ba_using_speedangle'].notnull()]

# Prepare training data
X = filtered_df[['launch_speed', 'launch_angle']]
y = filtered_df['estimated_ba_using_speedangle']

# Split and train model using scikit-learn 1.1.3
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)
model_compatible = RandomForestRegressor(n_estimators=100, random_state=42)
model_compatible.fit(X_train, y_train)

# Save the retrained model
model_path_compatible = "xba_rf_model.joblib"
joblib.dump(model_compatible, model_path_compatible)

model_path_compatible
