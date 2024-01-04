import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Load the data
data = pd.read_csv('TestDatasetValorantVCT_Dataset.csv')

# Select features and target variable
features = ['rounds', 'rating', 'average_combat_score', 'kills_per_round', 'assists_per_round',
            'first_kills_per_round', 'first_deaths_per_round', 'headshot_percentage',
            'clutch_success_percentage', 'total_kills', 'total_deaths', 'total_assists',
            'total_first_kills', 'total_first_deaths']

target_variable = 'player'  # Replace with the actual target variable if different

X = data[features]
y = data[target_variable]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the target variable to categorical
y_train = y_train.astype('category')
y_test = y_test.astype('category')

# Get categorical codes with proper handling of unknown categories
cat_codes_train, unique_values_train = pd.factorize(y_train, sort=True)
cat_codes_test, unique_values_test = pd.factorize(y_test, sort=True)

# Initialize the XGBoost classifier
model = XGBClassifier()

# Train the model
model.fit(X_train, cat_codes_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(cat_codes_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')