import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
try:
    with open('data.pickle', 'rb') as f:
        data_dict = pickle.load(f)

    data = np.array(data_dict['data'])
    labels = np.array(data_dict['labels'])
    
    if len(data) == 0 or len(labels) == 0:
        raise ValueError("Dataset is empty. Please check the data file.")

except (FileNotFoundError, EOFError, ValueError) as e:
    print(f"Error loading dataset: {e}")
    exit()

# Split dataset into training (70%), validation (15%), and testing (15%)
x_train, x_temp, y_train, y_temp = train_test_split(
    data, labels, test_size=0.3, shuffle=True, stratify=labels, random_state=42
)

x_val, x_test, y_val, y_test = train_test_split(
    x_temp, y_temp, test_size=0.5, shuffle=True, stratify=y_temp, random_state=42
)

# Initialize and train the model with controlled complexity
model = RandomForestClassifier(
    n_estimators=100,  
    max_depth=15,
    min_samples_split=5, 
    random_state=42
)
model.fit(x_train, y_train)

# Cross-validation to get a realistic accuracy estimate
cv_scores = cross_val_score(model, x_train, y_train, cv=5)
print(f"\nCross-Validation Accuracy: {cv_scores.mean() * 100:.2f}% ± {cv_scores.std() * 100:.2f}%")

# Evaluate on validation set
y_val_predict = model.predict(x_val)
val_accuracy = accuracy_score(y_val, y_val_predict)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# Evaluate on test set
y_test_predict = model.predict(x_test)
test_accuracy = accuracy_score(y_test, y_test_predict)

# Calculate classification report and confusion matrix
class_report = classification_report(y_test, y_test_predict)
conf_matrix = confusion_matrix(y_test, y_test_predict)

# Display results
print(f'\nFinal Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)')
print("\nClassification Report:\n", class_report)
print("Confusion Matrix:\n", conf_matrix)

# Save the trained model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("\nModel saved successfully as 'model.p'.")
