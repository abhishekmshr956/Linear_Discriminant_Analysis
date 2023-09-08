import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# generate a synthetic dataset
np.random.seed(0)
num_samples = 100
num_features = 16

# Generate random feature vectors
X = np.random.rand(num_samples, num_features)

# Generate corresponding labels (e.g. binary labels 0 or 1)
y = np.random.randint(2, size = num_samples)

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and fit the LDA model
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lda.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Display a classification report
report = classification_report(y_test, y_pred)
print(report)