import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
from sklearn.svm import SVC

# Load the training and test datasets
train_data = pd.read_csv('datasets/verafiles_dataset/train.csv')
test_data = pd.read_csv('datasets/verafiles_dataset/test.csv')

# Extract the labels and text from the datasets
train_labels = train_data['RATING']
train_text = train_data['CONCAT QUOTES']
test_labels = test_data['RATING']
test_text = test_data['CONCAT QUOTES']

# Create a CountVectorizer to convert text into numerical features
vectorizer = CountVectorizer()
train_features = vectorizer.fit_transform(train_text)
test_features = vectorizer.transform(test_text)

# Train an SVM classifier
classifier = SVC()
classifier.fit(train_features, train_labels)

# Predict the labels for the test set
predictions = classifier.predict(test_features)

# Calculate the accuracy, f1 score, and confusion matrix
accuracy = accuracy_score(test_labels, predictions)
f1 = f1_score(test_labels, predictions, average='macro')
confusion = confusion_matrix(test_labels, predictions)

print("Accuracy:", accuracy)
print("F1 Score:", f1)

import matplotlib.pyplot as plt
# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, cmap='Blues', fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')

# Add labels to the confusion matrix
tick_labels = ["FAKE", "FALSE", "MISLEADING"]
plt.xticks(ticks=[0.5, 1.5, 2.5], labels=tick_labels)
plt.yticks(ticks=[0.5, 1.5, 2.5], labels=tick_labels)

# Save the figure as a PNG file
plt.savefig('confusion_matrix.png')
