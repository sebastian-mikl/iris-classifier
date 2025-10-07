import argparse
import os
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train iris classifier')
parser.add_argument('--test-size', type=float, default=0.2, help='Test set size (default: 0.2)')
parser.add_argument('--random-state', type=int, default=42, help='Random state for reproducibility (default: 42)')
args = parser.parse_args()

# Create outputs folder if it doesn't exist
os.makedirs('outputs', exist_ok=True)

# Load data
iris = load_iris()
X = iris.data      # shape (150, 4)
y = iris.target    # shape (150,)
print(iris.feature_names, iris.target_names)

# Split data using command-line arguments
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=args.test_size,
    random_state=args.random_state
)

# Train Decision Tree
model = DecisionTreeClassifier(random_state=args.random_state)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Predictions:", y_pred[:5])
print("True labels:", y_test[:5])

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Generate and save confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix - Decision Tree')
plt.savefig('outputs/confusion_matrix.png', dpi=150, bbox_inches='tight')

# Train k-NN
model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)
print("k-NN accuracy:", accuracy_score(y_test, y_pred2))