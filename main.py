import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load data
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
data = pd.read_csv(url, sep='\t', names=['label', 'message'])

# Chuyển label thành số
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Tách dữ liệu
X = data['message']
y = data['label']

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Chuyển text thành số
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Dự đoán
y_pred = model.predict(X_test_vec)

# Đánh giá
print("Accuracy:", accuracy_score(y_test, y_pred))

# Test thử
msg = ["Congratulations! You won a free ticket"]
msg_vec = vectorizer.transform(msg)
print("Prediction:", model.predict(msg_vec))