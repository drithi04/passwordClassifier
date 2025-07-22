# Step 0: Install TensorFlow if needed (usually pre-installed on Colab)
!pip install -q tensorflow

# Step 1: Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dropout, Dense

# Step 2: Upload dataset manually in Colab
from google.colab import files
uploaded = files.upload()

# Step 3: Load dataset
file_name = list(uploaded.keys())[0]
df = pd.read_csv(file_name)

# Step 4: Clean missing data
df['password'] = df['password'].fillna('').astype(str)
num_empty = (df['password'] == '').sum()
print(f"Number of empty/missing passwords replaced: {num_empty}")

# Step 5: Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['strength'])  # 0=Weak, 1=Medium, 2=Strong
y_cat = to_categorical(y, num_classes=3)

# Step 6: Tokenize passwords at char-level and pad sequences
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(df['password'])
X_seq = tokenizer.texts_to_sequences(df['password'])
max_len = 32
X_pad = pad_sequences(X_seq, maxlen=max_len)

# Step 7: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_pad, y_cat, test_size=0.3, stratify=y, random_state=42)

# Step 8: Compute class weights
weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
class_weights = {i: weights[i] for i in range(3)}
print("Class weights:", class_weights)

# Step 9: Build model
vocab_size = len(tokenizer.word_index) + 1
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=64, input_length=max_len),
    Conv1D(128, 5, activation='relu'),
    GlobalMaxPooling1D(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Step 10: Train model
history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=512,
    validation_split=0.1,
    class_weight=class_weights,
    verbose=1
)

# Step 11: Evaluate model
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# Metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print("\n=== Overall Performance ===")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred, target_names=["Weak", "Medium", "Strong"]))

# Step 12: Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Step 13: Suggestion Function
def password_suggestions(password):
    suggestions = []
    if len(password) < 8:
        suggestions.append("Make your password at least 8 characters long.")
    if password.lower() == password:
        suggestions.append("Include uppercase letters.")
    if password.upper() == password:
        suggestions.append("Include lowercase letters.")
    if not any(char.isdigit() for char in password):
        suggestions.append("Add numbers.")
    if not any(char in "!@#$%^&*()-_=+[]{}|;:',.<>?/" for char in password):
        suggestions.append("Add special characters like !@#$%^&*.")
    if not suggestions:
        suggestions.append("Your password looks strong!")
    return suggestions

# Step 14: Prediction Function
class_text_map = {0: "Weak", 1: "Medium", 2: "Strong"}

def predict_password_strength(password):
    seq = tokenizer.texts_to_sequences([password])
    pad = pad_sequences(seq, maxlen=max_len)
    pred_prob = model.predict(pad)
    pred_class = np.argmax(pred_prob, axis=1)[0]
    class_name = class_text_map.get(pred_class, "Unknown")
    
    print(f"\nPredicted strength: {pred_class} - {class_name.upper()} ({pred_prob[0][pred_class]*100:.2f}%)")

    if pred_class in [0, 1]:
        print("\nSuggestions to improve your password:")
        for sug in password_suggestions(password):
            print("- " + sug)

# Step 15: Input loop
while True:
    pw = input("\nEnter a password to check strength (or type 'exit' to stop): ")
    if pw.lower() == 'exit':
        break
    predict_password_strength(pw)
