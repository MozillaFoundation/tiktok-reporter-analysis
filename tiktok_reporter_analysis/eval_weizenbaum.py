import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix

# Load the csv file into a pandas DataFrame
df = pd.read_csv("data/classification_data/weizenbaum.csv")
print(f"Number of rows in the dataframe: {len(df)}")
print(f"Number of unique ids in the dataframe: {df['id'].nunique()}")

# Select only the 'id' and 'category' columns
df = df[["id", "category"]]
# Load the parquet file into a pandas DataFrame
df2 = pd.read_parquet("data/results/video_descriptions.parquet")

# Extract the filename without path and extension as 'id'
df2["id"] = df2["video_path"].apply(lambda x: x.split("/")[-1].split(".")[0])

# Select only the 'id' and 'description' columns
df2 = df2[["id", "description"]]

# Rename 'description' column to 'category'
df2.rename(columns={"description": "category"}, inplace=True)
# Replace the category with its last word and convert to lower case
df2["category"] = df2["category"].apply(
    lambda x: ''.join(e for e in x.strip().split()[-1].lower() if e.isalnum())
    if len(x.strip().split()) != 0 else "other"
)

df["id"] = df["id"].astype(str)
# Merge the two dataframes on 'id'
merged_df = df.merge(df2, on="id", how="inner").drop_duplicates(subset="id", keep="first")


binary_flag = True  # Set this flag to False for multi-class evaluation
debug_flag = False  # Set this flag to True for debug prints

true_labels = []
predicted_labels = []

# Iterate over each row in the merged dataframe
for index, row in merged_df.iterrows():
    # Check if the categories match
    if binary_flag:
        if row["category_x"] == "informative" and row["category_y"] == "informative":
            true_labels.append("informative")
            predicted_labels.append("informative")
            if debug_flag:
                print(f"Match found for id {row['id']}. Category: {row['category_x']}")
        elif row["category_x"] != "informative" and row["category_y"] != "informative":
            true_labels.append("non-informative")
            predicted_labels.append("non-informative")
            if debug_flag:
                print(f"Match found for id {row['id']}. Category: Non-informative")
        else:
            if row["category_x"] == "informative":
                true_labels.append("informative")
                predicted_labels.append("non-informative")
            else:
                true_labels.append("non-informative")
                predicted_labels.append("informative")
            if debug_flag:
                print(f"No match found for id {row['id']}. Categories: {row['category_x']} and {row['category_y']}")
    else:
        if row["category_x"] == row["category_y"]:
            true_labels.append(row["category_x"])
            predicted_labels.append(row["category_y"])
            if debug_flag:
                print(f"Match found for id {row['id']}. Category: {row['category_x']}")
        else:
            true_labels.append(row["category_x"])
            predicted_labels.append(row["category_y"])
            if debug_flag:
                print(f"No match found for id {row['id']}. Categories: {row['category_x']} and {row['category_y']}")

# Calculate and print precision, recall, F1, and accuracy


from sklearn.utils import resample
import numpy as np

def compute_confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    se = np.std(data, ddof=1) / np.sqrt(n)
    h = se * 1.96  # For 95% confidence
    return mean, mean - h, mean + h

# Bootstrap resampling to compute confidence intervals
n_iterations = 1000
precision_scores = []
recall_scores = []
f1_scores = []
accuracy_scores = []

for _ in range(n_iterations):
    indices = resample(range(len(true_labels)), replace=True)
    resampled_true = [true_labels[i] for i in indices]
    resampled_pred = [predicted_labels[i] for i in indices]
    
    precision_scores.append(precision_score(resampled_true, resampled_pred, average="binary", pos_label="informative"))
    recall_scores.append(recall_score(resampled_true, resampled_pred, average="binary", pos_label="informative"))
    f1_scores.append(f1_score(resampled_true, resampled_pred, average="binary", pos_label="informative"))
    accuracy_scores.append(accuracy_score(resampled_true, resampled_pred))

precision = np.mean(precision_scores)
recall = np.mean(recall_scores)
f1 = np.mean(f1_scores)
accuracy = np.mean(accuracy_scores)

precision_mean, precision_lower, precision_upper = compute_confidence_interval(precision_scores)
recall_mean, recall_lower, recall_upper = compute_confidence_interval(recall_scores)
f1_mean, f1_lower, f1_upper = compute_confidence_interval(f1_scores)
accuracy_mean, accuracy_lower, accuracy_upper = compute_confidence_interval(accuracy_scores)

tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()

print(f"Precision: {precision} (95% CI: {precision_lower} - {precision_upper})")
print(f"Recall: {recall} (95% CI: {recall_lower} - {recall_upper})")
print(f"F1 Score: {f1} (95% CI: {f1_lower} - {f1_upper})")
print(f"Accuracy: {accuracy} (95% CI: {accuracy_lower} - {accuracy_upper})")
print(f"True Positives: {tp}")
print(f"False Positives: {fp}")
print(f"True Negatives: {tn}")
print(f"False Negatives: {fn}")
