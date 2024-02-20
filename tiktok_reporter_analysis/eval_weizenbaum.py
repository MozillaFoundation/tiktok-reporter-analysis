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
    lambda x: x.strip().split()[-1].lower()
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


precision = precision_score(true_labels, predicted_labels, average="binary", pos_label="informative")
recall = recall_score(true_labels, predicted_labels, average="binary", pos_label="informative")
f1 = f1_score(true_labels, predicted_labels, average="binary", pos_label="informative")
accuracy = accuracy_score(true_labels, predicted_labels)

tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Accuracy: {accuracy}")
print(f"True Positives: {tp}")
print(f"False Positives: {fp}")
print(f"True Negatives: {tn}")
print(f"False Negatives: {fn}")
