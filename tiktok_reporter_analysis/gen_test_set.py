import pandas as pd
import sys

# Load the data
df = pd.read_csv('data/classification_data/weizenbaum.csv')

# Remove duplicates
df = df.drop_duplicates(subset='id')

# Check command line argument
if sys.argv[1] == 'test1':
    # Sample 50 ids where category is informative
    informative_ids = df[df['category'] == 'informative'].sample(50)['id'].tolist()

    # Sample 50 other ids
    other_ids = df[df['category'] != 'informative'].sample(50)['id'].tolist()

    # Combine the lists
    sampled_ids = informative_ids + other_ids

    # Write the list of ids to a file, one id per line
    with open('data/classification_data/test1.txt', 'w') as f:
        for id in sampled_ids:
            f.write(str(id) + '\n')

elif sys.argv[1] == 'train':
    # Load test1 ids
    with open('data/classification_data/test1.txt', 'r') as f:
        test1_ids = [line.strip() for line in f]

    # Filter out test1 ids
    df = df[~df['id'].isin(test1_ids)]

    # Sample one id where category is informative
    informative_id = df[df['category'] == 'informative'].sample(1)['id'].tolist()

    # Sample one other id
    other_id = df[df['category'] != 'informative'].sample(1)['id'].tolist()

    # Combine the lists
    sampled_ids = informative_id + other_id

    # Write the list of ids to a file, one id per line
    with open('data/classification_data/train.txt', 'w') as f:
        for id in sampled_ids:
            f.write(str(id) + '\n')

        # while read number; do cp ~/Documents/Lion\'s\ data/WI_repro_videos_only/${number}.mp4 '/Users/jessemccrosky/git/tiktok-reporter-analysis/data/videos/test1'; done  < /Users/jessemccrosky/git/tiktok-reporter-analysis/data/classification_data/test1.txt
# python3 tiktok_reporter_analysis/eval_weizenbaum.py
# time python3 -m tiktok_reporter_analysis analyze_reported data/videos/test1/ --prompt_file=tiktok_reporter_analysis/prompts/llava_prompt_weizenbaum.txt --multimodal --model=lmstudio