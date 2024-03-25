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

elif sys.argv[1] == 'test2':
    # Load test1 and train ids
    with open('data/classification_data/test1.txt', 'r') as f:
        test1_ids = [line.strip() for line in f]
    with open('data/classification_data/train.txt', 'r') as f:
        train_ids = [line.strip() for line in f]
    used_ids = test1_ids + train_ids

    # Filter out used ids
    df = df[~df['id'].isin(used_ids)]

    # Sample 250 ids where category is informative
    informative_ids = df[df['category'] == 'informative'].sample(250)['id'].tolist()

    # Sample 250 other ids
    other_ids = df[df['category'] != 'informative'].sample(250)['id'].tolist()

    # Combine the lists
    sampled_ids = informative_ids + other_ids

    # Write the list of ids to a file, one id per line
    with open('data/classification_data/test2.txt', 'w') as f:
        for id in sampled_ids:
            f.write(str(id) + '\n')

elif sys.argv[1] == 'train' or sys.argv[1] == 'moretrain':
    # Load test1 ids
    with open('data/classification_data/test1.txt', 'r') as f:
        test1_ids = [line.strip() for line in f]

    # Load train ids if 'moretrain' option is selected
    if sys.argv[1] == 'moretrain':
        with open('data/classification_data/train.txt', 'r') as f:
            train_ids = [line.strip() for line in f]
        test1_ids += train_ids

    # Filter out test1 ids
    df = df[~df['id'].isin(test1_ids)]

    # Sample one id where category is informative
    informative_id = df[df['category'] == 'informative'].sample(1)['id'].tolist()

    # Sample one other id
    other_id = df[df['category'] != 'informative'].sample(1)['id'].tolist()

    # Combine the lists
    sampled_ids = informative_id + other_id

    # Write the list of ids to a file, one id per line
    if sys.argv[1] == 'train':
        with open('data/classification_data/train.txt', 'w') as f:
            for id in sampled_ids:
                f.write(str(id) + '\n')
    elif sys.argv[1] == 'moretrain':
        for id in sampled_ids:
            print(str(id))

        # while read number; do cp ~/Documents/Lion\'s\ data/WI_repro_videos_only/${number}.mp4 '/Users/jessemccrosky/git/tiktok-reporter-analysis/data/videos/test2'; done  < /Users/jessemccrosky/git/tiktok-reporter-analysis/data/classification_data/test2.txt
# python3 tiktok_reporter_analysis/eval_weizenbaum.py
# time python3 -m tiktok_reporter_analysis analyze_reported data/videos/test1/ --prompt_file=tiktok_reporter_analysis/prompts/llava_prompt_weizenbaum.txt --multimodal --model=lmstudio