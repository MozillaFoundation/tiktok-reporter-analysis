import os

import pandas as pd


def generate_html_report(results_dir):
    # Read both CSVs
    events_df = pd.read_csv(os.path.join(results_dir, "frame_event_data.csv"))
    descriptions_df = pd.read_parquet(os.path.join(results_dir, "video_descriptions.parquet"))

    # Determine the video number
    events_df["video"] = (events_df["event_name"] == "Scrolling").shift().fillna(0).cumsum().astype(int)

    # Before merging and manipulating, let's round the timestamps
    events_df["timestamp"] = pd.to_datetime(events_df["timestamp"], format="%M:%S.%f")
    events_df["timestamp"] = (events_df["timestamp"] - pd.to_datetime(events_df["timestamp"].dt.date)).dt.round(
        "100ms"
    )  # round to nearest tenth of a second
    events_df["timestamp"] = events_df["timestamp"].dt.seconds + events_df["timestamp"].dt.microseconds / 1e6
    events_df["timestamp"] = events_df["timestamp"].apply(lambda x: "{:02d}:{:04.1f}".format(int(x // 60), x % 60))

    # Merge on video number
    merged_df = events_df.merge(descriptions_df[["video_path", "video", "description"]], on="video", how="left")

    # Start the HTML string with headers, styles, and scripts
    html_str = """
    <style>
        table {
            border-collapse: collapse;
            width: 80%; /* limiting the width */
            margin: auto; /* centering the table */
            font-family: Arial, sans-serif;
        }
        th, td {
            padding: 10px;
            border: 1px solid #ddd;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
        .video-grey {
            background-color: #DCDCDC;
        }
        .video-white {
            background-color: #FFFFFF;
        }
        .other-lightblue {
            background-color: #DCE6F1;
        }
        .other-lightcoral {
            background-color: #FAD3D3;
        }
        tr:hover {
            background-color: #ddd;
        }
    </style>

    <table>
    <thead>
    <tr>
        <th>Video File</th>
        <th>Frame</th>
        <th>Timestamp</th>
        <th>Event Name</th>
        <th>Description</th>
    </tr>
    </thead>
    <tbody>
    """

    # Group by 'video' and create table rows
    color_flag = True
    for _, group in merged_df.groupby(["video_path", "video"]):
        rowspan = len(group)
        first_row = group.iloc[0]
        video_class = "video-grey" if color_flag else "video-white"
        other_class = "other-lightblue" if color_flag else "other-lightcoral"
        video_path_td = f'<td class="{other_class}">{first_row["video_path"]}</td>'
        timestamp_td = f'<td class="{other_class}">{first_row["timestamp"]}</td>'
        frame_td = f'<td class="{other_class}">{first_row["frame"]}</td>'
        event_name_td = f'<td class="{other_class}">{first_row["event_name"]}</td>'
        html_str += f"<tr>{video_path_td}{frame_td}{timestamp_td}{event_name_td}"
        description_td = f'<td rowspan="{rowspan}" class="{video_class}">{first_row["description"]}</td></tr>'
        html_str += description_td

        for _, row in group.iloc[1:].iterrows():
            video_path_td = f'<td class="{other_class}">{row["video_path"]}</td>'
            timestamp_td = f'<td class="{other_class}">{row["timestamp"]}</td>'
            frame_td = f'<td class="{other_class}">{row["frame"]}</td>'
            event_name_td = f'<td class="{other_class}">{row["event_name"]}</td></tr>'
            html_str += f"<tr>{video_path_td}{frame_td}{timestamp_td}{event_name_td}"
        color_flag = not color_flag

    html_str += "</tbody></table>"

    # Write the HTML string to a file
    with open(os.path.join(results_dir, "output.html"), "w") as file:
        file.write(html_str)
