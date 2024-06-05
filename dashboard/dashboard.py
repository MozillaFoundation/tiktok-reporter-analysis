# For images: https://discuss.streamlit.io/t/cannot-display-imagecolumns-with-streamlit/50957

import streamlit as st
import pandas as pd
import os
from concurrent.futures import TimeoutError
from google.cloud import pubsub_v1

import gzip
import json
from io import BytesIO
from streamlit.runtime.scriptrunner import get_script_run_ctx, add_script_run_ctx


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "regrets-reporter-dev-f440bd973b35.json"

st.title("Realtime FYP Reporter Data Dashboard")


project_id = "regrets-reporter-dev"
subscription_id = "dash-sub"
subscriber = pubsub_v1.SubscriberClient()
subscription_path = subscriber.subscription_path(project_id, subscription_id)


def process_ios_report(message):

    label_to_name = {
        "TikTok URL": "report_link",
        "Category": "category",
        "Comments": "comment",
        None: "other_category",
    }

    raw_report = message["metrics"]["text"]

    category_mapping = {
        "20227308-9f8c-4b9d-b9e6-d48960b4bc94": "Distressing or disturbing",
        "36214f91-b16f-4341-a3a2-57ed779a0e46": "Promotes violence or harm",
        "464683bc-f071-440d-ba3f-e4d495ff1425": "Promotes health/wellness information that is harmful or misleading",
        "a26f8b98-09dd-49b4-a844-8ea7d9f7bed3": "Infringes on my privacy",
        "39436ff6-11ad-45bf-93eb-b898c477ac69": "Spam and/or posted by a bot account",
        "e0fec485-e107-42cc-8849-85adccdb4c44": "Appears to be part of a disinformation campaign",
        "876c8435-dc3f-454c-b74a-bf050b529a53": "Harasses a group of people or an individual",
        "32412f6a-c385-4a5a-b1ff-945d167bbc66": "Political viewpoint I would like to see less of",
        "0d7785e7-b23d-4ead-aad3-2bcc2fd09ef7": "Stereotypes a group of people",
        "b010189e-171c-47be-bda1-8c2471647ce4": "Generally don't feel represented/seen by this video",
        "-otherDropdownItem": "Other",
    }

    def extract_value_by_label(row, label_to_name):
        try:
            items = json.loads(row)["items"]
            label_input_value = {
                item["formItem"].get("label"): item["inputValue"]
                for item in items
                if item["formItem"].get("label") in label_to_name
            }
            return {
                label_to_name[label]: label_input_value[label] for label in label_to_name if label in label_input_value
            }
        except (ValueError, KeyError):
            return {name: None for name in label_to_name.values()}

    report = {}
    for label, name in label_to_name.items():
        report[name] = extract_value_by_label(raw_report["tiktok_report.fields"], label_to_name).get(name, None)
    report["category"] = category_mapping[report["category"]]

    if "reports" not in st.session_state:
        st.session_state["reports"] = pd.DataFrame([report])
    else:
        st.session_state["reports"] = pd.concat(
            [st.session_state["reports"], pd.DataFrame([report])], ignore_index=True
        )


def process_android_report(message):

    raw_report = message["metrics"]["text"]

    report = {}
    for index, name in enumerate(["report_link", "category", "comment"]):
        report[name] = json.loads(raw_report["tiktok_report.fields"])["items"][index]["inputValue"]

    if "reports" not in st.session_state:
        st.session_state["reports"] = pd.DataFrame([report])
    else:
        st.session_state["reports"] = pd.concat(
            [st.session_state["reports"], pd.DataFrame([report])], ignore_index=True
        )


def pubsub_data_available(message: pubsub_v1.subscriber.message.Message) -> None:
    global ctx
    add_script_run_ctx(None, ctx)
    compressed_data = message.data
    with gzip.GzipFile(fileobj=BytesIO(compressed_data)) as f:
        decompressed_data = f.read()
    json_str = decompressed_data.decode("utf-8")
    json_data = json.loads(json_str)
    print(json.dumps(json_data, indent=4))
    if json_data["metadata"]["document_type"] == "tiktok-report":
        if json_data["metadata"]["document_namespace"] in [
            "org-mozilla-ios-tiktok-reporter",
            "org-mozilla-ios-tiktok-reporter-tiktok-reportershare",
        ]:
            process_ios_report(json_data)
        elif json_data["metadata"]["document_namespace"] == "org-mozilla-tiktokreporter":   # Android
            process_android_report(json_data)
    message.ack()
    st.session_state.pubsub_stream.cancel()
    st.rerun()


ctx = get_script_run_ctx()
streaming_pull_future = subscriber.subscribe(subscription_path, callback=pubsub_data_available)
st.session_state["pubsub_stream"] = streaming_pull_future
if "reports" not in st.session_state:
    st.write(f"Listening for first messages on {subscription_path}..\n")
else:
    st.dataframe(st.session_state.reports)

with subscriber:
    try:
        streaming_pull_future.result()
    except TimeoutError:
        streaming_pull_future.cancel()
        streaming_pull_future.result()
