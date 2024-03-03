import numpy as np
from detectron2.data import MetadataCatalog
import cv2
import streamlit as st


@st.cache_resource
def get_vid(file_path):
    return cv2.VideoCapture(file_path)


def get_vid_duration(vid):
    fps = vid.get(cv2.CAP_PROP_FPS)
    total_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = total_frames / fps

    return duration


@st.cache_data
def get_metadata(cfg):
    return MetadataCatalog.get(cfg.DATASETS.TRAIN[0])


@st.cache_data
def get_labels(metadata):
    thing_category_names = metadata.thing_classes
    stuff_category_names = metadata.stuff_classes
    return thing_category_names, stuff_category_names


def process_frame(frame, predictor, thing_category_names, stuff_category_names, object_locations, frame_number):
    panoptic_seg, segments_info = predictor(frame)["panoptic_seg"]
    for data in segments_info:
        object_id = data["id"]
        category_id = data["category_id"]
        isthing = data["isthing"]
        label = thing_category_names[category_id] if isthing else stuff_category_names[category_id]
        frame_key = f"frame_{frame_number}"

        if isthing:
            mask = panoptic_seg == object_id
            non_zero_rows, non_zero_cols = np.where(mask)
            if len(non_zero_rows) > 0 and len(non_zero_cols) > 0:
                bbox = [
                    int(min(non_zero_cols)),
                    int(min(non_zero_rows)),
                    int(max(non_zero_cols)),
                    int(max(non_zero_rows))
                ]

                if object_id not in object_locations:
                    object_locations[object_id] = {
                        "instance_id": data["instance_id"],
                        "category_id": category_id,
                        "label": label,
                        "bounding_boxes": {frame_key: bbox},
                        "isthing": isthing
                    }
                else:
                    object_locations[object_id]["bounding_boxes"][frame_key] = bbox
        else:
            if object_id not in object_locations:
                object_locations[object_id] = {
                    "category_id": category_id,
                    "label": label,
                    "bounding_boxes": {},
                    "isthing": isthing
                }


def process_vid(vid, predictor, thing_category_names, stuff_category_names, every_num_frames=15):
    object_locations = {}
    frame_number = 1
    frame_count = 0
    ret, frame = vid.read()
    while ret:
        # print(True)
        if frame_count % every_num_frames == 0:
            # print(True)
            process_frame(frame, predictor, thing_category_names, stuff_category_names, object_locations, frame_number)
            frame_number += 1

        frame_count += 1
        ret, frame = vid.read()
    vid.release()
    return object_locations
