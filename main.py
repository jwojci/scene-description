import os
import sys

import streamlit as st
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger

import frame_func
import nlp_func


def write_bytesio_to_file(filename, bytesio):
    with open(filename, "wb") as outfile:
        outfile.write(bytesio.getbuffer())


@st.cache_resource
def load_predictor():
    # Detectron2 Config
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
    cfg.MODEL.DEVICE = "cpu"
    # Init predictor

    return DefaultPredictor(cfg)


@st.cache_data
def process(video_path, predictor):
    vid = frame_func.get_vid(video_path)

    duration = frame_func.get_vid_duration(vid)
    if duration > 13:
        st.warning("The uploaded video must be 13 seconds or less.")
        sys.exit()
    metadata = frame_func.get_metadata(predictor.cfg)
    thing_category_names, stuff_category_names = frame_func.get_labels(metadata)
    object_locations = frame_func.process_vid(vid, predictor, thing_category_names, stuff_category_names,
                                              every_num_frames=10)
    background_detections, objects = nlp_func.get_background_foreground(object_locations)
    background_content, foreground_content = nlp_func.get_gpt_prompt(background_detections, objects)
    description = nlp_func.get_description(background_content, foreground_content)

    return description


def main():
    # todo Add token splitter if object_locations is too long (todo in nlp_func)
    # todo add input for other peoples api key
    # todo ? maybe add a button Wrong description, to give the model a chance to interpret the information one more time

    setup_logger()

    predictor = load_predictor()

    st.title("Scene description generation")
    st.header("Upload video or choose sample")

    # Sample videos file paths
    sample_videos = [f for f in os.listdir("samples") if f.endswith((".mp4", ".avi", ".mov"))]

    video_choice = st.radio("Choose a video", ["Upload your own"] + sample_videos, index=None,
                            label_visibility="hidden")
    if video_choice == "Upload your own":
        uploaded = st.file_uploader("upload video", label_visibility="hidden", type=["mp4", "avi", "mov"])
        temp_file_to_save = "./temp_file_1.mp4"
        if uploaded:
            # Display uploaded video
            st.video(uploaded)
            generate_button = st.button("Generate description", use_container_width=True)
            if generate_button:
                # save uploaded video to disc
                write_bytesio_to_file(temp_file_to_save, uploaded)
                with st.spinner("In progress..."):
                    stream = process(temp_file_to_save, predictor)
                st.write_stream(stream)
    elif video_choice in sample_videos:
        sample_video_path = os.path.join("samples", video_choice)
        st.video(sample_video_path)
        generate_button = st.button("Generate description", use_container_width=True)
        if generate_button:
            with st.spinner("In progress..."):
                stream = process(sample_video_path, predictor)
            st.write_stream(stream)


if __name__ == "__main__":
    main()
