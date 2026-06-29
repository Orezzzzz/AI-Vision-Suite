"""
Streamlit Web Application for a Deep Learning Detection Project

This application demonstrates three deep learning modules:
1. Object Detection
2. Number Plate Detection with OCR
3. Plant Disease Detection
"""

from __future__ import annotations

import sys
import tempfile
import threading
import uuid
from pathlib import Path
from pprint import pformat
from typing import Any

import cv2
import numpy as np
import streamlit as st
import yaml

try:
    import av
    from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, WebRtcMode, webrtc_streamer
except ImportError:
    av = None
    RTCConfiguration = None
    VideoProcessorBase = object
    WebRtcMode = None
    webrtc_streamer = None



# Project paths

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
UTILS_DIR = BASE_DIR / "utils"
SAMPLE_FILES_DIR = BASE_DIR / "sample_files"
RUNTIME_DIR = BASE_DIR / ".streamlit_runtime"

# Existing utility files use simple imports such as "from detector_NPDT import".
# Adding the utils folder to sys.path keeps those imports working from app.py.
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))


# small YAML files required by the detector.

object_classes = [
        'car', 
        'horse', 
        'person', 
        'bicycle', 
        'cat', 
        'dog', 
        'train',
        'aeroplane', 
        'diningtable', 
        'tvmonitor', 
        'chair', 
        'bird', 
        'bottle',
        'motorbike', 
        'pottedplant', 
        'boat', 
        'sofa', 
        'sheep', 
        'cow', 
        'bus'
]

number_plate_classes = ["number_plate"]

plant_classes = [
    'Anthracnose Fruit Rot', 'Blossom Blight', 'Cassava_Bacterial_Disease', 'Cassava_Brown_Leaf_Spot', 'Cassava_Healthy', 'Cassava_Mosaic', 'Cassava_Root_Rot', 'Chili___Anthracnose_fruit', 'Chili___Bacterial_leaf_spot', 'Chili___Healthy_fruit', 'Corn Healthy', 'Corn Smut', 'Corn Streak', 'Corn_Blight', 'Corn_Brown_Spots', 'Corn_Cercosporiose', 'Corn_Charcoal', 'Corn_Chlorotic_Leaf_Spot', 'Corn_Healthy', 'Corn_Insects_Damages', 'Corn_Mildiou', 'Corn_Purple_Discoloration', 'Corn_Rust', 'Corn_Smut', 'Corn_Streak', 'Corn_Stripe', 'Corn_Violet_Decoloration', 'Corn_Yellow_Spots', 'Corn_Yellowing', 'Eggplant___Colorado_potato_beetle', 'Eggplant___Fruit_rot', 'Eggplant___Healthy_fruit', 'Eggplant___Healthy_leaf', 'Gray Mold', 'Leaf Spot', 'Potato___Alternaria_solani_leaf', 'Potato___Common_scab_fruit', 'Potato___Healthy_fruit', 'Potato___Healthy_leaf', 'Potato___Phytopthora_infestans_leaf', 'Potato___Virus_leaf', 'Powdery Mildew Fruit', 'Powdery Mildew Leaf', 'Tomato_Brown_Spots', 'Tomato_Leaf_Curling', 'Tomato_Mildiou', 'Tomato_Mosaic', 'Tomato___Anthracnose_fruit', 'Tomato___Bacterial_spot_leaf', 'Tomato___Early_blight_leaf', 'Tomato___Healthy_fruit', 'Tomato___Healthy_leaf', 'Tomato___Late_blight_leaf', 'Tomato___Leaf_mold', 'Tomato_bacterial_wilt', 'Tomato_healthy', 'spider mites'
]


# Imports from local project utilities
try:
    from detector_OBDT_onnx import ObjectDetector
    from detector_NPDT import NumberPlateDetector
    from detector_PDDT import PlantDiseaseDetector
    from main_NPDT import ANPRSystem
    from video_detector import VideoDetector
    from webcam_detector import WebcamDetector
except ImportError as import_error:
    st.set_page_config(page_title="Computer Vision Suite", layout="wide")
    st.error("One or more project utility files could not be imported.")
    st.exception(import_error)
    st.stop()


# Small adapter class

class DrawableNumberPlateDetector:
    """
    Adapter class used for number plate video detection.
    """

    def __init__(self, detector: NumberPlateDetector):
        self.detector = detector
        self.labels = detector.labels

    def detect(self, image: np.ndarray):
        return self.detector.detect(image)

    def draw_detections(
        self,
        image: np.ndarray,
        boxes: list[list[int]],
        confidences: list[float],
        class_ids: list[int],
        indices: list[int],
    ) -> np.ndarray:
        output_image = image.copy()

        for i in indices:
            x, y, width, height = boxes[i]
            class_id = class_ids[i]
            confidence = confidences[i]
            label = get_class_name(self.labels, class_id)

            cv2.rectangle(
                output_image,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                2,
            )
            cv2.putText(
                output_image,
                f"{label}: {confidence * 100:.2f}%",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        return output_image


class ObjectDetectionVideoProcessor(VideoProcessorBase):
    """
    Real-time webcam processor for streamlit-webrtc.
    """

    def __init__(self, detector: ObjectDetector):
        self.webcam_detector = WebcamDetector(detector)
        self.frame_count = 0
        self.lock = threading.Lock()

    def recv(self, frame):
        frame_bgr = frame.to_ndarray(format="bgr24")

        with self.lock:
            self.frame_count += 1
            frame_skip = self.frame_count % 2 == 0

        if frame_skip:
            try:
                frame_bgr = self.webcam_detector.process_frame(frame_bgr)
            except Exception:
                #this returns the original frame if one frame detection is bad
                pass

        return av.VideoFrame.from_ndarray(frame_bgr, format="bgr24")


# Configuration data

DETECTION_OPTIONS = [
    "Object Detection",
    "Number Plate Detection",
    "Plant Disease Detection",
]

VALID_MODES = {
    "Object Detection": ["Image", "Video", "Webcam"],
    "Number Plate Detection": ["Image", "Video"],
    "Plant Disease Detection": ["Image"],
}

SAMPLE_FOLDERS = {
    "Object Detection": SAMPLE_FILES_DIR / "object_detection",
    "Number Plate Detection": SAMPLE_FILES_DIR / "number_plate_detection",
    "Plant Disease Detection": SAMPLE_FILES_DIR / "plant_disease_detection",
}

MODEL_PATHS = {
    "Object Detection": MODELS_DIR / "OBDT.onnx",
    "Number Plate Detection": MODELS_DIR / "NPDT.onnx",
    "Plant Disease Detection": MODELS_DIR / "PDDT.onnx",
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def ensure_project_folders() -> None:
    """Create folders used by the app if they do not already exist."""

    RUNTIME_DIR.mkdir(exist_ok=True)
    (RUNTIME_DIR / "uploads").mkdir(exist_ok=True)
    (RUNTIME_DIR / "processed_videos").mkdir(exist_ok=True)

    for folder in SAMPLE_FOLDERS.values():
        folder.mkdir(parents=True, exist_ok=True)


def pad_class_names(class_names: list[str], minimum_count: int = 100) -> list[str]:
    """
    Pad class names to avoid index errors when a model predicts an unknown id.

    During project development, the exact class list may change. Padding keeps
    the app stable while still making missing labels easy to notice.
    """

    padded_names = list(class_names)

    while len(padded_names) < minimum_count:
        padded_names.append(f"Class {len(padded_names)}")

    return padded_names


def get_class_name(class_names: list[str], class_id: int) -> str:
    """Return a readable class name for a numeric class id."""

    if 0 <= class_id < len(class_names):
        return class_names[class_id]

    return f"Class {class_id}"


def write_yaml_file(file_name: str, class_names: list[str]) -> Path:
    """
    Write a YOLO-style YAML file from a Python class list.

    The detector classes already expect a yaml_path. Generating the YAML file
    here lets students edit normal Python lists instead of maintaining separate
    configuration files.
    """

    yaml_path = RUNTIME_DIR / file_name
    names_for_model = pad_class_names(class_names)

    yaml_data = {
        "path": str(BASE_DIR),
        "train": "",
        "val": "",
        "nc": len(names_for_model),
        "names": names_for_model,
    }

    with yaml_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(yaml_data, file, sort_keys=False)

    return yaml_path


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """Convert an OpenCV BGR image into an RGB image for Streamlit display."""

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def load_uploaded_image(uploaded_file) -> np.ndarray:
    """Read an uploaded image file as an OpenCV BGR image."""

    file_bytes = np.frombuffer(uploaded_file.getvalue(), np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise ValueError("The uploaded image could not be read.")

    return image_bgr


def load_sample_image(sample_path: Path) -> np.ndarray:
    """Read a sample image from disk as an OpenCV BGR image."""

    image_bgr = cv2.imread(str(sample_path))

    if image_bgr is None:
        raise ValueError(f"The selected image could not be read: {sample_path}")

    return image_bgr


def save_uploaded_file(uploaded_file) -> Path:
    """
    Save an uploaded file to the runtime folder and return its path.

    Video processing libraries work more reliably with file paths than with
    in-memory uploaded file objects.
    """

    safe_name = Path(uploaded_file.name).name
    output_path = RUNTIME_DIR / "uploads" / f"{uuid.uuid4().hex}_{safe_name}"

    with output_path.open("wb") as file:
        file.write(uploaded_file.getvalue())

    return output_path


def list_sample_files(selected_detection: str, selected_mode: str) -> list[Path]:
    """Return all sample files that match the current detection and mode."""

    folder = SAMPLE_FOLDERS[selected_detection]

    if selected_mode == "Image":
        extensions = {".jpg", ".jpeg", ".png"}
    else:
        extensions = {".mp4", ".avi", ".mov"}

    return sorted(
        file_path
        for file_path in folder.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in extensions
    )


def register_current_input(
    source_type: str,
    selected_detection: str,
    selected_mode: str,
    file_name: str,
    file_path: Path | None = None,
    file_bytes: bytes | None = None,
) -> None:
    """Store the active input selection in Streamlit session state."""

    st.session_state["current_input"] = {
        "source_type": source_type,
        "detection": selected_detection,
        "mode": selected_mode,
        "file_name": file_name,
        "file_path": str(file_path) if file_path else None,
        "file_bytes": file_bytes,
    }


def get_current_input(selected_detection: str, selected_mode: str) -> dict[str, Any] | None:
    """Return the selected input only if it matches the current sidebar choice."""

    current_input = st.session_state.get("current_input")

    if current_input is None:
        return None

    if current_input["detection"] != selected_detection:
        return None

    if current_input["mode"] != selected_mode:
        return None

    return current_input


def build_detection_table(
    class_names: list[str],
    confidences: list[float],
    class_ids: list[int],
    indices: list[int],
) -> list[dict[str, Any]]:
    """Create a simple table of prediction results."""

    rows = []

    for index in indices:
        class_id = class_ids[index]
        confidence = confidences[index]

        rows.append(
            {
                "Predicted Class": get_class_name(class_names, class_id),
                "Confidence": f"{confidence * 100:.2f}%",
            }
        )

    return rows


def display_image_results(
    original_image_bgr: np.ndarray,
    annotated_image_bgr: np.ndarray,
    result_rows: list[dict[str, Any]],
    extra_text: str | None = None,
) -> None:
    """Display original image, processed image, and tabular results."""

    st.subheader("Results")

    left_column, right_column = st.columns(2)

    with left_column:
        st.markdown("**Original Image**")
        st.image(bgr_to_rgb(original_image_bgr), use_container_width=True)

    with right_column:
        st.markdown("**Processed Image**")
        st.image(bgr_to_rgb(annotated_image_bgr), use_container_width=True)

    if extra_text is not None:
        st.markdown("**Detected Number Plate**")
        if extra_text.strip():
            st.success(extra_text)
        else:
            st.warning("No readable number plate text was detected.")

    st.markdown("**Predicted Classes and Confidence Scores**")
    if result_rows:
        st.table(result_rows)
    else:
        st.warning("No detections were found for this input.")


def display_video_result(output_video_path: Path) -> None:
    """Display a processed video in Streamlit."""

    st.subheader("Results")
    st.markdown("**Processed Uploaded Video**")
    with open(output_video_path, "rb") as video_file:
        video_bytes = video_file.read()
    import os

    st.write("Video exists:", os.path.exists(output_video_path))
    st.write("Video size:", os.path.getsize(output_video_path))
    st.video(video_bytes)
    st.success("Video processing has completed successfully.")


def show_model_information(selected_detection: str) -> None:
    """Display model details and editable class lists."""

    st.subheader("Model Information")

    model_path = MODEL_PATHS[selected_detection]
    model_status = "Available" if model_path.exists() else "Missing"

    st.write(f"**Model file status:** {model_status}")

    if selected_detection == "Object Detection":
        description = "Detects common real-world objects such as people, vehicles, animals, and household items."
        classes_to_display = object_classes
    elif selected_detection == "Number Plate Detection":
        description = "Detects vehicle number plates and reads the licence plate number."
        classes_to_display = number_plate_classes
    else:
        description = "Detects plant disease categories from leaf or fruit images."
        classes_to_display = plant_classes

    st.write(description)

    with st.expander("Please expand this to see the list of available detections"):
        st.write(
            "Here is the list of possible detections for this model."
        )
        st.code(pformat(classes_to_display), language="python")


def show_input_tabs(selected_detection: str, selected_mode: str) -> None:
    """Display upload and sample file tabs."""

    st.subheader("Input")

    upload_tab, sample_tab = st.tabs(["Upload File", "Select Sample File"])

    if selected_mode == "Image":
        allowed_extensions = ["jpg", "jpeg", "png"]
    else:
        allowed_extensions = ["mp4", "avi", "mov"]

    with upload_tab:
        uploaded_file = st.file_uploader(
            "Upload input file",
            type=allowed_extensions,
            help="Upload an image or video based on the selected mode.",
        )

        if uploaded_file is not None:
            if selected_mode == "Image":
                register_button = st.button("Use Uploaded Image")
                if register_button:
                    register_current_input(
                        source_type="upload",
                        selected_detection=selected_detection,
                        selected_mode=selected_mode,
                        file_name=uploaded_file.name,
                        file_bytes=uploaded_file.getvalue(),
                    )
                    st.success(f"Selected uploaded image: {uploaded_file.name}")
            else:
                register_button = st.button("Use Uploaded Video")
                if register_button:
                    saved_path = save_uploaded_file(uploaded_file)
                    register_current_input(
                        source_type="upload",
                        selected_detection=selected_detection,
                        selected_mode=selected_mode,
                        file_name=uploaded_file.name,
                        file_path=saved_path,
                    )
                    st.success(f"Selected uploaded video: {uploaded_file.name}")

    with sample_tab:
        sample_files = list_sample_files(selected_detection, selected_mode)

        if not sample_files:
            st.warning(
                "No sample files are available yet. Add files to the matching sample_files folder."
            )
            return

        selected_sample = st.selectbox(
            "Choose a sample file",
            options=sample_files,
            format_func=lambda path: path.name,
        )

        if st.button("Use Selected Sample"):
            register_current_input(
                source_type="sample",
                selected_detection=selected_detection,
                selected_mode=selected_mode,
                file_name=selected_sample.name,
                file_path=selected_sample,
            )
            st.success(f"Selected sample file: {selected_sample.name}")


@st.cache_resource(show_spinner=False)
def load_object_detector(model_path: str, yaml_path: str, class_signature: tuple[str, ...]):
    """Load and cache the object detector."""

    return ObjectDetector(model_path=model_path, yaml_path=yaml_path)


@st.cache_resource(show_spinner=False)
def load_number_plate_detector(model_path: str, yaml_path: str, class_signature: tuple[str, ...]):
    """Load and cache the number plate detector."""

    return NumberPlateDetector(model_path=model_path, yaml_path=yaml_path)


@st.cache_resource(show_spinner=False)
def load_anpr_system(model_path: str, yaml_path: str, class_signature: tuple[str, ...]):
    """Load and cache the ANPR system."""

    return ANPRSystem(yolo_model_path=model_path, yaml_path=yaml_path)


@st.cache_resource(show_spinner=False)
def load_plant_detector(model_path: str, yaml_path: str, class_signature: tuple[str, ...]):
    """Load and cache the plant disease detector."""

    return PlantDiseaseDetector(model_path=model_path, yaml_path=yaml_path)


def get_image_from_current_input(current_input: dict[str, Any]) -> np.ndarray:
    """Load the active image input from upload bytes or a sample file path."""

    if current_input["source_type"] == "upload":
        file_bytes = np.frombuffer(current_input["file_bytes"], np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image_bgr is None:
            raise ValueError("The uploaded image could not be read.")

        return image_bgr

    return load_sample_image(Path(current_input["file_path"]))


def get_video_path_from_current_input(current_input: dict[str, Any]) -> Path:
    """Return the active video path."""

    return Path(current_input["file_path"])


def run_object_image_detection(current_input: dict[str, Any], yaml_path: Path) -> None:
    """Run object detection for image input."""

    detector = load_object_detector(
        str(MODEL_PATHS["Object Detection"]),
        str(yaml_path),
        tuple(object_classes),
    )

    image_bgr = get_image_from_current_input(current_input)

    with st.spinner("Running object detection...."):
        boxes, confidences, class_ids, indices = detector.detect(image_bgr)
        annotated_image = detector.draw_detections(
            image_bgr,
            boxes,
            confidences,
            class_ids,
            indices,
        )

    result_rows = build_detection_table(detector.labels, confidences, class_ids, indices)
    display_image_results(image_bgr, annotated_image, result_rows)


def run_object_video_detection(current_input: dict[str, Any], yaml_path: Path) -> None:
    """Run object detection for video input."""

    detector = load_object_detector(
        str(MODEL_PATHS["Object Detection"]),
        str(yaml_path),
        tuple(object_classes),
    )

    video_path = get_video_path_from_current_input(current_input)
    output_path = RUNTIME_DIR / "processed_videos" / f"object_{uuid.uuid4().hex}.mp4"

    with st.spinner("Processing video. This may take a few minutes..."):
        video_detector = VideoDetector(detector)
        processed_path = video_detector.process_video(
            input_video_path=str(video_path),
            output_video_path=str(output_path),
        )

    display_video_result(Path(processed_path))


def run_webcam_detection(yaml_path: Path) -> None:
    """Run real-time object detection using the webcam."""

    st.subheader("Webcam Detection")

    if webrtc_streamer is None or av is None:
        st.error(
            "Webcam mode requires `streamlit-webrtc` and `av`. "
            "Install them with: pip install streamlit-webrtc av"
        )
        return

    detector = load_object_detector(
        str(MODEL_PATHS["Object Detection"]),
        str(yaml_path),
        tuple(object_classes),
    )

    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    st.info("Allow camera access in the browser to start real-time detection.")
    try:
        
        webrtc_streamer(
            key="object-detection-webcam",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_configuration,
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=lambda: ObjectDetectionVideoProcessor(detector),
            async_processing=True,
            async_transform=True
        )
    except KeyError:
        st.warning(
            "Webcam session failed to load. Please refresh the page."
        )


def run_number_plate_image_detection(current_input: dict[str, Any], yaml_path: Path) -> None:
    """Run number plate detection and OCR for image input."""

    anpr_system = load_anpr_system(
        str(MODEL_PATHS["Number Plate Detection"]),
        str(yaml_path),
        tuple(number_plate_classes),
    )

    image_bgr = get_image_from_current_input(current_input)

    # ANPRSystem.predict expects an image path, so uploaded images are saved
    # temporarily before running detection and OCR.
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        temp_image_path = Path(temp_file.name)

    cv2.imwrite(str(temp_image_path), image_bgr)

    try:
        with st.spinner("Detecting number plate and reading text..."):
            results, annotated_image = anpr_system.predict(str(temp_image_path))
    finally:
        temp_image_path.unlink(missing_ok=True)

    result_rows = [
        {
            "Predicted Class": result["class_name"],
            "Confidence": f"{result['confidence'] * 100:.2f}%",
            "Plate Text": result["plate_text"] if result["plate_text"] else "Not readable",
        }
        for result in results
    ]

    plate_texts = [result["plate_text"] for result in results if result["plate_text"]]
    display_image_results(
        image_bgr,
        annotated_image,
        result_rows,
        extra_text=", ".join(plate_texts),
    )


def run_number_plate_video_detection(current_input: dict[str, Any], yaml_path: Path) -> None:
    """Run number plate detection for video input."""

    raw_detector = load_number_plate_detector(
        str(MODEL_PATHS["Number Plate Detection"]),
        str(yaml_path),
        tuple(number_plate_classes),
    )
    drawable_detector = DrawableNumberPlateDetector(raw_detector)

    video_path = get_video_path_from_current_input(current_input)
    output_path = RUNTIME_DIR / "processed_videos" / f"number_plate_{uuid.uuid4().hex}.mp4"

    with st.spinner("Processing number plate video..."):
        video_detector = VideoDetector(drawable_detector)
        processed_path = video_detector.process_video(
            input_video_path=str(video_path),
            output_video_path=str(output_path),
        )

    display_video_result(Path(processed_path))


def run_plant_image_detection(current_input: dict[str, Any], yaml_path: Path) -> None:
    """Run plant disease detection for image input."""

    detector = load_plant_detector(
        str(MODEL_PATHS["Plant Disease Detection"]),
        str(yaml_path),
        tuple(plant_classes),
    )

    image_bgr = get_image_from_current_input(current_input)

    with st.spinner("Running plant disease detection..."):
        boxes, confidences, class_ids, indices = detector.detect(image_bgr)
        annotated_image = detector.draw_detections(
            image_bgr,
            boxes,
            confidences,
            class_ids,
            indices,
        )

    result_rows = build_detection_table(detector.labels, confidences, class_ids, indices)
    display_image_results(image_bgr, annotated_image, result_rows)


def validate_model_file(selected_detection: str) -> bool:
    """Check whether the required ONNX model exists."""

    model_path = MODEL_PATHS[selected_detection]

    if model_path.exists():
        return True

    st.error(f"Model file was not found: {model_path}")
    return False


def run_selected_workflow(selected_detection: str, selected_mode: str) -> None:
    """Route the selected detection type and mode to the correct workflow."""

    if not validate_model_file(selected_detection):
        return

    yaml_paths = {
        "Object Detection": write_yaml_file("object_classes.yaml", object_classes),
        "Number Plate Detection": write_yaml_file("number_plate_classes.yaml", number_plate_classes),
        "Plant Disease Detection": write_yaml_file("plant_classes.yaml", plant_classes),
    }

    if selected_mode == "Webcam":
        run_webcam_detection(yaml_paths["Object Detection"])
        return

    current_input = get_current_input(selected_detection, selected_mode)

    if current_input is None:
        st.warning("Please upload a file or choose a sample file to continue.")
        return

    st.caption(f"Selected file: {current_input['file_name']}")

    try:
        if selected_detection == "Object Detection" and selected_mode == "Image":
            run_object_image_detection(current_input, yaml_paths["Object Detection"])
        elif selected_detection == "Object Detection" and selected_mode == "Video":
            run_object_video_detection(current_input, yaml_paths["Object Detection"])
        elif selected_detection == "Number Plate Detection" and selected_mode == "Image":
            run_number_plate_image_detection(current_input, yaml_paths["Number Plate Detection"])
        elif selected_detection == "Number Plate Detection" and selected_mode == "Video":
            run_number_plate_video_detection(current_input, yaml_paths["Number Plate Detection"])
        elif selected_detection == "Plant Disease Detection" and selected_mode == "Image":
            run_plant_image_detection(current_input, yaml_paths["Plant Disease Detection"])
    except Exception as error:
        st.error("Please refresh the page and try again.")
        st.exception(error)


def main() -> None:
    """Main function that builds the Streamlit page."""

    st.set_page_config(
        page_title="Computer Vision Suite",
        layout="wide",
    )

    ensure_project_folders()

    st.title("Artificial-Intelligence for next-gen cities and agriculture")
    st.write(
        "This  web-application demonstrates image, video, and webcam-based "
        "detection using trained deep learning vision models."
    )

    st.sidebar.title("Model selection")
    selected_detection = st.sidebar.selectbox(
        "Select Detection",
        DETECTION_OPTIONS,
    )

    selected_mode = st.sidebar.selectbox(
        "Select Mode",
        VALID_MODES[selected_detection],
    )

    st.sidebar.markdown("---")
    st.sidebar.write("**Detection Modules**")
    st.sidebar.write("Object Detection")
    st.sidebar.write("Number Plate Detection")
    st.sidebar.write("Plant Disease Detection")

    show_model_information(selected_detection)

    if selected_mode != "Webcam":
        show_input_tabs(selected_detection, selected_mode)

    st.divider()
    run_selected_workflow(selected_detection, selected_mode)

    # TODO: Add batch image processing.
    # TODO: Add model performance analytics.
    # TODO: Add prediction downloads.
    # TODO: Add prediction history.
    # TODO: Add user authentication.


if __name__ == "__main__":
    main()
