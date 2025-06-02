import streamlit as st
from PIL import Image
import cv2
import numpy as np
import os
import pytesseract
from skimage.metrics import structural_similarity as ssim
actual_image_path = "D:/GenAI/confusionMatrix.jpg"
pytesseract.pytesseract.tesseract_cmd = r"D:\GenAI\tesseract.exe" # Adjust if needed

def preprocess_image_ocr(image_path):
    """Applies preprocessing to improve OCR."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            st.error(f"Error: Could not open image at {image_path}")
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        kernel = np.ones((3, 3), np.uint8)
        dilate = cv2.dilate(thresh, kernel, iterations=1)
        return dilate
    except Exception as e:
        st.error(f"Preprocessing Error (OCR): {e}")
        return None

def extract_labels(image_path):
    """Extracts labels from the handwritten confusion matrix using OCR."""
    processed_image = preprocess_image_ocr(image_path)
    if processed_image is None:
        return {}

    h, w = processed_image.shape
    extracted_info = {}
    regions = {
        "True Class": (int(w * 0.2), int(h * 0.0), int(w * 0.6), int(h * 0.1)),
        "Predicted Class": (int(w * 0.0), int(h * 0.2), int(w * 0.2), int(h * 0.6)),
        "Positive_True": (int(w * 0.3), int(h * 0.15), int(w * 0.2), int(h * 0.1)),
        "Negative_True": (int(w * 0.55), int(h * 0.15), int(w * 0.2), int(h * 0.1)),
        "Positive_Pred": (int(w * 0.1), int(h * 0.3), int(w * 0.15), int(h * 0.2)),
        "Negative_Pred": (int(w * 0.1), int(h * 0.55), int(w * 0.15), int(h * 0.1)),
        "TP": (int(w * 0.25), int(h * 0.25), int(w * 0.25), int(h * 0.25)),
        "FP": (int(w * 0.5), int(h * 0.25), int(w * 0.25), int(h * 0.25)),
        "FN": (int(w * 0.25), int(h * 0.5), int(w * 0.25), int(h * 0.25)),
        "TN": (int(w * 0.5), int(h * 0.5), int(w * 0.25), int(h * 0.25)),
    }

    for label, (x, y, rw, rh) in regions.items():
        roi = processed_image[y:y + rh, x:x + rw]
        text = pytesseract.image_to_string(roi, config='--psm 6').strip().lower().replace(" ", "")
        extracted_info[label] = text
    return extracted_info

def compare_labels(extracted_labels, expected_labels):
    """Compares extracted labels with expected labels."""
    match_count = 0
    for expected_label_base in ["trueclass", "predictedclass", "positive", "negative", "tp", "fp", "fn", "tn"]:
        found = False
        for extracted_label_key, extracted_label_value in extracted_labels.items():
            cleaned_expected = expected_label_base.lower().replace(" ", "")
            cleaned_extracted = extracted_label_value.lower().replace(" ", "")
            if cleaned_expected in cleaned_extracted or cleaned_extracted in cleaned_expected:
                match_count += 1
                found = True
                break
        if not found:
            st.warning(f"Label related to '{expected_label_base}' not clearly found.")

    accuracy = match_count / len(expected_labels)
    return accuracy >= 0.7 # Adjust tolerance as needed

def preprocess_image_ssim(image_path):
    """Applies preprocessing for structural similarity comparison."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            st.error(f"Error: Could not open image at {image_path}")
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (500, 500)) # Standardize size
        return resized
    except Exception as e:
        st.error(f"Preprocessing Error (SSIM): {e}")
        return None

def compare_structure(uploaded_path, actual_path):
    """Compares the structural similarity of the two images."""
    img1 = preprocess_image_ssim(uploaded_path)
    img2 = preprocess_image_ssim(actual_path)

    if img1 is None or img2 is None:
        return 0.0

    score = ssim(img1, img2)
    return score

st.title("Handwritten vs. Actual Confusion Matrix Checker")

actual_image = Image.open(actual_image_path)
st.subheader("Actual Confusion Matrix:")
st.image(actual_image, caption="Reference Confusion Matrix", use_container_width=True)

uploaded_file = st.file_uploader("Upload your handwritten confusion matrix image...", type=["png", "jpg", "jpeg"])

expected_labels_list = ["True Class", "Predicted Class", "Positive", "Negative", "TP", "FP", "FN", "TN"]

if uploaded_file is not None:
    uploaded_image_path = "uploaded_matrix.png"
    with open(uploaded_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.subheader("Uploaded Handwritten Confusion Matrix:")
    uploaded_image = Image.open(uploaded_image_path)
    st.image(uploaded_image, caption="Your Uploaded Image", use_container_width=True)

    extracted = extract_labels(uploaded_image_path)
    st.subheader("Extracted Labels:")
    st.write(extracted)

    labels_match = compare_labels(extracted, expected_labels_list)
    structure_similarity = compare_structure(uploaded_image_path, actual_image_path)
    st.subheader(f"Structural Similarity: {structure_similarity:.2f}")

    label_threshold = 0.7
    structure_threshold = 0.6 # Adjust as needed

    if labels_match and structure_similarity >= structure_threshold:
        st.success("The uploaded handwritten confusion matrix is likely similar to the actual one.")
    elif labels_match:
        st.warning("The labels seem to match, but the overall structure has some differences.")
    else:
        st.error("The uploaded handwritten confusion matrix does not appear to be similar to the actual one.")

    os.remove(uploaded_image_path)
