import streamlit as st
from PIL import Image
import numpy as np
import torch
import pickle
from facenet_pytorch import InceptionResnetV1
import torchvision.transforms as transforms

# Load the trained SVM model and label encoder
@st.cache_resource
def load_model():
    with open("face_svm.pkl", "rb") as f:
        clf = pickle.load(f)
    with open("face_labels.pkl", "rb") as f:
        le = pickle.load(f)
    return clf, le

clf, le = load_model()

# Load FaceNet model
@st.cache_resource
def load_facenet():
    return InceptionResnetV1(pretrained='vggface2').eval()

model = load_facenet()

# Preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Streamlit UI
st.set_page_config(page_title="Face Recognition", page_icon="🤖")
st.title("🔍 Face Recognition App")
st.markdown("Upload a clear face image to identify the person.")

uploaded_file = st.file_uploader("Choose a JPG/PNG image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess and predict
        with st.spinner("Processing..."):
            img_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                embedding = model(img_tensor).numpy()

            pred_id = clf.predict(embedding)[0]
            pred_name = le.inverse_transform([pred_id])[0]

        st.success(f"✅ Prediction: {pred_name}")
    except Exception as e:
        st.error(f"❌ Error: {e}")
