import streamlit as st
from PIL import Image
import numpy as np
import torch
import pickle
from facenet_pytorch import InceptionResnetV1
import torchvision.transforms as transforms

# Load models
model = InceptionResnetV1(pretrained='vggface2').eval()

with open("face_svm.pkl", "rb") as f:
    clf = pickle.load(f)

with open("face_labels.pkl", "rb") as f:
    le = pickle.load(f)

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

st.title("Face Recognition App")

uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model(img_tensor).numpy()

    prediction = clf.predict(embedding)[0]
    name = le.inverse_transform([prediction])[0]

    st.success(f"Prediction: {name}")
