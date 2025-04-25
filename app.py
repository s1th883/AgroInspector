import streamlit as st
import torch
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn

# === Streamlit Setup ===
st.set_page_config(page_title="AgroInspector 🌿", layout="centered")

st.title("🌾 AgroInspector")
st.subheader("Crop Image Inspector using Deep Learning")
st.markdown("Upload a crop image to detect if it's **illegal**, and if not, assess its **health status** and **type**.")

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Class Labels ===
illegal_crop_types = [
    'Ayahuasca_Vine', 'Cannabis', 'Chacruna', 'Coca', 'Datura',
    'Khat', 'Opium_Poppy', 'Peyote', 'Psilocybin_Mushrooms', 'Salvia_divinorum'
]

disease_types = [
    "Apple – Apple Black rot", "Apple – Apple Scab", "Apple – Cedar apple rust",
    "Bell pepper – Bell pepper Bacterial spot", "Cherry – Cherry Powdery mildew",
    "Citrus – Citrus Black spot", "Citrus – Citrus canker", "Citrus – Citrus greening",
    "Corn – Corn Common rust", "Corn – Corn Gray leaf spot", "Corn – Corn Northern Leaf Blight",
    "Grape – Grape Black Measles", "Grape – Grape Black rot", "Grape – Grape Isariopsis Leaf Spot",
    "Holy_Basil – holybasil_insect_bite", "Holy_Basil – holybasil_white_spots",
    "Onion – Onion_White_rot", "Peach – Peach Bacterial spot",
    "Potato – Potato Early blight", "Potato – Potato Late blight",
    "Strawberry – Strawberry Leaf scorch",
    "Sugarcane_leafs – Sugarcane_BacterialBlights", "Sugarcane_leafs – Sugarcane_Mosaic",
    "Sugarcane_leafs – Sugarcane_RedRot", "Sugarcane_leafs – Sugarcane_Rust",
    "Sugarcane_leafs – Sugarcane_Yellow",
    "Tomato – Tomato Bacterial spot", "Tomato – Tomato Early blight",
    "Tomato – Tomato Late blight", "Tomato – Tomato Leaf Mold",
    "Tomato – Tomato Mosaic virus", "Tomato – Tomato Septoria leaf spot",
    "Tomato – Tomato Spider mites", "Tomato – Tomato Target Spot",
    "Tomato – Tomato Yellow Leaf Curl Virus", "Watermelon – Watermelon_Anthracnose"
]

healthy_crop_types = [
    "Apple", "Bell pepper", "Cherry", "Citrus", "Corn", "Grape", "Holy Basil",
    "Onion", "Peach", "Potato", "Strawberry", "Sugarcane leafs", "Tomato", "Watermelon"
]

# === Load ResNet-based Model ===
@st.cache_resource
def load_model(path, num_classes):
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

# === Load All 5 Models ===
model1 = load_model("illegal_binary_classifier.pth", 2)
model2 = load_model("illegal_crop_type_classifier.pth", 10)
model3 = load_model("healthy_vs_diseased_classifier.pth", 2)
model4 = load_model("disease_classifier.pth", 46)
model5 = load_model("healthy_crop_classifier.pth", 14)

# === Transform ===
infer_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === Inference Pipeline ===
def run_pipeline(image: Image.Image):
    img = image.convert("RGB")
    tensor = infer_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out1 = model1(tensor)
        pred1 = torch.argmax(out1).item()
        step1 = "Legal" if pred1 == 0 else "Illegal"
        result = {"Step 1: Legal or Illegal": step1}

        if step1 == "Illegal":
            pred2 = torch.argmax(model2(tensor)).item()
            result["Step 2: Illegal Crop Type"] = illegal_crop_types[pred2]
        else:
            pred3 = torch.argmax(model3(tensor)).item()
            step2 = "Healthy" if pred3 == 0 else "Diseased"
            result["Step 3: Healthy or Diseased"] = step2

            if step2 == "Healthy":
                pred5 = torch.argmax(model5(tensor)).item()
                result["Step 4: Healthy Crop Type"] = healthy_crop_types[pred5]
            else:
                pred4 = torch.argmax(model4(tensor)).item()
                result["Step 4: Disease Type"] = disease_types[pred4]

    return result

# === UI Upload + Results ===
uploaded_file = st.file_uploader("📤 Upload a crop image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="📸 Uploaded Image", use_column_width=True)

    with st.spinner("Running deep learning inference..."):
        result = run_pipeline(image)

    st.success("✅ Inspection Complete!")

    st.markdown("---")
    for k, v in result.items():
        st.markdown(f"**{k}**: {v}")
