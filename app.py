import streamlit as st
from pathlib import Path
from PIL import Image
import torch
import cv2
import numpy as np
from torchvision import datasets, transforms
from model import RetinaNet
from gradcam import GradCAM
from utils import *

# -------------------- Setup --------------------
st.set_page_config(page_title="RetinaLive Pro", layout="wide", page_icon="🩺")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)

# Load class names
dataset = datasets.ImageFolder('datasets/train')
CLASSES = [c.capitalize() for c in dataset.classes]

# Load Model
@st.cache_resource
def load_model():
    model = RetinaNet(num_classes=len(CLASSES))
    model.load_state_dict(torch.load("retina_model.pth", map_location=DEVICE))
    model.eval().to(DEVICE)
    return model

model = load_model()

# -------------------- CSS Styling --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&family=Roboto:wght@400;700&display=swap');

body {background: linear-gradient(to bottom right, #f0f4f8, #d9e2ec); font-family:'Poppins', sans-serif;}
h1,h2,h3,h4,h5 {color:#4834D4; font-family:'Roboto', sans-serif;}
.sidebar .sidebar-content {
    background: linear-gradient(160deg, #6C63FF, #4834D4);
    color:white; box-shadow: 3px 3px 20px rgba(0,0,0,0.2);
}
.stButton>button {
    background: linear-gradient(145deg, #6C63FF, #4834D4);
    color:white; border-radius:20px; border:none;
    height:50px; width:180px; font-size:16px; font-weight:bold;
    box-shadow: 3px 3px 15px rgba(0,0,0,0.2); transition:0.3s;
}
.stButton>button:hover {background: linear-gradient(145deg, #4834D4, #6C63FF); transform: scale(1.05);}
.stTextInput>div>div>input {
    border-radius:15px; height:45px; font-size:16px; padding:0 10px; border:1px solid #ddd;
}
.metric-card {
    background: linear-gradient(145deg, #ffffff, #f3f4ff);
    border-radius:20px; padding:20px; margin:10px;
    text-align:center; box-shadow: 2px 2px 15px rgba(0,0,0,0.1);
}
.card {
    background:white; border-radius:20px; padding:20px; margin:15px 0;
    box-shadow: 3px 3px 25px rgba(0,0,0,0.1); transition: all 0.3s ease;
}
.card:hover {transform: scale(1.02); box-shadow: 5px 5px 30px rgba(72,52,212,0.2);}
.badge-verified {background-color:#28a745; color:white; padding:5px 10px; border-radius:10px; font-weight:bold;}
.badge-pending {background-color:#ffc107; color:white; padding:5px 10px; border-radius:10px; font-weight:bold;}
.upload-card {
    background:#fff; border-radius:15px; padding:15px; margin:10px 0;
    box-shadow: 2px 2px 15px rgba(0,0,0,0.05);
}
.upload-card:hover {background-color:#f0f4ff; transform: scale(1.01);}
</style>
""", unsafe_allow_html=True)

# -------------------- Utilities --------------------
def load_image_streamlit(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return transform(image).unsqueeze(0)

# -------------------- Login / Signup --------------------
def login_page():
    st.markdown("<h2 style='color:#4834D4;text-align:center;'>👁️ RetinaLive Pro</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#555;'>Login or Signup to continue</p>", unsafe_allow_html=True)
    menu = ["Login","Signup"]
    choice = st.selectbox("Select Option", menu)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if choice=="Signup":
        role = st.selectbox("Role", ["user","doctor","admin"])
        if st.button("Signup"):
            if create_user(username, password, role):
                st.success("✅ Account created successfully! Please login.")
            else:
                st.error("⚠️ Username already exists.")
    if choice=="Login":
        if st.button("Login"):
            auth = authenticate_user(username, password)
            if auth:
                st.session_state["user"] = auth
                st.success(f"Logged in as {auth['role']}")
                st.stop()
            else:
                st.error("❌ Invalid username or password.")

# -------------------- Logout --------------------
def logout_button():
    if st.sidebar.button("Logout"):
        st.session_state.pop("user", None)
        st.stop()

# -------------------- Metrics --------------------
def show_metrics():
    users = get_all_users()
    uploads = get_all_uploads()
    verified = sum([1 for u in uploads if u[5]==1])
    unverified = len(uploads)-verified
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(f"<div class='metric-card'><h3>👥 Users</h3><h2>{len(users)}</h2></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric-card'><h3>📤 Uploads</h3><h2>{len(uploads)}</h2></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='metric-card'><h3>✅ Verified</h3><h2>{verified}</h2></div>", unsafe_allow_html=True)
    col4.markdown(f"<div class='metric-card'><h3>⏳ Pending</h3><h2>{unverified}</h2></div>", unsafe_allow_html=True)

# -------------------- User Dashboard --------------------
def user_dashboard(user_id):
    st.markdown("<h2 class='card'>👤 User Dashboard</h2>", unsafe_allow_html=True)
    logout_button()
    show_metrics()
    st.markdown("<h4>📸 Upload Retinal Image</h4>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg","png","jpeg"])

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)
        tensor = load_image_streamlit(uploaded_file).to(DEVICE)

        with torch.no_grad():
            output = model(tensor)
            _, pred = torch.max(output, 1)
            prediction = CLASSES[pred.item()]
        st.success(f"🧠 Model Prediction: **{prediction}**")

        # Grad-CAM
        try:
            target_layer = model.model.features[-1]
            gradcam = GradCAM(model, target_layer)
            cam = gradcam.generate_cam(tensor, pred.item())
            cam_img = np.array(img.resize((224,224)))
            heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
            superimposed = cv2.addWeighted(cam_img, 0.6, heatmap, 0.4, 0)
            gradcam_path = UPLOAD_FOLDER/f"gradcam_{uploaded_file.name}"
            Image.fromarray(superimposed).save(gradcam_path)
            st.image(superimposed, caption="Grad-CAM Heatmap", use_column_width=True)
        except:
            gradcam_path = None

        img_path = UPLOAD_FOLDER/f"{uploaded_file.name}"
        uploaded_file.seek(0)
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        save_upload(user_id, str(img_path), str(gradcam_path), prediction)
        st.success("✅ Uploaded and sent for doctor verification!")

    st.subheader("📂 Your Upload History")
    uploads = get_user_uploads(user_id)
    for u in uploads:
        img_name = Path(u[2]).name if u[2] else "N/A"
        prediction = u[4] if u[4] else "Unknown"
        comments = u[6] if len(u) > 6 and u[6] else "No comments yet"
        status_badge = "<span class='badge-verified'>Verified</span>" if u[5] else "<span class='badge-pending'>Pending</span>"
        st.markdown(f"""
        <div class='upload-card'>
        <b>🖼️ Image:</b> {img_name}<br>
        <b>Prediction:</b> {prediction}<br>
        <b>Status:</b> {status_badge}<br>
        <b>Doctor Comments:</b> {comments}
        </div>
        """, unsafe_allow_html=True)

# -------------------- Doctor Dashboard --------------------
def doctor_dashboard():
    st.markdown("<h2 class='card'>🩺 Doctor Dashboard</h2>", unsafe_allow_html=True)
    logout_button()
    show_metrics()
    uploads = get_all_uploads()
    for u in uploads:
        user_info = get_user_by_id(u[1])
        username = user_info[1] if user_info else "Unknown"
        status_badge = "<span class='badge-verified'>Verified</span>" if u[5] else "<span class='badge-pending'>Pending</span>"
        st.markdown(f"<div class='card'><b>User:</b> {username}<br><b>Prediction:</b> {u[4]}<br><b>Status:</b> {status_badge}</div>", unsafe_allow_html=True)
        if u[2]: st.image(u[2], width=250)
        if u[3]: st.image(u[3], width=250, caption="Grad-CAM")
        if u[5]==0:
            comments = st.text_input(f"Comments for Upload ID {u[0]}", key=f"c{u[0]}")
            if st.button(f"Verify {u[0]}", key=f"b{u[0]}"):
                verify_upload(u[0], comments)
                st.success("✅ Verified successfully!")
                st.stop()

# -------------------- Admin Dashboard --------------------
def admin_dashboard():
    st.markdown("<h2 class='card'>👨‍💻 Admin Dashboard</h2>", unsafe_allow_html=True)
    logout_button()
    show_metrics()
    st.subheader("🧾 All Users")
    users = get_all_users()
    for u in users:
        st.markdown(f"<div class='card'><b>ID:</b> {u[0]}<br><b>Username:</b> {u[1]}<br><b>Role:</b> {u[2]}</div>", unsafe_allow_html=True)

    st.subheader("📦 All Uploads")
    uploads = get_all_uploads()
    for u in uploads:
        user_info = get_user_by_id(u[1])
        username = user_info[1] if user_info else "Unknown"
        status_badge = "<span class='badge-verified'>Verified</span>" if u[5] else "<span class='badge-pending'>Pending</span>"
        comments = u[6] if len(u)>6 and u[6] else "No comments"
        st.markdown(f"""
        <div class='card'>
        <b>ID:</b> {u[0]} | <b>User:</b> {username}<br>
        <b>Prediction:</b> {u[4]}<br>
        <b>Status:</b> {status_badge}<br>
        <b>Comments:</b> {comments}
        </div>
        """, unsafe_allow_html=True)

# -------------------- Main --------------------
def main():
    if "user" not in st.session_state:
        login_page()
    else:
        role = st.session_state["user"]["role"]
        user_id = st.session_state["user"]["user_id"]
        st.sidebar.markdown(f"<h4 style='color:white'>👤 {st.session_state['user']['username']} ({role})</h4>", unsafe_allow_html=True)
        if role=="user":
            user_dashboard(user_id)
        elif role=="doctor":
            doctor_dashboard()
        else:
            admin_dashboard()

if __name__=="__main__":
    main()
