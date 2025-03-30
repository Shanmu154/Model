import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the model class (same architecture as trained model)
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = torch.nn.Linear(3, 1)  # Modify based on your model architecture
    
    def forward(self, x):
        return self.fc1(x)

# Load the model
model = MyModel()
model.load_state_dict(torch.load(r"C:\Users\sugan\Desktop\license_plate_detector(1) (1).pt"))
model.eval()  # Set model to evaluation mode

# Streamlit UI
st.title("🚀 PyTorch Model Deployment - Hugging Face")

st.sidebar.header("🔍 User Input Options")
input_method = st.sidebar.radio("Select Input Method", ["Manual Entry", "Upload CSV"])

if input_method == "Manual Entry":
    feature1 = st.slider("Feature 1", 0.0, 10.0, 5.0)
    feature2 = st.slider("Feature 2", 0.0, 10.0, 5.0)
    feature3 = st.slider("Feature 3", 0.0, 10.0, 5.0)

    if st.button("Predict"):
        input_data = torch.tensor([[feature1, feature2, feature3]], dtype=torch.float32)
        prediction = model(input_data).item()
        st.success(f"📊 Prediction: {prediction}")

elif input_method == "Upload CSV":
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("📂 Uploaded Data:", df.head())

        if st.button("Predict for CSV"):
            input_tensor = torch.tensor(df.values, dtype=torch.float32)
            predictions = model(input_tensor).detach().numpy()
            df['Prediction'] = predictions
            st.write("✅ Prediction Results:", df)

            st.subheader("📊 Prediction Distribution")
            plt.figure(figsize=(8, 4))
            sns.histplot(df['Prediction'], bins=20, kde=True)
            st.pyplot(plt)

st.markdown("**Made with ❤️ using Streamlit & PyTorch**")
