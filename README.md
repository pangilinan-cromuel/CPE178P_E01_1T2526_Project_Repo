# Drug Abuse Detection via Facial Analysis
### Using PyTorch, FastAPI, and Flet

## 📘 Overview
This project is a **client–server application** that detects drug abuse through facial image analysis.  
It uses a **PyTorch-trained EfficientNet-B0 model** for classification, **FastAPI** for backend inference,  
and **Flet** for the desktop client interface.

---

## ⚙️ Setup Instructions

### 1. Install Dependencies
Make sure Python is installed, then install the required libraries:

```bash
pip install torch torchvision fastapi uvicorn pillow flet numpy python-multipart
```
## 🖥️ Running the Client 
### 1. Open Terminal Open a new terminal window on your computer. ### 2. Navigate to Client Folder Go to the folder where your client.py file is located:
```bash
cd path/to/client/folder
```
Execute the client script with:
```bash
python client.py
```

## 🖼️ Using the Flet GUI 
### Once the Flet client window opens, you can interact with the application as follows: 
### 1. Upload an Image 
- Click the **Upload** button.
- Select a facial image file in one of the supported formats: .jpg, .jpeg, .png.
- The selected image will be displayed as a preview in the GUI.
### 2. Analyze the Image 
- After uploading, click the **Analyze** button.
- The image will be sent to the server for prediction using the trained EfficientNet-B0 model.
### 3. View Prediction Results 
- The application will display: - **Prediction:** “Drug User” or “Non-User”
- **Confidence Score:** The model’s confidence in its prediction

## 📝 Notes 
1. Ensure the server is running before starting the client.
2. The model file (best_model.pt) must be in the same directory as server.py.
3. If you encounter connection issues, check that both the client and server are using the same host and port (localhost:8000).
4. Supported image formats for upload: .jpg, .jpeg, .png.

## 👨‍💻 Developers 
1. Figarola, Kirsten Cyrille M.
2. Lancero, Leonardo Rigel C.
3. Laureano, Rupert Jay C.
4. Pangilinan, Cromuel A.

**Course:** CPE178P/E01 – Deep Learning Applications 
**University:** Mapúa University, November 2025
