import asyncio
import base64
import json
import threading
import io
import os
from tkinter import Tk, Canvas, Button, PhotoImage, Label, filedialog
from PIL import Image, ImageTk
import websockets

# ----------------------------
# Tkinter Window Setup
# ----------------------------
window = Tk()
window.geometry("580x420")
window.configure(bg="#FFFFFF")
window.title("Drug Abuse Detection")

canvas = Canvas(window, bg="#FFFFFF", height=420, width=580, bd=0, highlightthickness=0, relief="ridge")
canvas.place(x=0, y=0)
canvas.create_rectangle(0.0, 0.0, 580.0, 84.0, fill="#0B8341", outline="")
canvas.create_text(290.0, 42.0, text="Drug Abuse Detection", fill="#FFFFFF",
                   font=("Arial", 32), anchor="center")

# ----------------------------
# Global Variables
# ----------------------------
uploaded_photo = None
uploaded_file_path = None
result_label = None

# ----------------------------
# WebSocket Server URL
# ----------------------------
SERVER_URL = "ws://localhost:8000/ws"

# ----------------------------
# Upload Area Setup
# ----------------------------
UPLOAD_AREA_WIDTH = 180
UPLOAD_AREA_HEIGHT = 180
upload_area_x = (580 - UPLOAD_AREA_WIDTH) // 2
upload_area_y = 120

upload_area = Label(window, bg="#F0F0F0", relief="solid", bd=2, cursor="hand2",
                   text="Click to Upload", font=("Arial", 9),
                   compound="center", fg="#666666")
upload_area.place(x=upload_area_x, y=upload_area_y,
                  width=UPLOAD_AREA_WIDTH, height=UPLOAD_AREA_HEIGHT)

# ----------------------------
# Utility Functions
# ----------------------------
def clear_results():
    global result_label
    if result_label:
        result_label.destroy()
        result_label = None

def handle_file_select(file_path):
    """Handles image upload and resizing."""
    global uploaded_photo, uploaded_file_path
    if not file_path:
        return

    try:
        uploaded_file_path = file_path
        img = Image.open(file_path).convert("RGB")

        # Resize image to fit upload area
        img.thumbnail((UPLOAD_AREA_WIDTH - 10, UPLOAD_AREA_HEIGHT - 10))
        background = Image.new("RGB", (UPLOAD_AREA_WIDTH, UPLOAD_AREA_HEIGHT), "#F0F0F0")
        x_offset = (UPLOAD_AREA_WIDTH - img.width) // 2
        y_offset = (UPLOAD_AREA_HEIGHT - img.height) // 2
        background.paste(img, (x_offset, y_offset))

        uploaded_photo = ImageTk.PhotoImage(background)
        upload_area.config(image=uploaded_photo, text="")
        clear_results()
        print(f"✓ Image uploaded: {os.path.basename(file_path)}")

    except Exception as e:
        print(f"✗ Error loading image: {e}")

def upload_image(event=None):
    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    handle_file_select(file_path)

# ----------------------------
# Detection Logic
# ----------------------------
def detect_image():
    if not uploaded_file_path:
        print("✗ No image uploaded")
        return
    print("🔍 Starting detection...")
    threading.Thread(target=run_detection, daemon=True).start()

def run_detection():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(send_image_to_server(uploaded_file_path))
    finally:
        loop.close()

async def send_image_to_server(file_path):
    """Sends image to server and waits for prediction result."""
    try:
        print("🔄 Connecting to server...")
        async with websockets.connect(SERVER_URL, ping_timeout=60) as ws:
            with open(file_path, "rb") as image_file:
                image_bytes = image_file.read()
            b64_img = base64.b64encode(image_bytes).decode("utf-8")
            await ws.send(json.dumps({"image": b64_img}))
            print("📤 Image sent to server")

            response = await ws.recv()
            data = json.loads(response)

            prediction = data.get("prediction", "error")
            confidence = data.get("confidence", 0.0)

            print(f"🎯 Prediction: {prediction}, Confidence: {confidence:.4f}")
            window.after(0, show_prediction_result, prediction, confidence)

    except Exception as e:
        print(f"✗ Error communicating with server: {e}")
        import traceback
        traceback.print_exc()

def show_prediction_result(prediction, confidence):
    """Display classification or face detection message."""
    global result_label
    clear_results()

    if prediction == "not_user":
        result_text = f"✅ NOT A DRUG USER ({confidence*100:.1f}%)"
        color = "green"
    elif prediction == "drug_user":
        result_text = f"❌ DRUG USER ({confidence*100:.1f}%)"
        color = "red"
    else:
        result_text = "❗ Error: Unable to process image."
        color = "gray"

    result_label = Label(window, text=result_text, font=("Arial", 12, "bold"), fg=color, bg="#FFFFFF")
    result_label.place(x=290, y=upload_area_y + UPLOAD_AREA_HEIGHT + 20, anchor="center")
    print(f"Result displayed: {result_text}")

# ----------------------------
# Button Setup
# ----------------------------
upload_area.bind("<Button-1>", upload_image)

detect_button_y = upload_area_y + UPLOAD_AREA_HEIGHT + 60
try:
    button_image_1 = PhotoImage(file="button_1.png")
    button_1 = Button(window, image=button_image_1, borderwidth=0, highlightthickness=0,
                     relief="flat", command=detect_image)
    button_1.place(x=206, y=detect_button_y, width=168.0, height=41.0)
except Exception as e:
    print(f"⚠ Could not load button_1.png: {e}")
    button_1 = Button(window, text="DETECT", command=detect_image,
                     bg="#2196F3", fg="white", font=("Arial", 14, "bold"))
    button_1.place(x=206, y=detect_button_y, width=168.0, height=41.0)

# Hover effect
if "button_image_1" in locals():
    try:
        button_image_hover_1 = PhotoImage(file="button_hover_1.png")
        def button_hover(e): button_1.config(image=button_image_hover_1)
        def button_leave(e): button_1.config(image=button_image_1)
        button_1.bind("<Enter>", button_hover)
        button_1.bind("<Leave>", button_leave)
    except Exception as e:
        print(f"⚠ Could not load hover image: {e}")

# ----------------------------
# Startup
# ----------------------------
print("🚀 Application started successfully!")
window.resizable(False, False)
window.mainloop()
