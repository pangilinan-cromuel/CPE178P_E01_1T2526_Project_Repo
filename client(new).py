import flet as ft
import requests
import threading
from PIL import Image
import cv2
import numpy as np

SERVER_URL = "http://127.0.0.1:8000/upload"

def main(page: ft.Page):
    page.title = "Drug User Detection System"
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.bgcolor = ft.Colors.GREY_50
    page.padding = 20

    selected_file = {"path": None, "name": None}

    # Create FilePicker first and add to page overlay
    file_picker = ft.FilePicker()
    page.overlay.append(file_picker)

    # Load OpenCV Haar Cascade for face detection
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    except:
        face_cascade = None

    def detect_face_in_image(image_path):
        """Detect if the image contains a human face using OpenCV"""
        if face_cascade is None:
            return True  # Skip validation if cascade couldn't be loaded
        
        try:
            image = cv2.imread(image_path)
            if image is None:
                return False
                
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(60, 60)
            )
            
            return len(faces) > 0
            
        except Exception as e:
            print(f"Face detection error: {e}")
            return False

    # Title
    title = ft.Container(
        content=ft.Column([
            ft.Text(
                "DRUG USER DETECTION", 
                size=32, 
                weight=ft.FontWeight.BOLD,
                color=ft.Colors.BLUE_900
            ),
            ft.Text(
                "Upload a clear face image for detection",
                size=16,
                color=ft.Colors.GREY_600
            )
        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
        margin=ft.margin.only(bottom=30)
    )

    # Result text displays
    result_display = ft.Text(
        "Upload a face image and click detect", 
        size=16, 
        color=ft.Colors.GREY_600
    )

    confidence_display = ft.Text(
        "", 
        size=14, 
        color=ft.Colors.GREY_700,
        visible=False
    )

    result_container = ft.Container(
        content=ft.Column([
            ft.Text("Detection Result", size=18, weight=ft.FontWeight.BOLD),
            result_display,
            confidence_display
        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
        padding=20,
        border_radius=10,
        bgcolor=ft.Colors.GREY_100,
        visible=True
    )

    def on_file_picked(e: ft.FilePickerResultEvent):
        if e.files:
            file = e.files[0]
            
            if not file.name.lower().endswith(('.jpg', '.jpeg', '.png')):
                show_error("❌ Please upload only JPG, JPEG, or PNG files")
                return
                
            if not detect_face_in_image(file.path):
                show_error("❌ No face detected. Please upload a clear face image.")
                return
            
            selected_file["path"] = file.path
            selected_file["name"] = file.name
            
            img_display.content.src = file.path
            upload_area.visible = False
            img_display.visible = True
            select_button.visible = True
            
            result_display.value = "Face detected - Ready for analysis"
            result_display.color = ft.Colors.GREEN_600
            result_container.bgcolor = ft.Colors.GREEN_50
            error_text.visible = False
            confidence_display.visible = False
            
            detect_button.disabled = False
            page.update()

    def select_file(e=None):
        file_picker.pick_files(
            allow_multiple=False,
            allowed_extensions=["jpg", "jpeg", "png"]
        )

    def detect_image(e):
        if not selected_file["path"]:
            show_error("⚠️ Please select an image first.")
            return

        if not detect_face_in_image(selected_file["path"]):
            show_error("❌ No face detected. Please upload a clear face image.")
            return

        progress_ring.visible = True
        detect_button.disabled = True
        error_text.visible = False
        result_display.value = "Analyzing image for drug usage..."
        result_display.color = ft.Colors.BLUE_600
        result_container.bgcolor = ft.Colors.BLUE_50
        confidence_display.visible = False
        page.update()

        threading.Thread(target=send_image_thread, args=(selected_file["path"],), daemon=True).start()

    def send_image_thread(file_path):
        try:
            with open(file_path, "rb") as f:
                response = requests.post(SERVER_URL, files={"file": f}, timeout=30)

            if response.status_code == 200:
                data = response.json()
                result = data.get("result", "Unknown")
                confidence = data.get("confidence", None)
                
                print(f"Server response - Result: {result}, Confidence: {confidence}")

                if "no face" in result.lower():
                    update_result("❌ No Face Detected", 
                                 "No face detected in the image. Please upload a clear face image.", 
                                 is_error=True)
                    return

                if confidence is not None:
                    confidence_text = f"Confidence: {confidence * 100:.2f}%"
                else:
                    confidence_text = "Analysis complete"
                
                result_lower = result.lower()
                if "not a drug user" in result_lower:
                    display_text = "✅ Not a Drug User"
                    text_color = ft.Colors.GREEN_600
                    bg_color = ft.Colors.GREEN_50
                elif "drug user" in result_lower:
                    display_text = "⚠️ Drug User"
                    text_color = ft.Colors.RED_600
                    bg_color = ft.Colors.RED_50
                else:
                    display_text = f"🔍 {result}"
                    text_color = ft.Colors.BLUE_600
                    bg_color = ft.Colors.BLUE_50
                
                update_result(display_text, confidence_text, text_color, bg_color)
                
            else:
                error_msg = f"Server error (Status: {response.status_code})"
                try:
                    error_data = response.json()
                    error_msg = error_data.get('detail', error_msg)
                except:
                    pass
                update_result("❌ Server Error", error_msg, is_error=True)
                
        except requests.exceptions.ConnectionError:
            update_result("❌ Connection Error", 
                         "Could not connect to server. Please make sure the server is running.", 
                         is_error=True)
        except requests.exceptions.Timeout:
            update_result("❌ Timeout Error", 
                         "Server took too long to respond. Please try again.", 
                         is_error=True)
        except requests.exceptions.RequestException as e:
            update_result("❌ Connection Error", 
                         f"Could not connect to server: {str(e)}", 
                         is_error=True)
        except Exception as e:
            update_result("❌ Error", 
                         f"An unexpected error occurred: {str(e)}", 
                         is_error=True)

    def update_result(main_text, sub_text, text_color=ft.Colors.BLUE_600, bg_color=ft.Colors.BLUE_50, is_error=False):
        progress_ring.visible = False
        detect_button.disabled = False
        
        result_display.value = main_text
        result_display.color = text_color
        result_container.bgcolor = bg_color

        if not is_error and sub_text:
            confidence_display.value = sub_text
            confidence_display.color = ft.Colors.GREY_800
            confidence_display.visible = True
        else:
            confidence_display.visible = False

        if is_error:
            error_text.value = sub_text
            error_text.visible = True
        else:
            error_text.visible = False
            
        page.update()

    def show_error(message):
        error_text.value = message
        error_text.visible = True
        page.update()

    def select_different_image(e):
        upload_area.visible = True
        img_display.visible = False
        select_button.visible = False
        detect_button.disabled = True
        result_display.value = "Upload a face image and click detect"
        result_display.color = ft.Colors.GREY_600
        result_container.bgcolor = ft.Colors.GREY_100
        error_text.visible = False
        confidence_display.visible = False
        page.update()

    # Upload Area
    upload_area = ft.Container(
        content=ft.Column([
            ft.Icon(ft.Icons.CLOUD_UPLOAD, size=60, color=ft.Colors.BLUE_400),
            ft.Text("Click to upload face image", size=18, weight=ft.FontWeight.W_500),
            ft.Text("Supports: JPG, PNG, JPEG", size=14, color=ft.Colors.GREY_600),
            ft.Container(
                content=ft.Column([
                    ft.Icon(ft.Icons.INFO, size=16, color=ft.Colors.BLUE_600),
                    ft.Text("Must contain a clear human face", size=12, color=ft.Colors.BLUE_600),
                ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=5),
                margin=ft.margin.only(top=10)
            )
        ], 
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        spacing=10),
        width=400,
        height=300,
        border=ft.border.all(2, ft.Colors.BLUE_200),
        border_radius=15,
        bgcolor=ft.Colors.BLUE_50,
        alignment=ft.alignment.center,
        on_click=select_file
    )

    # Image Display
    img_display = ft.Container(
        content=ft.Image(
            width=350, 
            height=250, 
            fit=ft.ImageFit.CONTAIN,
            border_radius=10
        ),
        width=400,
        height=300,
        border=ft.border.all(2, ft.Colors.BLUE_200),
        border_radius=15,
        bgcolor=ft.Colors.WHITE,
        alignment=ft.alignment.center,
        visible=False
    )

    # Error message
    error_text = ft.Text(
        "", 
        size=14, 
        color=ft.Colors.RED_700,
        weight=ft.FontWeight.W_500,
        visible=False
    )

    # Buttons
    detect_button = ft.ElevatedButton(
        "Detect Drug Usage",
        icon=ft.Icons.SEARCH,
        style=ft.ButtonStyle(
            color=ft.Colors.WHITE,
            bgcolor=ft.Colors.BLUE_600,
            padding=20,
        ),
        disabled=True,
        width=200
    )

    select_button = ft.TextButton(
        "Select Different Image",
        icon=ft.Icons.CHANGE_CIRCLE,
        visible=False
    )

    progress_ring = ft.ProgressRing(visible=False)

    # Assign handlers
    file_picker.on_result = on_file_picked
    detect_button.on_click = detect_image
    select_button.on_click = select_different_image

    # Layout
    page.add(
        ft.Column([
            title,
            ft.Stack([
                upload_area,
                img_display
            ], width=400, height=300),
            select_button,
            ft.Container(
                content=ft.Row([
                    detect_button,
                    progress_ring
                ], alignment=ft.MainAxisAlignment.CENTER, spacing=10),
                margin=ft.margin.only(top=20, bottom=10)
            ),
            error_text,
            result_container,
        ], 
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        spacing=0)
    )

if __name__ == "__main__":
    ft.app(target=main)
