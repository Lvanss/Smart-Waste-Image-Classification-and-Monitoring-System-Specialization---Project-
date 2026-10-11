import mimetypes
import requests
import flet as ft

API_URL = "http://127.0.0.1:8000/predict"


def main(page: ft.Page):
    page.title = "Smart Waste Classification System"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 30
    page.window.width = 800
    page.window.height = 650

    selected_file = {"path": None, "name": None}

    title = ft.Text(
        "Smart Waste Classification",
        size=28,
        weight=ft.FontWeight.BOLD,
    )

    subtitle = ft.Text(
        "Upload a waste image to classify it into one of six categories.",
        size=14,
    )

    file_label = ft.Text("No image selected", size=14)
    status_label = ft.Text("Ready", size=14)
    result_label = ft.Text(
        "Your prediction will appear here.",
        size=20,
        weight=ft.FontWeight.BOLD,
    )
    confidence_label = ft.Text("")
    latency_label = ft.Text("")

    def show_message(message, color=None):
        status_label.value = message
        if color is not None:
            status_label.color = color
        page.update()

    async def choose_image(e):
        files = await ft.FilePicker().pick_files(
            allow_multiple=False,
            file_type=ft.FilePickerFileType.IMAGE,
        )

        if files:
            selected_file["path"] = files[0].path
            selected_file["name"] = files[0].name

            file_label.value = f"Selected: {files[0].name}"
            result_label.value = "Ready to classify"
            confidence_label.value = ""
            latency_label.value = ""
            show_message("Image selected.")

    def classify_image(e):
        image_path = selected_file["path"]

        if not image_path:
            show_message("Please select an image first.", "red")
            return

        try:
            show_message("Classifying image. Please wait...")

            with open(image_path, "rb") as image_file:
                response = requests.post(
                    API_URL,
                    files={
                        "file": (
                            selected_file["name"],
                            image_file,
                            mimetypes.guess_type(selected_file["name"])[0]
                            or "application/octet-stream",
                        )
                    },
                    timeout=60,
                )

            if response.status_code != 200:
                try:
                    detail = response.json().get(
                        "detail", "The API returned an error."
                    )
                except ValueError:
                    detail = "The API returned an error."

                show_message(f"Classification failed: {detail}", "red")
                return

            data = response.json()

            result_label.value = (
                f"Predicted class: {data['predicted_class'].title()}"
            )
            confidence_label.value = (
                f"Confidence: {data['confidence_percent']:.2f}%"
            )
            latency_label.value = (
                f"Inference latency: "
                f"{data['inference_latency_ms']:.2f} ms"
            )

            show_message("Classification completed successfully.")

        except requests.exceptions.ConnectionError:
            show_message(
                "Cannot connect to FastAPI. Make sure the API server is running.",
                "red",
            )

        except requests.exceptions.Timeout:
            show_message("The prediction request timed out.", "red")

        except (OSError, KeyError, ValueError) as error:
            show_message(f"Unable to process result: {error}", "red")

    page.add(
        ft.Column(
            controls=[
                title,
                subtitle,
                ft.Divider(),
                ft.Text(
                    "1. Select a waste image",
                    size=18,
                    weight=ft.FontWeight.BOLD,
                ),
                ft.Button(
                    "Choose Image",
                    icon=ft.Icons.UPLOAD_FILE,
                    on_click=choose_image,
                ),
                file_label,
                ft.Text(
                    "2. Run classification",
                    size=18,
                    weight=ft.FontWeight.BOLD,
                ),
                ft.Button(
                    "Classify Image",
                    icon=ft.Icons.SEARCH,
                    on_click=classify_image,
                ),
                ft.Divider(),
                ft.Text(
                    "Classification Result",
                    size=18,
                    weight=ft.FontWeight.BOLD,
                ),
                result_label,
                confidence_label,
                latency_label,
                ft.Divider(),
                status_label,
                ft.Text(
                    "Classes: Cardboard • Glass • Metal • Paper • Plastic • Trash",
                    size=12,
                    color=ft.Colors.GREY_700,
                ),
            ],
            spacing=18,
            scroll=ft.ScrollMode.AUTO,
            expand=True,
        )
    )


if __name__ == "__main__":
    ft.run(main)