import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageOps
import colorsys
import numpy as np
import os
import cv2
import threading

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class PSXTextureApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("PSX Texture Maker")
        self.geometry("1200x900")

        self.image = None
        self.image_path = None
        self.video_path = None
        self.output_frames = []
        self.preview_size = 450  # Increased from 300 for bigger previews

        self.build_ui()

    def build_ui(self):
        top_frame = ctk.CTkFrame(self)
        top_frame.pack(pady=10, fill="x")

        self.load_button = ctk.CTkButton(top_frame, text="Load Media", command=self.load_media)
        self.load_button.pack(side="left", padx=20)

        self.export_button = ctk.CTkButton(top_frame, text="Export", command=self.run_export_thread)
        self.export_button.pack(side="left", padx=20)

        # Before/After Preview Frame
        preview_frame = ctk.CTkFrame(self)
        preview_frame.pack(pady=20)

        self.before_preview_label = ctk.CTkLabel(preview_frame, text="Before")
        self.before_preview_label.pack(side="left", padx=10)

        self.after_preview_label = ctk.CTkLabel(preview_frame, text="After")
        self.after_preview_label.pack(side="left", padx=10)

        # Export progress
        self.progress_frame = ctk.CTkFrame(self)
        self.progress_frame.pack(pady=10)
        self.progress_bar = ctk.CTkProgressBar(self.progress_frame, orientation="horizontal", width=600)
        self.progress_bar.pack(side="left", padx=10)
        self.progress_label = ctk.CTkLabel(self.progress_frame, text="0%")
        self.progress_label.pack(side="left")

        control_frame = ctk.CTkFrame(self)
        control_frame.pack(pady=10, fill="x")

        # Resolution slider
        resolution_row = ctk.CTkFrame(control_frame)
        resolution_row.pack(pady=5)
        ctk.CTkLabel(resolution_row, text="Resolution (Max Dimension)").pack(side="left")
        self.resolution_slider = ctk.CTkSlider(resolution_row, from_=64, to=512, number_of_steps=14, command=self.update_preview)
        self.resolution_slider.set(256)
        self.resolution_slider.pack(side="left", padx=5)
        self.resolution_value = ctk.CTkEntry(resolution_row, width=80)
        self.resolution_value.insert(0, "256x256")
        self.resolution_value.pack(side="left")

        # Posterize slider
        posterize_row = ctk.CTkFrame(control_frame)
        posterize_row.pack(pady=5)
        ctk.CTkLabel(posterize_row, text="Posterize Bits").pack(side="left")
        self.posterize_slider = ctk.CTkSlider(posterize_row, from_=1, to=8, number_of_steps=7, command=self.update_preview)
        self.posterize_slider.set(2)
        self.posterize_slider.pack(side="left", padx=5)
        self.posterize_value = ctk.CTkEntry(posterize_row, width=50)
        self.posterize_value.insert(0, "2")
        self.posterize_value.pack(side="left")

        # Hue slider
        hue_row = ctk.CTkFrame(control_frame)
        hue_row.pack(pady=5)
        ctk.CTkLabel(hue_row, text="Hue Shift").pack(side="left")
        self.hue_slider = ctk.CTkSlider(hue_row, from_=0, to=360, number_of_steps=360, command=self.update_preview)
        self.hue_slider.set(0)
        self.hue_slider.pack(side="left", padx=5)
        self.hue_value = ctk.CTkEntry(hue_row, width=50)
        self.hue_value.insert(0, "0")
        self.hue_value.pack(side="left")

        # Dithering intensity slider
        dither_row = ctk.CTkFrame(control_frame)
        dither_row.pack(pady=5)
        ctk.CTkLabel(dither_row, text="Dithering Intensity").pack(side="left")
        self.dither_slider = ctk.CTkSlider(dither_row, from_=0, to=1.0, number_of_steps=10, command=self.update_preview)
        self.dither_slider.set(0.5)
        self.dither_slider.pack(side="left", padx=5)
        self.dither_value = ctk.CTkEntry(dither_row, width=50)
        self.dither_value.insert(0, "0.5")
        self.dither_value.pack(side="left")

    def run_export_thread(self):
        thread = threading.Thread(target=self.export_media)
        thread.start()

    def load_media(self):
        path = filedialog.askopenfilename(filetypes=[("Media files", "*.png;*.jpg;*.jpeg;*.bmp;*.mp4;*.avi")])
        if not path:
            return

        ext = os.path.splitext(path)[1].lower()
        if ext in ['.png', '.jpg', '.jpeg', '.bmp']:
            self.image = Image.open(path).convert("RGB")
            self.image_path = path
            self.video_path = None
            self.update_preview()
        elif ext in ['.mp4', '.avi']:
            self.image = None
            self.video_path = path
            self.image_path = path
            self.update_preview(video_preview=True)
        else:
            messagebox.showerror("Invalid file", "Unsupported media type.")

    def export_media(self):
        self.export_button.configure(state="disabled")
        self.load_button.configure(state="disabled")
        self.progress_bar.set(0)
        self.progress_label.configure(text="0%")

        original_name = os.path.splitext(os.path.basename(self.image_path))[0]
        default_filename = original_name + "-psx"

        if self.image:
            save_path = filedialog.asksaveasfilename(defaultextension=".png", initialfile=default_filename)
            if save_path:
                processed = self.process_image(self.image)
                processed.save(save_path)

        elif self.video_path:
            cap = cv2.VideoCapture(self.video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            save_path = filedialog.asksaveasfilename(defaultextension=".mp4", initialfile=default_filename)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            ret, frame = cap.read()
            if not ret:
                self.export_button.configure(state="normal")
                self.load_button.configure(state="normal")
                return

            h, w = self.get_scaled_size(frame.shape[1], frame.shape[0], int(self.resolution_slider.get()))
            out = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img = self.process_image(img)
                img = img.resize((w, h), Image.NEAREST)
                frame_out = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                out.write(frame_out)
                count += 1

                progress = count / total_frames
                self.progress_bar.set(progress)
                self.progress_label.configure(text=f"{int(progress * 100)}%")

            cap.release()
            out.release()
            messagebox.showinfo("Done", "Video exported successfully.")

        self.export_button.configure(state="normal")
        self.load_button.configure(state="normal")
        self.progress_label.configure(text="Done")

    def get_scaled_size(self, width, height, max_size):
        ratio = min(max_size / width, max_size / height)
        return int(width * ratio), int(height * ratio)

    def apply_ordered_dithering(self, img, intensity):
        # Bayer 4x4 matrix scaled 0-255
        bayer = np.array([
            [  0, 128,  32, 160],
            [192,  64, 224,  96],
            [ 48, 176,  16, 144],
            [240, 112, 208,  80]
        ], dtype=np.uint8)

        img = img.convert("RGB")
        arr = np.array(img).astype(np.float32)
        h, w, _ = arr.shape

        # Tile Bayer matrix to image size
        threshold_map = np.tile(bayer, (h // 4 + 1, w // 4 + 1))[:h, :w]

        # Normalize and scale threshold map based on intensity
        threshold_map = (threshold_map / 255.0 - 0.5) * 255.0 * intensity

        # Add threshold map to all RGB channels
        dithered = np.clip(arr + threshold_map[:, :, np.newaxis], 0, 255).astype(np.uint8)
        return Image.fromarray(dithered)

    def process_image(self, img):
        max_size = int(self.resolution_slider.get())

        poster_bits = int(self.posterize_slider.get())
        self.posterize_value.delete(0, "end")
        self.posterize_value.insert(0, str(poster_bits))

        hue_angle = int(self.hue_slider.get())
        self.hue_value.delete(0, "end")
        self.hue_value.insert(0, str(hue_angle))

        dither_intensity = float(self.dither_slider.get())
        self.dither_value.delete(0, "end")
        self.dither_value.insert(0, f"{dither_intensity:.2f}")

        img = ImageOps.exif_transpose(img)
        w, h = img.size
        new_w, new_h = self.get_scaled_size(w, h, max_size)
        img = img.resize((new_w, new_h), Image.NEAREST)
        img = ImageOps.posterize(img, poster_bits)
        img = self.shift_hue(img, hue_angle)

        if dither_intensity > 0:
            img = self.apply_ordered_dithering(img, dither_intensity)

        self.resolution_value.delete(0, "end")
        self.resolution_value.insert(0, f"{new_w}x{new_h}")
        return img

    def update_preview(self, _=None, video_preview=False):
        if self.image:
            original = self.image.copy()
            processed = self.process_image(self.image)

            original_preview = original.resize(self.get_scaled_size(*original.size, self.preview_size), Image.NEAREST)
            processed_preview = processed.resize(self.get_scaled_size(*processed.size, self.preview_size), Image.NEAREST)

            self.tk_before = ImageTk.PhotoImage(original_preview)
            self.tk_after = ImageTk.PhotoImage(processed_preview)

            self.before_preview_label.configure(image=self.tk_before, text="")
            self.after_preview_label.configure(image=self.tk_after, text="")

        elif self.video_path and video_preview:
            cap = cv2.VideoCapture(self.video_path)
            ret, frame = cap.read()
            cap.release()
            if ret:
                original = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                processed = self.process_image(original)

                original_preview = original.resize(self.get_scaled_size(*original.size, self.preview_size), Image.NEAREST)
                processed_preview = processed.resize(self.get_scaled_size(*processed.size, self.preview_size), Image.NEAREST)

                self.tk_before = ImageTk.PhotoImage(original_preview)
                self.tk_after = ImageTk.PhotoImage(processed_preview)

                self.before_preview_label.configure(image=self.tk_before, text="")
                self.after_preview_label.configure(image=self.tk_after, text="")

    def shift_hue(self, img, angle):
        np_img = np.array(img).astype(np.float32) / 255.0
        r, g, b = np_img[:, :, 0], np_img[:, :, 1], np_img[:, :, 2]
        hls = np.vectorize(colorsys.rgb_to_hls)(r, g, b)
        h = (hls[0] + angle / 360.0) % 1.0
        new_rgb = np.vectorize(colorsys.hls_to_rgb)(h, hls[1], hls[2])
        result = np.dstack((new_rgb[0], new_rgb[1], new_rgb[2])) * 255.0
        return Image.fromarray(result.astype(np.uint8))


if __name__ == "__main__":
    app = PSXTextureApp()
    app.mainloop()
