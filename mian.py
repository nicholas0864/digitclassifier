import numpy as np
import cv2
import tkinter as tk
from PIL import Image, ImageDraw
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset & train model
digits = datasets.load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)
model = SVC()
model.fit(X_train, y_train)
print("Model trained successfully!")


# GUI for drawing digits
class DigitRecognizer:
    def __init__(self, root):
        self.root = root
        self.canvas = tk.Canvas(root, width=280, height=280, bg="white")
        self.canvas.pack()
        self.button_predict = tk.Button(root, text="Predict", command=self.predict_digit)
        self.button_predict.pack()
        self.button_clear = tk.Button(root, text="Clear", command=self.clear_canvas)
        self.button_clear.pack()
        self.canvas.bind("<B1-Motion>", self.draw)
        self.image = Image.new("L", (280, 280), 255)
        self.draw_obj = ImageDraw.Draw(self.image)
        self.result_label = tk.Label(self.root, text="Prediction: ", font=("Arial", 16))
        self.result_label.pack()


    def draw(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x-8, y-8, x+8, y+8, fill="black", outline="black")
        self.draw_obj.ellipse([x-8, y-8, x+8, y+8], fill="black")

    def preprocess_image(self):
        img_resized = self.image.resize((8, 8), Image.Resampling.LANCZOS)
        img_array = np.array(img_resized)

        print("Raw Image Array:", img_array)  # Debugging step

        img_array = 16 - (img_array // 16)  # Normalize pixel values (0-16 scale)
        return img_array.flatten().reshape(1, -1)


    def predict_digit(self):
        self.root.update_idletasks()  # Ensures UI updates before processing
        print("Predict button clicked!")  # Debugging step 1
        img_vector = self.preprocess_image()
        
        print("Processed Image Vector:", img_vector)  # Debugging step 2
        prediction = model.predict(img_vector)
        
        print(f"Predicted Digit: {prediction[0]}")  # Debugging step 3
        self.result_label.config(text=f"Prediction: {prediction[0]}")



    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), 255)
        self.draw_obj = ImageDraw.Draw(self.image)

# Run the GUI
root = tk.Tk()
digit_recognizer = DigitRecognizer(root)
root.mainloop()
