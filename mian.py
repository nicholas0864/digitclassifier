import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from collections import Counter

# Load dataset & train model
digits = datasets.load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42
)
# Adjust gamma if needed (the default might not work well for some cases)
model = SVC(kernel='rbf', gamma=0.001)
model.fit(X_train, y_train)
print("Model trained successfully!")
print("Training data label distribution:", Counter(y_train))
print("Test data label distribution:", Counter(y_test))

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
        # Create a PIL image with a white background
        self.image = Image.new("L", (280, 280), 255)
        self.draw_obj = ImageDraw.Draw(self.image)
        self.result_label = tk.Label(root, text="Prediction: ", font=("Arial", 16))
        self.result_label.pack()

    def draw(self, event):
        x, y = event.x, event.y
        radius = 8
        # Draw on both the canvas and the PIL image
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill="black", outline="black")
        self.draw_obj.ellipse([x - radius, y - radius, x + radius, y + radius], fill="black")

    def preprocess_image(self):
        # Resize the drawn image to 8x8 pixels (as in the training dataset)
        img_resized = self.image.resize((8, 8), Image.Resampling.LANCZOS)
        img_array = np.array(img_resized)
        print("Raw resized image array (8x8):\n", img_array)

        # Invert and scale pixel values so that:
        # White (255) becomes 0 and Black (0) becomes 16,
        # matching the sklearn digits dataset where digit strokes are darker.
        img_array = (255 - img_array) / 255.0 * 16
        print("Processed image array scaled to 0-16:\n", img_array)

        # Flatten to create a feature vector
        img_vector = img_array.flatten().reshape(1, -1)
        print("Flattened image vector:\n", img_vector)
        return img_vector

    def predict_digit(self):
        self.root.update_idletasks()  # Ensure UI updates before processing
        print("Predict button clicked!")
        img_vector = self.preprocess_image()
        prediction = model.predict(img_vector)
        print("Predicted Digit:", prediction[0])
        self.result_label.config(text=f"Prediction: {prediction[0]}")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), 255)
        self.draw_obj = ImageDraw.Draw(self.image)

# Run the GUI
root = tk.Tk()
root.title("Handwritten Digit Recognizer")
digit_recognizer = DigitRecognizer(root)
root.mainloop()
