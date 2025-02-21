import tensorflow as tf 
import tkinter as tk
from tkinter import Canvas
import numpy as np
from PIL import Image, ImageDraw
import cv2

model = tf.keras.models.load_model('mnist.h5')

window = tk.Tk()
window.title("Handwritten Digit Recognition")

canvas_width = 280
canvas_height = 280
canvas = Canvas(window, width=canvas_width, height=canvas_height, bg="white")
canvas.pack()

image = Image.new("L", (canvas_width, canvas_height), 255)
draw = ImageDraw.Draw(image)

def predict_digit():
    img = image.resize((28, 28))
    
    img = np.array(img)
    
    img = 255 - img
    
    img = img / 255.0
    
    img = img.reshape(1, 28, 28, 1)
    
    prediction = model.predict(img)
    
    predicted_digit = np.argmax(prediction)
    
    result_label.config(text=f"Predicted: {predicted_digit}")

def draw_digit(event):
    x, y = event.x, event.y
    canvas.create_oval(x, y, x+10, y+10, fill="black", width=2)
    draw.ellipse([x, y, x+10, y+10], fill="black")

def clear_canvas():
    canvas.delete("all")
    draw.rectangle((0, 0, canvas_width, canvas_height), fill="white")
    result_label.config(text="")

canvas.bind("<B1-Motion>", draw_digit)

predict_button = tk.Button(window, text="Predict", command=predict_digit)
predict_button.pack()
clear_button = tk.Button(window, text="Clear", command=clear_canvas)
clear_button.pack()

result_label = tk.Label(window, text="", font=("Arial", 16))
result_label.pack()

window.mainloop()