import tkinter as tk
from tkinter import filedialog
import pickle
import numpy as np
import cv2
from ttkthemes import ThemedStyle

from PIL import Image, ImageTk

with open('model1.pkl','rb') as file:
    model = pickle.load(file)

# Define the function to process the uploaded image
def process_image():
    # Open the uploaded image
    image_path = filedialog.askopenfilename()
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (224, 224))

    # Normalize the pixel values
    resized_image = resized_image.astype('float32') / 255.0

    # Add an extra dimension to represent the batch size
    input_image = np.expand_dims(resized_image, axis=0)
    prediction = model.predict(np.array(input_image))
    if prediction[0] < 0.63 :
        output_label = "Authentic Image"
    else:
        output_label="Forged Image"
    result_label.configure(text=output_label)
    result_label.pack()
    # probability = prediction[0][0]
    # if probability < 0.5:
    #     print(f'{image_path} is classified as authentic with probability {probability:.4f}')
    # else:
    #     print(f'{image_path} is classified as forged with probability {probability:.4f}')

    # Perform image processing to determine if it's forged or real
    # Replace this with your own image processing and classification logic

    # Display the result in a message box
    # result = tk.messagebox.showinfo("Image Result", "Forged" if is_forged else "Real")

# Create the main Tkinter window
window = tk.Tk()
window.title("Image Uploader")
window.geometry("600x400")
window.configure(bg="silver")


label = tk.Label(window, text="Detection of Tampered Image",
width = 40, height = 3, borderwidth = 2, relief = "solid",
font = ("Lucida Console", 12, "bold"), background = 'yellow')
label.pack(pady=20)
image1 = Image.open("output.jpg")
image2 = Image.open("input.jpg")
small_size = (150, 150)
# Create a PhotoImage object
image1 = image1.resize(small_size)
photo = ImageTk.PhotoImage(image1)
label1 = tk.Label(window, image=photo)
x = 400  # x-coordinate of the desired location
y = 100   # y-coordinate of the desired location
label1.place(x=x, y=y)
image2 = image2.resize(small_size)
photo2 = ImageTk.PhotoImage(image2)
label2 = tk.Label(window, image=photo2)
x = 50  # x-coordinate of the desired location
y = 100   # y-coordinate of the desired location
label2.place(x=x, y=y)
#label3 = tk.Label(window, image=photo2)
#label3.pack(side=tk.RIGHT, padx=10, pady=10)

#label3.pack(anchor="w")
upload_button = tk.Button(window, text="Upload Image", command=process_image,bg="red", fg="white")
upload_button.place(relx=1, rely=1, anchor="center")
upload_button.pack(pady=20)
result_label=tk.Label(window,text="The Image is : ",width=40)
result_label.pack(pady=20)
# Run the Tkinter event loop
window.mainloop()
