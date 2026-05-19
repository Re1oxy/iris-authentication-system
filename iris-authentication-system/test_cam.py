import cv2
from PIL import Image, ImageTk
import tkinter as tk

root = tk.Tk()
root.geometry('640x480')
cap = cv2.VideoCapture(0)
lbl = tk.Label(root)
lbl.pack()

def update():
    ret, frame = cap.read()
    if ret:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(rgb))
        lbl.config(image=img)
        lbl.image = img
    root.after(33, update)

update()
root.mainloop()