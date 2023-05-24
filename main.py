from PIL import ImageTk, Image, ImageDraw
import PIL
from tkinter import *
from utils import *

width = 200  # canvas width
height = 200  # canvas height
center = height // 2
white = (255, 255, 255)  # canvas back
predictor = load_model()


# GUI for drawing
class DrawingApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Draw")

        self.canvas = Canvas(self.master, width=width, height=height, bg='white')
        self.canvas.pack()

        self.delete_button = Button(self.master, text="Delete", command=self.delete)
        self.delete_button.pack(side=LEFT)

        self.save_button = Button(self.master, text="Save", command=self.save)
        self.save_button.pack(side=RIGHT)

        self.pred_button = Button(self.master, text="Predict", command=self.predict)
        self.pred_button.pack()

        self.image = PIL.Image.new("RGB", (width, height), "white")
        self.draw = PIL.ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        x1, y1 = (event.x - 3), (event.y - 3)
        x2, y2 = (event.x + 3), (event.y + 3)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", width=7)
        self.draw.line([x1, y1, x2, y2], fill="black", width=7)

    def delete(self):
        self.canvas.delete("all")
        self.image = PIL.Image.new("RGB", (width, height), "white")
        self.draw = PIL.ImageDraw.Draw(self.image)

    def save(self):
        filename = "user_input.jpg"
        self.image.save(filename)

    def predict(self):
        img = prepare_img(self.image)
        number = predict_number(img, predictor)
        self.delete()
        self.popup_bonus(number)
        # print(number)

    def popup_bonus(self, num):
        win = Toplevel()
        win.wm_title("Predicted")
        win.geometry('100x90')
        l = Label(win, text=num, font=('Arial', 50))
        l.pack()
        win.after(2000, lambda: win.destroy())


def main():
    model = load_model()
    root = Tk()
    app = DrawingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
