from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk

import datasets


class Labeler:
    def __init__(self, root, start=0):
        root.title('Labler')

        main_frame = ttk.Frame(root, padding='3 3 12 12')
        main_frame.grid(column=0, row=0, sticky='n w e s')

        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        self.crash_file = 'crash_frames.txt'

        self.index = start
        self.images = datasets.ordered_frames()
        self.tk_image = ImageTk.PhotoImage(Image.open(self.images[self.index]))

        self.image_count = len(self.images)
        self.image_count_text = f'{self.index}/{self.image_count}'

        self.text_label = ttk.Label(main_frame, text=f'{self.images[self.index]} {self.image_count_text}')
        self.text_label.grid(column=2, row=5, sticky='e')

        self.image_label = ttk.Label(main_frame)
        self.image_label['image'] = self.tk_image
        self.image_label.grid(column=2, row=1, rowspan=4, sticky='n')

        ttk.Button(main_frame, text="next", command=self._next).grid(column=1, row=1) #, sticky='w')
        ttk.Button(main_frame, text="prev", command=self._prev).grid(column=1, row=2) #, sticky='w')
        ttk.Button(main_frame, text="crash", command=self.crash).grid(column=1, row=3) #, sticky='w')
        ttk.Button(main_frame, text="undo", command=self.undo).grid(column=1, row=4) #, sticky='w')

        root.bind("<Right>", self._next)
        root.bind("<Left>", self._prev)
        root.bind('<c>', self.crash)
        root.bind('<u>', self.undo)

    def _next(self, *args):
        self.index += 1
        self.tk_image = ImageTk.PhotoImage(Image.open(self.images[self.index]))
        self.image_label['image'] = self.tk_image
        self.image_count_text = f'{self.index}/{self.image_count}'
        self.text_label['text'] = f'{self.images[self.index]} {self.image_count_text}'
        self.text_label.grid(column=2, row=5, sticky='e')

    def _prev(self, *args):
        self.index -= 1
        self.tk_image = ImageTk.PhotoImage(Image.open(self.images[self.index]))
        self.image_label['image'] = self.tk_image
        self.image_count_text = f'{self.index}/{self.image_count}'
        self.text_label['text'] = f'{self.images[self.index]} {self.image_count_text}'
        self.text_label.grid(column=2, row=5, sticky='e')

    def crash(self, *args):
        with open(self.crash_file, 'a+') as f:
            f.write(f'{self.images[self.index]}\n')

    def undo(self, *args):
        with open(self.crash_file, 'r') as f:
            lines = f.readlines()
        with open(self.crash_file, 'w') as f:
            for line in lines:
                if line.strip('\n') != str(self.images[self.index]):
                    f.write(line)

# 2017 is where crash starts
# 2036 is where first change of seeing crash ends (press comes back down)
root = tk.Tk()
Labeler(root, 2016)
root.mainloop()
