#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Basic user interface in tkinter.

Author: Daniel Dantas
Last edited: July 2018
"""

# python 3
import tkinter as tk

#python 2
#import Tkinter as tk

import my
import math as m
import os

from PIL import Image
from PIL import ImageTk

import ImgCanvas as ic

WIN_TITLE = "Kernel"

class WinKernel(tk.Frame):

    ## Object constructor.
    #  @param self The object pointer.
    #  @param root The object root, usualy created by calling tkinter.Tk().
    def __init__(self, root, canvas=None):

        self.root = root
        self.top = tk.Toplevel(self.root)
        self.top.title(WIN_TITLE + " - parameters")

        self.frame = tk.Frame.__init__(self, self.top)

        self.canvas = canvas
        self.image = self.canvas.get_image()

        self.l1 = tk.Label(self.top, text="Width", padx=0)
        self.l1.grid(row=0, column=0, ipady=5)

        self.e1 = tk.Entry(self.top)
        self.e1.grid(row=0, column=1, ipady=5)

        self.l2 = tk.Label(self.top, text="Height", padx=00)
        self.l2.grid(row=1, column=0, ipady=5)

        self.e2 = tk.Entry(self.top)
        self.e2.grid(row=1, column=1, ipady=5)

        self.b0 = tk.Button(self.top, text="Change size", command=self.cb_ok)
        self.b0.grid(row=2, column=0, ipady=5)

        self.b1 = tk.Button(self.top, text="Ok", command=self.cb_ok)
        self.b1.grid(row=3, column=0, ipady=5)

        self.b2 = tk.Button(self.top, text="Cancel", command=self.cb_cancel)
        self.b2.grid(row=3, column=1, ipady=5)

        #self.f1.pack(side=tk.BOTTOM)

    def cb_resize(self, event=None):
        print("%f, %d" % (self.s1.get(), self.s2.get()))
        r = self.s1.get()
        m = self.s2.get()
        result = my.contrast(self.image, r, m)
        self.canvas.set_preview(result)

    def cb_ok(self, event=None):
        self.canvas.ok_preview()
        self.top.destroy()

    def cb_cancel(self, event=None):
        self.canvas.cancel_preview()
        self.top.destroy()

        


############################################################
# Main function
############################################################

if __name__ == "__main__":

  root = tk.Tk()
  root.rowconfigure(0, weight=1)
  root.columnconfigure(0, weight=1)
  root.title(WIN_TITLE)
  root.minsize(600, 270)

  """  
  win = tk.Toplevel(root)

  root.frame_main = ic.ImgCanvas(root)
  root.frame_main.grid(row=0, column=0, stick='nswe', ipadx=5)

  root.frame_toolbox = tk.Frame(root, bg='orange')
  root.frame_toolbox.grid(row=0, column=1, stick='nswe', ipadx=5)

  #win.title = "ImgCanvas"
  img = my.imread("images/lena.tiff")
  root.frame_main.set_image(img)
  """

  img = my.imread("images/lena.tiff")
  root.ic = ic.ImgCanvas(root)
  root.ic.set_image(img)

  app = WinKernel(root, root.ic)
  app.mainloop()

