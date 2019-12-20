"""
2D gridworld rendering framework using matplotlib
"""
from __future__ import division
import os
import sys

import math
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Viewer(object):
    def __init__(self, width, height, legend_loc='upper right', display=None):
        self.width = width
        self.height = height
        
        self.fig, self.ax = plt.subplots()
        self.legend = self.ax.legend(loc=legend_loc, shadow=False)


    def close(self):
        self.window.close()

    def window_closed_by_user(self):
        self.isopen = False

    def set_bounds(self, left, right, bottom, top):
        assert right > left and top > bottom
        scalex = self.width/(right-left)
        scaley = self.height/(top-bottom)
        self.transform = Transform(
            translation=(-left*scalex, -bottom*scaley),
            scale=(scalex, scaley))

    """ props is a list of (x,y, [rgb]) coordinates
    """
    def add_propositions(self, props, prop_names):
        return

    """ traj_x is a list of x coordinates
        color = 'b', 'r', 'g', etc
    """
    def add_trajectory(self, traj_x, traj_y, color, traj_name):
        self.ax.plot(traj_x, traj_y, c=color, label='')

    def add_point(self):
        return

    def render(self, return_rgb_array=False):
        implot = plt.imshow(dom, cmap="Greys_r")
        return

    def draw_array(self):
        return

    def __del__(self):
        self.close()

# ================================================================

class SimpleImageViewer(object):
    def __init__(self, display=None, maxwidth=500):
        self.window = None
        self.isopen = False
        self.display = display
        self.maxwidth = maxwidth
    def imshow(self, arr):
        if self.window is None:
            height, width, _channels = arr.shape
            if width > self.maxwidth:
                scale = self.maxwidth / width
                width = int(scale * width)
                height = int(scale * height)
            self.window = pyglet.window.Window(width=width, height=height, 
                display=self.display, vsync=False, resizable=True)            
            self.width = width
            self.height = height
            self.isopen = True

            @self.window.event
            def on_resize(width, height):
                self.width = width
                self.height = height

            @self.window.event
            def on_close():
                self.isopen = False

        assert len(arr.shape) == 3, "You passed in an image with the wrong number shape"
        image = pyglet.image.ImageData(arr.shape[1], arr.shape[0], 
            'RGB', arr.tobytes(), pitch=arr.shape[1]*-3)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, 
            gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        texture = image.get_texture()
        texture.width = self.width
        texture.height = self.height
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        texture.blit(0, 0) # draw
        self.window.flip()
    def close(self):
        if self.isopen and sys.meta_path:
            # ^^^ check sys.meta_path to avoid 'ImportError: sys.meta_path is None, Python is likely shutting down'
            self.window.close()
            self.isopen = False

    def __del__(self):
        self.close()