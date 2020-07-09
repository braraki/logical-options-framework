"""
2D gridworld rendering framework using matplotlib
"""
from __future__ import division
import os
import sys
import time

import math
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches

from celluloid import Camera

class Viewer(object):
    def __init__(self, mode='human'):
        self.mode = mode
        self.fig = plt.figure()
        self.ax = plt.subplot(111)

        # shrink the plot so the legend will fit into the window
        box = self.ax.get_position()
        self.ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        self.camera = None
        if mode=='anim':
            self.camera = Camera(self.fig)

    def render(self, env):
        # get non-whitespace in the domain
        nonwhitespace = (np.sum(env.obj_state, -1) > 0).astype(float)
        # make a map of the whitespace
        whitespace = np.ones(env.dom_size) - nonwhitespace
        # add whitespace as an extra layer to env.obj_state
        obj_state = np.append(env.obj_state, whitespace[...,None], axis=-1)
        # add whitespace color as an extra layer to env.color_array
        white = np.array([[1, 1, 1]])
        color_array = np.append(env.color_array, white, axis=0)

        # multiplying the domain, (X x Y x O) with (O x 3)
        # to get an RGB array (X x Y x 3)
        image = np.matmul(obj_state, color_array)
        # flip the y axis to graph it naturally
        image = np.flip(image, axis=1)
        # transpose x and y coordinates to graph naturally
        image = np.transpose(image, [1, 0, 2])

        implot = plt.imshow(image)

        ##### LEGEND STUFF #####

        legend_keys = []
        for obj in env.obj_dict.values():
            legend_keys.append(mpatches.Patch(color=obj.color, label=obj.name))
        legend1 = self.ax.legend(
            loc='upper right', 
            shadow=False,
            handles=legend_keys,
            bbox_to_anchor=(1.35, 1.)
            )

        prop_legend_keys = []
        for prop in env.prop_dict.values():
            prop_legend_keys.append(mpatches.Patch(color=[1,1,1], label=prop.name + ' = ' + str(prop.value)))
        self.ax.legend(
            loc = 'lower center',
            shadow=False,
            handles=prop_legend_keys,
            bbox_to_anchor=(0.5, -0.3),
            ncol = 2
        )

        plt.gca().add_artist(legend1)

        if self.mode == 'human':
            plt.draw()
            plt.waitforbuttonpress(0)
            plt.cla()
            # plt.close(self.fig)
        elif self.mode == 'fast':
            plt.draw()
            plt.pause(0.01)
            plt.cla()
        elif self.mode=='anim':
            self.camera.snap()

        return self.camera

    def render_rrt(self, env, path):
        # get non-whitespace in the domain
        nonwhitespace = (np.sum(env.obj_state, -1) > 0).astype(float)
        # make a map of the whitespace
        whitespace = np.ones(env.dom_size) - nonwhitespace
        # add whitespace as an extra layer to env.obj_state
        obj_state = np.append(env.obj_state, whitespace[...,None], axis=-1)
        # add whitespace color as an extra layer to env.color_array
        white = np.array([[1, 1, 1]])
        color_array = np.append(env.color_array, white, axis=0)

        # multiplying the domain, (X x Y x O) with (O x 3)
        # to get an RGB array (X x Y x 3)
        image = np.matmul(obj_state, color_array)
        # flip the y axis to graph it naturally
        image = np.flip(image, axis=1)
        # transpose x and y coordinates to graph naturally
        image = np.transpose(image, [1, 0, 2])

        implot = plt.imshow(image)

        if path is not None:
            plt.plot([x for (x, y) in path], [7-y for (x, y) in path], '-r')

        ##### LEGEND STUFF #####

        legend_keys = []
        for obj in env.obj_dict.values():
            legend_keys.append(mpatches.Patch(color=obj.color, label=obj.name))
        legend1 = self.ax.legend(
            loc='upper right', 
            shadow=False,
            handles=legend_keys,
            bbox_to_anchor=(1.35, 1.)
            )

        prop_legend_keys = []
        for prop in env.prop_dict.values():
            prop_legend_keys.append(mpatches.Patch(color=[1,1,1], label=prop.name + ' = ' + str(prop.value)))
        self.ax.legend(
            loc = 'lower center',
            shadow=False,
            handles=prop_legend_keys,
            bbox_to_anchor=(0.5, -0.3),
            ncol = 2
        )

        plt.gca().add_artist(legend1)

        if self.mode == 'human':
            plt.draw()
            plt.waitforbuttonpress(0)
            plt.cla()
            # plt.close(self.fig)
        elif self.mode == 'fast':
            plt.draw()
            plt.pause(0.01)
            plt.cla()
        elif self.mode=='anim':
            self.camera.snap()

        return self.camera

# class Viewer(object):
#     def __init__(self, width, height, legend_loc='upper right', display=None):
#         self.width = width
#         self.height = height
        
#         self.fig, self.ax = plt.subplots()
#         self.legend = self.ax.legend(loc=legend_loc, shadow=False)


#     def close(self):
#         plt.close()

#     def window_closed_by_user(self):
#         self.isopen = False

#     def set_bounds(self, left, right, bottom, top):
#         assert right > left and top > bottom
#         scalex = self.width/(right-left)
#         scaley = self.height/(top-bottom)
#         self.transform = Transform(
#             translation=(-left*scalex, -bottom*scaley),
#             scale=(scalex, scaley))

#     """ props is a list of (x,y, [rgb]) coordinates
#     """
#     def add_propositions(self, props, prop_names):
#         return

#     """ traj_x is a list of x coordinates
#         color = 'b', 'r', 'g', etc
#     """
#     def add_trajectory(self, traj_x, traj_y, color, traj_name):
#         self.ax.plot(traj_x, traj_y, c=color, label='')

#     def add_point(self):
#         return

#     def render(self, return_rgb_array=False):
#         implot = plt.imshow(dom, cmap="Greys_r")
#         return

#     def draw_array(self):
#         return

#     def __del__(self):
#         self.close()

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