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

"""
Author: Brandon Araki
2D gridworld rendering framework using matplotlib
"""

class Viewer(object):
    def __init__(self, mode='human'):
        self.mode = mode
        self.fig = plt.figure()
        self.ax = plt.subplot(111)

        # shrink the plot so the legend will fit into the window
        box = self.ax.get_position()
        self.ax.set_position([box.x0, box.y0, box.width * 0.5, box.height])

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

        # self.ax.axis('off')
        plt.tick_params(
            # axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            left=False,
            right=False,
            top=False,         # ticks along the top edge are off
            labelbottom=False,
            labelleft=False) # labels along the bottom edge are off

        ##### LEGEND STUFF #####

        legend_keys = []
        for obj in env.obj_dict.values():
            legend_keys.append(mpatches.Patch(color=obj.color, label=obj.name))
        legend1 = self.ax.legend(
            loc='upper right', 
            shadow=False,
            handles=legend_keys,
            bbox_to_anchor=(1.45, 1.)
            )

        prop_legend_keys = []
        for prop in env.prop_dict.values():
            prop_legend_keys.append(mpatches.Patch(color=[1,1,1], label=prop.name + ' = ' + str(prop.value)))
        self.ax.legend(
            loc = 'lower center',
            shadow=False,
            handles=prop_legend_keys,
            bbox_to_anchor=(0.5, -0.5),
            ncol = 2
        )

        plt.gca().add_artist(legend1)
        plt.tight_layout()

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
