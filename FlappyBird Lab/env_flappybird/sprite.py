import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.classic_control import rendering

import pyglet
from pyglet.gl import *


class Sprite(rendering.Geom):
    def __init__(self, image, width, height):
        super().__init__()
        self.width = width
        self.height = height
        self.img = pyglet.sprite.Sprite( img=image )
        self.flip = False

    def render1(self):
        self.img.draw()
    
    def __del__(self):
        del self.img


class ShadowedText(rendering.Geom):
    def __init__(self, text, x, y, width=100, height=100, font_name='Arial', bold=True, font_size=60, color=(255, 255, 255, 255) ):
        super().__init__()

        self.width = width
        self.height = height

        self.label1 = pyglet.text.Label(text,
        font_name=font_name,
        bold=bold,
        font_size=font_size,
        color=color,
        x=x, y=y,
        anchor_x='center', anchor_y='center')

        self.label2 = pyglet.text.Label(text,
        font_name='Arial',
        color=(0, 0, 0, 128),
        bold=bold,
        font_size=font_size,
        x=x+5, y=y-5,
        anchor_x='center', anchor_y='center')

        self.flip = False
        self.visible = True

    def render1(self):
        if self.visible:
            self.label2.draw()
            self.label1.draw()

    def set_text(self, text):

        self.label1.text = text
        self.label2.text = text

    def __del__(self):
        del self.label1
        del self.label2
