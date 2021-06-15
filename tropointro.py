"""
Created on Tue Jun 15 22:09:17 2021

@author: mathew

"""

from manim import *
import numpy as np


class cycloneintro(Scene):
    def construct(self):
        title = Tex(r"So the cyclone actually rotates over the ocean")
        basel = Tex(r"(we will come to that part later)")
        self.play(
            Write(title),
        )
        self.wait()
        
        transform_title = Tex("Now we can set up 2D cyclonic circulation")        
        transform_title.to_corner(UP + LEFT)
        self.play(
            Transform(title, transform_title),
            LaggedStart(*[FadeOutAndShift(obj, direction=DOWN) for obj in basel]),
        )
        self.wait()
        circle = Circle()
        self.play(ShowCreation(circle))
        
        