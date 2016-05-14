import os
from .html import html
__all__ = ['html', 'mandelbrot_c']

codepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'mandelbrot.c')
mandelbrot_c = ''.join(open(codepath, 'r').readlines())

