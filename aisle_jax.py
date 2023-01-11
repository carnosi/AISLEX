# -*- coding: utf-8 -*-
"""
aisle_jax.py: Handles learning entropy computational acceleration with JAX by Google

Prebuilt community packages of JAX are available on: https://github.com/cloudhan/jax-windows-builder
Other way you can also build jax on your own from https://github.com/google/jax

__doc__ using Sphnix Style
"""
# Copyright 2022 University Southern Bohemia

# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

__author__ = "Ondrej Budik"
__copyright__ = "<2023> <University Southern Bohemia> <Czech Technical University>"
__credits__ = ["Ivo Bukovsky"]

__license__ = "MIT (X11)"
__version__ = "1.0.0"
__maintainer__ = ["Ondrej Budik"]
__email__ = ["obudik@prf.jcu.cz"]
__status__ = "alpha"

__python__ = "3.8.0"

import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random


