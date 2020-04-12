from distutils.core import setup, Extension

ta_mod = Extension('momentum', sources=['src/momentum.c'], extra_compile_args=['-O3', '-mavx'])

setup(ext_modules=[ta_mod])
