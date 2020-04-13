from distutils.core import setup, Extension

ta_mod = Extension('momentum',
                   sources=['src/momentum.c'],
                   extra_compile_args=['-mavx'])

setup(name = 'fast_ta',
      version = 0.1,
      description = "Fast Technical Analysis Library Written In C",
      author = "Cristian Bicheru",
      author_email = "c.bicheru0@gmail.com",
      ext_modules=[ta_mod])
