from Cython.Distutils import build_ext
try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension

class FastTABuild(build_ext):
    def run(self):
        import numpy
        import detect_simd
        self.include_dirs.append(numpy.get_include())
        simd = detect_simd.detect()
        if not self.define:
            self.define = []
        if simd['AVX512'] == 1:
            self.define.append(('AVX512', '1'))
        elif simd['AVX'] == 1:
            self.define.append(('AVX', '1'))
        elif simd['SSE2'] == 1:
            self.define.append(('SSE2', '1'))
        build_ext.run(self)

common_backend = ['fast_ta/src/error_methods.c', 'fast_ta/src/funcs.c']
compile_args = ['-O2', '-ffast-math', '-march=native', '-mno-align-double',
                '-fomit-frame-pointer', '-frename-registers']
        
momentum_ext = Extension('fast_ta/momentum',
                   sources=['fast_ta/src/momentum.c', 'fast_ta/src/momentum_backend.c',
                            'fast_ta/src/parallel_momentum_backend.c']+common_backend,
                   extra_compile_args=compile_args)
volume_ext = Extension('fast_ta/volume',
                   sources=['fast_ta/src/volume.c', 'fast_ta/src/volume_backend.c']+common_backend,
                   extra_compile_args=compile_args)

setup(name = 'fast_ta',
      packages = ["fast_ta"],
      version = '0.1.3',
      license = 'MIT',
      license_file = "LICENSE.md",
      description = "Fast Technical Analysis Library Written In C",
      long_description = ("Fast TA is an optimized, high-level technical analysis library "
                          "used to compute technical indicators on financial datasets. "
                          "It is written entirely in C, and uses AVX vectorization as well. "
                          "Fast TA is built with the NumPy C API."),
      author = "Cristian Bicheru, Calder White",
      author_email = "c.bicheru0@gmail.com, calderwhite1@gmail.com",
      maintainer = "Cristian Bicheru, Calder White",
      maintainer_email ="c.bicheru0@gmail.com, calderwhite1@gmail.com",
      url = 'https://fast-ta.readthedocs.io/',
      download_url = 'https://github.com/cristian-bicheru/fast-ta/archive/v0.1.3.tar.gz',
      keywords = ['technical analysis', 'python3', 'numpy'],
      install_requires = [
          'numpy',
          'detect-simd'
      ],
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Financial and Insurance Industry',
        'Topic :: Office/Business :: Financial :: Investment',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
      ],
      project_urls = {
        'Documentation': 'https://fast-ta.readthedocs.io/',
        'Bug Reports': 'https://github.com/cristian-bicheru/fast-ta/issues',
        'Source': 'https://github.com/cristian-bicheru/fast-ta',
      },
      cmdclass = {'build_ext': FastTABuild},
      ext_modules=[momentum_ext, volume_ext])
