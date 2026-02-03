from setuptools import setup, Extension
from Cython.Build import cythonize

ext_modules = cythonize(
    Extension(
        name="recommenders.util.cython.tools",  # 这里是完整模块名
        sources=["recommenders/util/cython/tools.pyx"],  # 这里是源码路径
    )
)

setup(
    name="recommenders_cython_tools",
    ext_modules=ext_modules,
    packages=[
        "recommenders",
        "recommenders.util",
        "recommenders.util.cython",
    ],
)
