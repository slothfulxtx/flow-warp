from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="flow_warp",
    packages=['flow_warp'],
    version="0.1.5",
    ext_modules=[
        CUDAExtension(
            name="flow_warp._C",
            sources=[
            "cuda_warp/warp.cu",
            "forward_warp.cu",
            "ext.cpp"] #,
            # extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]}
        )],
    cmdclass={
        'build_ext': BuildExtension
    }
)