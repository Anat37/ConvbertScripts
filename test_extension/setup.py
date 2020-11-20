from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='example_module',
    ext_modules=[
        CUDAExtension('example_module', [
            'example.cpp',
            #'example_kernel.cu',
            'dynamicconv_cuda_forward.cu',
            'dynamicconv_cuda_backward.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })