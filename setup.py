from setuptools import setup, find_packages

setup(
    name="enterprise-ocr-app",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'streamlit>=1.22.0',
        'python-doctr[torch]>=0.6.0',
        'easyocr>=1.7.0',
        'pillow>=9.5.0',
        'opencv-python-headless>=4.7.0.72',
        'numpy>=1.23.5',
        'pandas>=1.5.3',
        'matplotlib>=3.7.1'
    ]
)