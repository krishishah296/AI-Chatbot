from setuptools import setup, find_packages

setup(
    name="virtual_clinic_simulation",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "flask",
        "openai-whisper",
        "pyttsx3"
    ],
    description="A virtual clinic simulation application",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/virtual_clinic_simulation",
)
