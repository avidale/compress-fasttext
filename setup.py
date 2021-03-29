import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="compress-fasttext",
    version="0.1.0",
    author="David Dale",
    author_email="dale.david@mail.ru",
    description="A set of tools to compress gensim fasttext models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/avidale/compress-fasttext",
    packages=['compress_fasttext'],
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'gensim>=4.0.0',
        'numpy',
    ],
    extras_require={
        'full': [
            'scikit-learn',
            'pqkmeans',
        ],
    }
)
