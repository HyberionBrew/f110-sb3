from setuptools import setup, find_packages

setup(
    name='f110_rl',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # List of dependencies your project needs to run.
        # For example: 'requests', 'numpy', 'pandas', etc.
    ],
    author='Fabian Kresse',
    author_email='your.email@example.com',
    description='A brief description of your project',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/your_project_name',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.8',
)
