from setuptools import setup, find_packages

with open('requirements_gpu.txt') as file:
    REQUIRED_PACKAGES = file.read()

setup(name='dofaker',
      version='0.0',
      keywords=('face swap'),
      description='A simple face swap tool',
      url='https://github.com/justld/dofaker',
      author='justld',
      author_email='1207540056@qq.com',
      packages=find_packages(),
      include_package_data=True,
      platforms='any',
      install_requires=REQUIRED_PACKAGES,
      scripts=[],
      license='GPL 3.0',
      entry_points={'console_scripts': [
          'dofaker = web_ui:main',
      ]})
