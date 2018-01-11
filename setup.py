from setuptools import setup, find_packages

setup(
    name='geneactivpy',
    version='0.1',
    author='acs-monkey',
    packages=find_packages(),
    long_description='''
# geneactivpy

## Description


## Examples

## Repo Structure 

geneactivpy:
┣━ README.md
┣━ LICENSE
┣━ setup.py
┣━ geneactivpy:
┃   ┗━ __init__.py
┗━ tests:
    ┗━ test.py
''',
    license='''
#   Copyright (C) 2018  acs-monkey
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.
    ''',
    include_package_data=True,
    install_requires=[
        'Click',
    ],
    entry_points='''
        [console_scripts]
        geneactivpy=geneactivpy.__init__:main
    ''',
)