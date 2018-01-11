
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
    
import os
import click

@click.group()
def repurpose():
    """
    Where the files have already been preprocessed and saved.
    """
    pass

@click.command()
@click.option('--intermediate/--no-intermediate',default=True, help="Determine if intermediate files should be created.")
@click.option('--target_directory','-t',default=os.path.join(os.getcwd(),"ActiwatchData"),help="The top directory for all the output of the application.")
@click.argument('binary_files', type=click.Path(exists=True),nargs=-1,help="Binary files that the application will process.")
def binary(intermediate,target_directory,binary_files):
    '''
    Takes binary files and extracts the data within them.
    '''
    click.echo("Number of arguments: {}".format(len(binary_files)))
    pass
    
