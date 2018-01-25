
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
from .patient import Patient

@click.group()
def cli():
    """
    CLI to process GENEActiv watch binary data.
    It can extract the raw data, calibrate it, calculate the wrist angles and more.
    """
    pass


@cli.command('binary')
@click.option('--intermediate/--no-intermediate',default=True, help="Determine if intermediate files should be created.")
@click.option('--target_directory','-t',default=os.path.join(os.getcwd(),"ActiwatchData"),help="The top directory for all the output of the application.")
@click.option('--endpoint','-e',type=click.Choice(['raw','calibrate','roll','angles','inactivity','compress']),default='compress',help="Process the file up to that step and then save progress.")
@click.option('--compress_minutes',type=int,default=5)
@click.argument('input_files', type=click.Path(exists=True),nargs=-1)
def binary(intermediate,target_directory,endpoint,compress_minutes,input_files):
    '''
    Takes binary files and extracts the data within them.
    '''
    bin_files=[i for i in input_files if i.endswith(".bin") ]
    click.echo("Binary files: {}".format(bin_files))
    click.echo("Endpoint: {}".format(endpoint))
    for i,f in enumerate(bin_files):
        # Create an Patient instance
        click.echo("Processing file: {}".format(f))
        tmp=Patient(path_binary=f,endpoint=endpoint,compress_minutes=compress_minutes)
        click.echo("Writing file: {}".format(f))
        tmp.write_inactivity(target_directory)
        tmp.write_dev_sleep(target_directory)

@cli.command('combine')
@click.option('--target_directory','-t',default=os.path.join(os.getcwd(),"ActiwatchData"),help="The top directory for all the output of the application.")
@click.option('--endpoint','-e',type=click.Choice(['raw','calibrate','roll','angles','inactivity','compress']),default='calibrate',help="Process the file up to that step and then save progress.")
@click.argument('input_files', type=click.Path(exists=True),nargs=2)
def combine(target_directory,endpoint,input_files):
    """
    Takes two binary files, processes them up to 'endpoint',
    combines the dataframes and then saves it to a file.
    """
    if len(input_files) != 2:
        click.echo("Need to give two binary files to join them.")
        return None
    f1=Patient(path_binary=input_files[0],endpoint=endpoint)
    f2=Patient(path_binary=input_files[1],endpoint=endpoint)
    combined=f1+f2

    # Output to file
    if not os.path.exists(target_directory):
        os.mkdir(target_directory)
    patient_name=f1.fn
    sub_dir=os.path.join(target_directory,patient_name)
    if not os.path.exists(sub_dir):
        os.mkdir(sub_dir)
    output_file=os.path.join(sub_dir,(patient_name+"___combined.csv"))
    combined.to_csv(output_file,date_format="%Y-%m-%d %H:%M:%S.%f")
