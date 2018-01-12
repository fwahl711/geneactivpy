
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
    
#from .context import geneactivpy
from ..geneactivpy.patient import Patient
from time import time

f1="../../Data/EAT201__031959_2017-06-22 13-59-41.bin"
f2="../../Data/EAT201__031959_2017-07-06 11-16-27.bin"

if __name__=="__main__":
    t0=time()
    p1_1=Patient(path_binary=f1,endpoint='raw')
    t1=time()
    print("Shape of p1_1: {}".format(p1_1.latest_df.shape))
    print("File 1 took {} seconds to process.".format((t1-t0)))
    p1_2=Patient(path_binary=f2,endpoint='raw')
    t2=time()
    print("Shape of p1_2: {}".format(p1_2.latest_df.shape))
    print("File 2 took {} seconds to process.".format((t2-t1)))
    combined=p1_1+p1_2
    t3=time()
    print("Shape of combined: {}".format(combined.shape))
    print("Addition/merger took {} seconds.".format((t3-t2)))
    print("Head and tail of the df:\n{}\n\n{}".format(combined.head(5),combined.tail(5)))
    print("Total processing time for getting raw and adding them: {} seconds".format((t3-t0)))
