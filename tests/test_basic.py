
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
    
#from context import geneactivpy
import os
import sys
import pickle
PACKAGE_PARENT=os.path.abspath("..")
sys.path.append(PACKAGE_PARENT)

from geneactivpy.patient import Patient
from time import time

f0="../../Data/head_bin_file.bin"
f1="../../Data/EAT201__031959_2017-06-22 13-59-41.bin"
f2="../../Data/EAT201__031959_2017-07-06 11-16-27.bin"

if __name__=="__main__":
    print(os.getcwd())
    t0=time()
    p1_1=Patient(path_binary=f0)
    print("Inactivity at the end.")
    print(p1_1.inactivity.head(100))
    print(p1_1.inactivity.tail(100))
    p1_1.write_inactivity()
    p1_1.write_dev_sleep()
    t1=time()
    print("Shape of p1_1: {}".format(p1_1.latest_df.shape))
    print("File 1 took {} seconds to process.".format((t1-t0)))

    pickle.dump(p1_1.angles,open("example_angles.p",'wb'))

    #p1_2=Patient(path_binary=f2,endpoint='raw')
    #t2=time()
    #print("Shape of p1_2: {}".format(p1_2.latest_df.shape))
    #print("File 2 took {} seconds to process.".format((t2-t1)))
    #combined=p1_1+p1_2
    #t3=time()
    #print("Shape of combined: {}".format(combined.shape))
    #print("Addition/merger took {} seconds.".format((t3-t2)))
    #print("Head and tail of the df:\n{}\n\n{}".format(combined.head(5),combined.tail(5)))
    #print("Total processing time for getting raw and adding them: {} seconds".format((t3-t0)))
