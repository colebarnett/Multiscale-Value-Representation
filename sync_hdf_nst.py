# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 14:13:21 2022

@author: yz25424
"""

from os import path as ospath
# import sys
import numpy as np
import scipy as sp
# from scipy import io
import matplotlib.pyplot as plt
# import tables
import os
import time
import warnings
#import neurodsp

# BMI_FOLDER = r"C:\Users\crb4972\Desktop\bmi_python"
BMI_FOLDER = r"C:\Users\coleb\Desktop\Santacruz Lab\bmi_python"
NS_FOLDER = ospath.join(BMI_FOLDER,'riglib','ripple','pyns','pyns')
os.chdir(BMI_FOLDER)



# os.chdir(r"D:\dropbox\Dropbox\Samantha lab\BMI_Airport") # on windows
#os.chdir(r"/Users/zhaoyi/Dropbox/Samantha lab/BMI_Airport") # on mac
from riglib.ripple.pyns.pyns.nsexceptions import NeuroshareError, NSReturnTypes
import riglib.ripple.pyns.pyns.nsparser
from riglib.ripple.pyns.pyns.nsparser import ParserFactory
from riglib.ripple.pyns.pyns.nsentity import AnalogEntity, SegmentEntity, EntityType, EventEntity, NeuralEntity
from riglib.blackrock.brpylib import NsxFile
# os.chdir(r"D:\dropbox\Dropbox\Samantha lab\BMI_Airport\riglib\ripple\pyns\pyns) # on windows
#os.chdir(r"/Users/zhaoyi/Dropbox/Samantha lab/BMI_Airport/riglib/ripple/pyns/pyns") # on windows

os.chdir(NS_FOLDER)
from nsfile import NSFile
warnings.filterwarnings('ignore')

#%%
class nsyncHDF:
    """General class used to extract non-neural information from Ripple files
    to synchronize with behavioral data saved in linked HDF files.  
    """
    def __init__(self, filename):
        """Initialize new File instance.
        Parameters:
        filename -- relative path to wanted .ns5 file
        """
        self.name = ospath.basename(filename)[:-4]
        self.path = ospath.dirname(filename)

        # Analogsignals for digital events
        # Naming convention
        # 0 - 3  : SMA 1 - 4
        # 4 - 27 : Pin 1 - 24
        # 28 - 29: Audio 1 - 2 
        # Here we use Pin 1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19 (based on Arduino setup)
        self.pins_util = np.array([1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19]) + 3

        if filename[-4:]=='.ns5':
            self.nsfile = NsxFile(filename)
            self.output = self.nsfile.getdata() 
        else:
            raise Exception('Error: Not an .ns5 file')

    def extract_rows(self):
        """Create .mat synchronization file for synchronizing Ripple and behavioral data (saved in .hdf file).
        Parameters:
        Return:
        hdf_times -- dict, contains row numbers and corresponding time stamps
        """

        # Create dictionary to store synchronization data
        hdf_times = dict()
        hdf_times['row_number'] = []          # PyTable row number
        hdf_times['ripple_samplenumber'] = []    # Corresponding Ripple sample number
        hdf_times['ripple_dio_samplerate'] = []  # Sampling frequency of DIO signal recorded by Ripple system
        hdf_times['ripple_recording_start'] = [] # Ripple sample number when behavior recording begins

        signals = self.output['data']
        fs = self.output['samp_per_s']
        msgtype = signals[self.pins_util[8:], :]
        rownum = signals[self.pins_util[:8], :]

        # Convert to 0 or 1 integers (0 ~ 5000 mV from the recordings)
        rstart = (signals[22 + 3,:] > 2500).astype(int)
        strobe = (signals[20 + 3,:] > 2500).astype(int)
        msgtype = np.flip((msgtype > 2500).astype(int), axis = 0)
        rownum = np.flip((rownum > 2500).astype(int), axis = 0)

        # Convert the binary digits into arrays
        print('Converting binary digits into arrays..')
        MSGTYPE = np.zeros(msgtype.shape[1])
        ROWNUMB = np.zeros(rownum.shape[1])
        start_time = time.time()
        for tp in range(MSGTYPE.shape[0]):
            MSGTYPE[tp] = int(''.join(str(i) for i in msgtype[:,tp]), 2)
            ROWNUMB[tp] = int(''.join(str(i) for i in rownum[:,tp]), 2)
			
            if tp == 10000 or (tp%1000000==0 and tp!=0):
                time_taken=time.time() - start_time
                time_left= np.round(time_taken / tp * (MSGTYPE.shape[0] - tp) / 60)
                print(f'Approx time remaining: {time_left} mins')

        find_recording_start = np.ravel(np.nonzero(strobe))[0]
        find_data_rows = np.logical_and(np.ravel(np.equal(MSGTYPE,13)),np.ravel(np.greater(strobe,0)))  
        find_data_rows_ind = np.ravel(np.nonzero(find_data_rows))        

        rows = ROWNUMB[find_data_rows_ind]    # row numbers (mod 256)
        
        
        
        # ## Sanity check plot
        # starttime=18
        # endtime = starttime+8

        # fig,axs=plt.subplots(3,1)
        # axs[0].set_title('strobe - Indicates when behavior is sending message')
        # axs[0].plot(strobe,'.')
        # axs[0].plot(find_data_rows_ind,np.ones(len(find_data_rows_ind)),'.') #1 is to signal that a message is being sent
        # axs[0].set_xticks([])
        # axs[1].set_title('MSGTYPE - Indicates message type')
        # axs[1].plot(MSGTYPE,'.')
        # axs[1].plot(find_data_rows_ind,13*np.ones(len(find_data_rows_ind)),'.') #13 is the code for when the message is a row number 
        # axs[0].set_xlim([starttime*fs,endtime*fs])
        # axs[1].set_xlim([starttime*fs,endtime*fs])
        # # axs[1].set_xticks(np.linspace(starttime*fs_DIOx,endtime*fs_DIOx,5),np.linspace(starttime,endtime,5))
        # # axs[1].set_xlabel('Time (sec)')
        # axs[1].set_xticks([])


        # axs[2].set_title('ROWNUMB - Row numbers and other msgs')
        # axs[2].plot(ROWNUMB,'.')
        # axs[2].plot(find_data_rows_ind,rows,'.')
        # axs[2].set_xlim([starttime*fs,endtime*fs])
        # axs[2].set_xticks(np.linspace(starttime*fs,endtime*fs,5),np.linspace(starttime,endtime,5))
        # axs[2].set_xlabel('Time (sec)')

        # fig.suptitle(self.name)
        # fig.tight_layout()
        # xxx
        

        prev_row = rows[0]  # placeholder variable for previous row number
        counter = 0         # counter for number of cycles (i.e. number of times we wrap around from 255 to 0) in hdf row numbers

        print('Unwrapping row numbers..')
        start_time = time.time()
        for ind in range(1,len(rows)):
            row = rows[ind]
            cycle = (row < prev_row) # row counter has cycled when the current row number is less than the previous
            counter += cycle
            rows[ind] = counter*256 + row
            prev_row = row    
			
            if ind==1000: #to give an estimate of how long file will take to process
                time_taken=time.time() - start_time
                time_left= np.round(time_taken / ind * (len(rows) - ind) / 60)
                print(f'Approx time remaining: {time_left} mins')

        # Load data into dictionary
        hdf_times['row_number'] = rows
        hdf_times['ripple_samplenumber'] = find_data_rows_ind
        hdf_times['ripple_recording_start'] = find_recording_start
        hdf_times['ripple_dio_samplerate'] = fs

        return hdf_times


    def make_syncHDF_file(self):
        """Create .mat synchronization file for synchronizing Ripple and behavioral data (saved in .hdf file).
        """

        # Create dictionary to store synchronization data
        hdf_times = self.extract_rows()

        # Save syncing data as .mat file
        print('Saving..')
        mat_filename = self.path + '/' + self.name + '_syncHDF.mat'
        print(mat_filename)
        sp.io.savemat(mat_filename,hdf_times)

        return
#%%
# os.chdir(r'D:\brazos') # files that are downloaded
#os.chdir(r"D:\dropbox\Dropbox\Samantha lab\baseline") # on 
#os.chdir(r"Z:\storage\rawdata\ripple")
#os.chdir(r"/Volumes/Backup Plus/Airport/airp")
#os.chdir(r"E:")
# braz = nsyncHDF('braz20220407_04_te243.ns5')
# braz = nsyncHDF(r"C:\Users\coleb\Desktop\Santacruz Lab\Whitehall\Analysis\braz20250326_04_te1923.ns5") #bad
# braz = nsyncHDF(r"C:\Users\coleb\Desktop\Santacruz Lab\Whitehall\Analysis\braz20250225_04_te1880.ns5") #good
#signals = braz.output['data']
#fs = braz.output['samp_per_s']



PROJ_FOLDER = r"C:\Users\coleb\Desktop\Santacruz Lab\Whitehall\Analysis"
sessions = ['braz20240927_01_te5384','braz20241001_03_te5390','braz20241002_04_te5394',
            'braz20241004_02_te5396','braz20250221_03_te1873','braz20250225_04_te1880',
            'braz20250228_03_te1888','braz20250326_04_te1923','braz20250327_04_te1927']

for session in sessions:
    fname = os.path.join(PROJ_FOLDER,session)
    braz = nsyncHDF(fname)
    braz.make_syncHDF_file()
    


# braz.make_syncHDF_file() 
