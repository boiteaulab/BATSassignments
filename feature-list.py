# feature list assembly, gapfilling, and m/z error filtering 
# Christian Dewey, 10 December 2024

import os
import warnings
import pandas as pd

warnings.filterwarnings('ignore')
import sys
sys.path.append('./')

os.chdir('/CoreMS/')

from coremstools.Parameters import Settings
from coremstools.DataSet import DataSet

def write_sample_list(data_dir):
    raw_fs = pd.DataFrame({'File':[f for f in os.listdir(data_dir) if ('.raw' in f) ]}).to_csv(data_dir + 'sample_list.csv')


def process(data_dir):
    Settings.raw_file_directory = data_dir
    Settings.assignments_directory = Settings.raw_file_directory
    Settings.time_interval = 2
    #Settings.blank_sample_name = 'RMB_190828_BATS24_blnk.raw'

    dset = DataSet(path_to_sample_list= data_dir + 'sample_list.csv')
    #dset.run_assignment_error_plots(n_molclass=6)
    
    #dset.run_dispersity_calcs()
    dset.run_alignment(include_dispersity = False)
    dset.export_feature_list('aligned_features.csv')

    feature_list = pd.read_csv(data_dir + 'aligned_features.csv')
    dset.feature_list_df = feature_list
    dset.run_consolidation()
    dset.export_feature_list('consolidated_features.csv')

    feature_list = pd.read_csv(data_dir + 'consolidated_features.csv')
    dset.feature_list_df = feature_list
    dset.flag_errors()
    dset.export_csv('feature-list-all.csv')

    feature_list = pd.read_csv(data_dir + "feature-list-all.csv")
    dset.feature_list_ddf = feature_list
    dset.stoichiometric_classification()
    dset.export_csv('feature-list-all-stoi.csv')

if __name__ == '__main__':

    data_dir = r'/CoreMS/usrdata/'

    write_sample_list(data_dir)
    
    process(data_dir)
