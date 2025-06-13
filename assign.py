import os
import warnings
import pandas as pd
import gc
import psutil 
import time    
import sys 
from datetime import datetime

warnings.filterwarnings('ignore')

os.chdir('/CoreMS/')
from corems.mass_spectra.input import rawFileReader
from corems.molecular_id.search.molecularFormulaSearch import SearchMolecularFormulas
from corems.encapsulation.factory.parameters import MSParameters, LCMSParameters
from corems.mass_spectrum.calc.Calibration import MzDomainCalibration

# Helper function to track memory usage
def log_memory_usage(stage):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {stage}: "
          f"RSS = {mem_info.rss / (1024 * 1024):.2f} MB, "
          f"VMS = {mem_info.vms / (1024 * 1024):.2f} MB")


def assign_formula(data_dir, file, times, interval, cal_error_range):
    # Set MS parameters
    MSParameters.molecular_search.min_ppm_error = -0.3
    MSParameters.molecular_search.max_ppm_error = 0.3
    MSParameters.molecular_search.db_chunk_size = 1000
    MSParameters.molecular_search.error_method = None

    MSParameters.mass_spectrum.min_calib_ppm_error = cal_error_range[0]
    MSParameters.mass_spectrum.max_calib_ppm_error = cal_error_range[1]
    MSParameters.mass_spectrum.calib_pol_order = 2
    MSParameters.mass_spectrum.calib_sn_threshold = 50
    MSParameters.mass_spectrum.min_picking_mz = 220
    MSParameters.mass_spectrum.max_picking_mz = 900

    MSParameters.mass_spectrum.noise_threshold_log_nsigma = 15 #increased from 12 on 12/9/2024
    MSParameters.ms_peak.peak_min_prominence_percent = 0.02
    MSParameters.ms_peak.legacy_resolving_power = False

    MSParameters.molecular_search.url_database = 'postgresql+psycopg2://coremsappdb:coremsapppnnl@corems-molformdb-1:5432/coremsapp'
    MSParameters.molecular_search.score_method = "prob_score"
    MSParameters.molecular_search.output_score_method = "prob_score"
    MSParameters.molecular_search.mz_error_score_weight = 0.5
    MSParameters.molecular_search.isotopologue_score_weight = 0.5
    LCMSParameters.lc_ms.scans = (-1, -1)

    print("Loading file: " + file)

    results = []
    
    #with rawFileReader.ImportMassSpectraThermoMSFileReader(file) as parser:
    parser = rawFileReader.ImportMassSpectraThermoMSFileReader(file)
    #tic = parser.get_tic(ms_type='MS',smooth=False)[0]
    tic=parser.get_tic(ms_type='MS',smooth=False,peak_detection=False)[0]
    tic_df = pd.DataFrame({'time': tic.time, 'scan': tic.scans})
    refmasslist = file.replace('.raw', "_calibrants_pos.ref")

    for timestart in times:

        print('running: ' + str(timestart))
        scans = tic_df[tic_df.time.between(timestart, timestart + interval)].scan.tolist()

        mass_spectrum = parser.get_average_mass_spectrum_by_scanlist(scans)
        mass_spectrum.filter_by_min_resolving_power(21, 0.6)

        # Run calibration
        MzDomainCalibration(mass_spectrum, refmasslist).run()

        # Molecular search settings
        mass_spectrum.molecular_search_settings.min_dbe = 0
        mass_spectrum.molecular_search_settings.max_dbe = 20
        mass_spectrum.molecular_search_settings.usedAtoms = {
            'C': (1, 50),
            'H': (4, 100),
            'O': (1, 20),
            'N': (0, 10),
            'S': (0, 2),
            'Na': (0, 1)
        }

        mass_spectrum.molecular_search_settings.used_atom_valences = {
            'C': 4, '13C': 4, 
            'H': 1, 
            'O': 2, '17O': 2, '18O': 2,
            'N': 3, '15N': 3, 
            'S': 2, '34S': 2, 
            'Na': 1}
        
        mass_spectrum.molecular_search_settings.max_hc_filter = 3
        mass_spectrum.molecular_search_settings.max_oc_filter = 1.2
        mass_spectrum.molecular_search_settings.isProtonated = True

        # Search molecular formulas
        SearchMolecularFormulas(mass_spectrum).run_worker_mass_spectrum()

        # Collect results
        mass_spectrum.percentile_assigned(report_error=True)
        assignments = mass_spectrum.to_dataframe()
        assignments['Time'] = timestart
        results.append(assignments)

        # Free up memory for large objects
        del mass_spectrum, scans
        gc.collect()


    # Combine results into a DataFrame
    results_df = pd.concat(results, ignore_index=True)
    results_df['file'] = os.path.basename(file)
    results_df['Molecular Class'] = results_df['Molecular Formula'].replace(' ', '', regex=True).replace(r'\d+', '', regex=True)
    results_df.loc[results_df['Heteroatom Class'] == 'unassigned', 'Molecular Class'] = 'Unassigned'
    results_df.loc[results_df['Is Isotopologue'] == 1, 'Molecular Class'] = 'Isotope'

    return results_df


if __name__ == '__main__':

    data_dir = r'/CoreMS/usrdata/'

    interval = 2
    time_min = 2
    time_max = 36
    times = list(range(time_min, time_max, interval))

    #flist = [sys.argv[1]]
    flist = [data_dir + f for f in os.listdir(data_dir) if '.raw' in f]   
    #flist = [data_dir + f for f in list(pd.read_csv(data_dir + 'unassigned-samplelist.csv')['File'])]
    cal_range = pd.read_csv(os.path.join(data_dir, 'error_range.csv'), header = None)
    cal_range.columns = ['file','min m/z error (ppm)','max m/z error (ppm)']
    range_dict = {row['file']: (row['min m/z error (ppm)'], row['max m/z error (ppm)']) for _, row in cal_range.iterrows()}
    
    now = datetime.now()
    formatted_datetime = now.strftime("%Y-%m-%d %H:%M:%S")

    with open(data_dir + 'assignments.log', 'a') as logf:
        logf.write(f'^^^^ {formatted_datetime} ^^^^\n')

    for i, f in enumerate(flist, start=1):

        j = f.split('/')[-1] 
        fname = f.replace('.raw', '.csv')
        cal_error_range = range_dict.get(j)

        if cal_error_range is not None:

            now = datetime.now()
            formatted_datetime = now.strftime("%H:%M:%S, %Y-%m-%d")
            with open(data_dir + 'assignments.log', 'a') as logf:
                logf.write(f'----Starting assignment for {j} at {formatted_datetime}. \n')

            try:
                output = assign_formula(data_dir=data_dir, file=f, times=times, interval=interval, cal_error_range=cal_error_range)
                output.to_csv(fname, index=False)
                now = datetime.now()
                formatted_datetime = now.strftime("%H:%M:%S, %Y-%m-%d")
                with open(data_dir + 'assignments.log', 'a') as logf:
                    logf.write(f'Successfully assigned to {j} at {formatted_datetime}\n')

            except Exception as e:
                with open(data_dir + 'assignments.log', 'a') as logf:
                    logf.write(f'****Failed to assign to {j}. {e}\n')
            gc.collect()

        else:
            with open(data_dir + 'assignments.log', 'a') as logf:
                logf.write(f'****No calibration file for {j} - did not assign to {j}\n')

        now = datetime.now()
        formatted_datetime = now.strftime("%H:%M:%S, %Y-%m-%d")
        with open(data_dir + 'assignments.log', 'a') as logf:
            logf.write(f'Assignments completed at {formatted_datetime}\n\n')
