import os
from tempfile import tempdir
import warnings

import pandas as pd
import csv

os.chdir(r'/CoreMS/')

from corems.mass_spectra.input import rawFileReader
from corems.molecular_id.search.molecularFormulaSearch import SearchMolecularFormulas
from corems.encapsulation.factory.parameters import MSParameters, LCMSParameters

import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.linear_model import LinearRegression
import numpy as np

warnings.filterwarnings('ignore')
import sys
sys.path.append('./')


global data_dir
global cal_dir
data_dir = r'/CoreMS/usrdata/'
cal_dir = data_dir

def assign_formula(file, times):

    MSParameters.mass_spectrum.calib_pol_order = 2
    MSParameters.mass_spectrum.min_picking_mz = 200
    MSParameters.mass_spectrum.max_picking_mz = 900
    MSParameters.mass_spectrum.noise_threshold_method = 'log'
    MSParameters.mass_spectrum.noise_threshold_log_nsigma = 20

    MSParameters.ms_peak.peak_min_prominence_percent = 0.02
    MSParameters.ms_peak.legacy_resolving_power = False

    MSParameters.molecular_search.min_ppm_error = -2
    MSParameters.molecular_search.max_ppm_error = 4
    MSParameters.molecular_search.db_chunk_size = 1000
    MSParameters.molecular_search.mz_error_score_weight = 0.5
    MSParameters.molecular_search.isotopologue_score_weight = 0.5
    MSParameters.molecular_search.url_database = 'postgresql+psycopg2://coremsappdb:coremsapppnnl@corems-molformdb-1:5432/coremsapp'
    MSParameters.molecular_search.output_score_method = "prob_score"
    LCMSParameters.lc_ms.scans=(-1,-1)

    print("\nLoading file: "+ file)

    #with rawFileReader.ImportMassSpectraThermoMSFileReader(file) as parser:

    parser = rawFileReader.ImportMassSpectraThermoMSFileReader(file) 
    MSfiles={}
    MSfiles[file]=parser

    tic=parser.get_tic(ms_type='MS',smooth=False,peak_detection=False)[0]
    tic_df=pd.DataFrame({'time': tic.time,'scan': tic.scans})

    results = []
    for timestart in times:

        print('running: ' + str(timestart))
        scans=tic_df[tic_df.time.between(timestart,timestart+interval)].scan.tolist()

        mass_spectrum = parser.get_average_mass_spectrum_by_scanlist(scans)

        mass_spectrum.molecular_search_settings.min_dbe = 0
        mass_spectrum.molecular_search_settings.max_dbe = 16

        mass_spectrum.molecular_search_settings.usedAtoms['C'] = (1, 40)
        mass_spectrum.molecular_search_settings.usedAtoms['H'] = (4, 80)
        mass_spectrum.molecular_search_settings.usedAtoms['O'] = (1, 12)
        mass_spectrum.molecular_search_settings.usedAtoms['N'] = (0, 2)

        mass_spectrum.molecular_search_settings.isProtonated = True

        mass_spectrum.molecular_search_settings.max_oc_filter=1.2
        mass_spectrum.molecular_search_settings.max_hc_filter=3
        mass_spectrum.molecular_search_settings.used_atom_valences = {'C': 4,
                                        '13C': 4,
                                        'H': 1,
                                        'O': 2,
                                        'N': 3}

        SearchMolecularFormulas(mass_spectrum).run_worker_mass_spectrum()
        mass_spectrum.percentile_assigned(report_error=True)

        assignments=mass_spectrum.to_dataframe()
        assignments['Time']=timestart
        results.append(assignments)

    results=pd.concat(results,ignore_index=True)
    results['file'] = file.split('/')[-1]
    results['Molecular Class']  = results['Molecular Formula'].replace(' ', '',regex=True).replace('\d+', '',regex=True)
    results['Molecular Class'][results['Heteroatom Class'] == 'unassigned'] = 'Unassigned'
    results['Molecular Class'][results['Is Isotopologue'] == 1] = 'Isotope'

    return(results)

def make_error_dist_fig(output,f):

    #### Plot and save error distribution figure
    fig, ((ax1, ax2)) = plt.subplots(1,2)
    fig.set_size_inches(12, 6)

    sns.scatterplot(x='m/z',y='m/z Error (ppm)',hue='Molecular Class',data=output,ax=ax1, edgecolor='none')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False)
    ax1.set_title('a', fontweight='bold', loc='left')
    sns.kdeplot(x='m/z Error (ppm)',data=output,hue='Time',ax=ax2,legend=True)
    ax2.set_title('b', fontweight='bold', loc='left')
    fig.tight_layout()

    fig.savefig(cal_dir + f.replace('.raw','_errorplot.jpg').split('/')[-1], dpi=200,format='jpg')


def get_ci(df,fname, ci, it):

    from scipy.optimize import curve_fit

    # pip install uncertainties, if needed
    try:
        import uncertainties.unumpy as unp
        import uncertainties as unc
    except:
        try:
            from pip import main as pipmain
        except:
            from pip._internal import main as pipmain
        pipmain(['install','uncertainties'])
        import uncertainties.unumpy as unp
        import uncertainties as unc

    # import data
    x = df['m/z'].values
    y = df['m/z Error (ppm)'].values
    n = len(y)

    def f(x, a, b, c, d):
        return a*x**3 + b*x**2 + c*x + d

    popt, pcov = curve_fit(f, x, y)

    # retrieve parameter values
    a = popt[0]
    b = popt[1]
    c = popt[2]
    d = popt[3]

    # calculate parameter confidence interval
    a,b,c,d = unc.correlated_values(popt, pcov)

    # calculate regression confidence interval
    px = x
    py = a*px**3 + b*px**2 + c*px + d
    nom = unp.nominal_values(py)
    std = unp.std_devs(py)

    def predband(x, xd, yd, p, func, conf=ci):
        alpha = 1.0 - conf    # significance
        N = xd.size          # data sample size
        var_n = len(p)  # number of parameters
        # Quantile of Student's t distribution for p=(1-alpha/2)
        q = stats.t.ppf(1.0 - alpha / 2.0, N - var_n)
        # Stdev of an individual measurement
        se = np.sqrt(1. / (N - var_n) * \
                    np.sum((yd - func(xd, *p)) ** 2))
        # Auxiliary definitions
        sx = (x - xd.mean()) ** 2
        sxd = np.sum((xd - xd.mean()) ** 2)
        # Predicted values (best-fit model)
        yp = func(x, *p)
        # Prediction band
        dy = q * se * np.sqrt(1.0+ (1.0/N) + (sx/sxd))
        # Upper & lower prediction bands.
        lpb, upb = yp - dy, yp + dy
        return lpb, upb

    lpb, upb = predband(px, x, y, popt, f, conf=ci)

    x_safe = []
    y_safe = []
    x_removed = []
    y_removed = []

    for i in range(len(x)):
        if (y[i] < upb[i]) and (y[i] > lpb[i]):
            x_safe.append(x[i])
            y_safe.append(y[i])
        else:
            x_removed.append(x[i])
            y_removed.append(y[i])

    px = np.arange(min(x),max(x),0.01)
    py = a*px**3 + b*px**2 + c*px + d
    nom = unp.nominal_values(py)
    lpb, upb = predband(px, x, y, popt, f, conf=ci)

    fig, ax = plt.subplots()
    # plot the regression
    ax.scatter(x_safe,y_safe,color = 'C0')
    ax.scatter(x_removed, y_removed, color = 'red')
    ax.plot(px, nom, c='black',marker='o',linewidth=0, markersize = 0.75) #,label='y=a*x^3 + b*x^2 + c*x + d')

    ci_p = ci * 100
    # uncertainty lines (95% confidence)
    ##ax.plot(px, nom - 1.96 * std, c='orange',\
    ##        label= str(int(ci_p)) +'% Confidence Region')
    ##ax.plot(px, nom + 1.96 * std, c='orange')
    # prediction band (95% confidence)
    ax.plot(px, lpb, 'C1o', linewidth=0,markersize=0.5) #,label=str(int(ci_p)) +'% Prediction Band' )
    ax.plot(px, upb, 'C1o', linewidth=0,markersize=0.5)
    ax.set_ylabel('m/z Error (ppm)')
    ax.set_xlabel('m/z')
    #ax.legend(loc='best')

    fig.savefig(fname.split('.')[0] + f'_regression{it}.png', dpi=200)

    filtered_df = df[df['m/z'].isin(x_safe)]
    std_cal = np.std(df['m/z Error (ppm)'])
    return filtered_df, std_cal


def make_cal_list(output, polarity, refmasslist, fname, expected_spread, ci=0.68, max_it = 4):

    cal_list = output[output['Confidence Score'] > 0.6]
    cal_list = cal_list[cal_list['Ion Charge'] == 1 * polarity]
    cal_list = cal_list[cal_list['Molecular Class']!='Isotope'].drop_duplicates(subset=['Molecular Formula'])

    it = 1
    cal_list, std_cal = get_ci(cal_list,fname, ci, it = it)
    while (std_cal > expected_spread) and (it < max_it):
        it += 1
        cal_list, std_cal = get_ci(cal_list,fname, ci, it = it)

    if std_cal > expected_spread:

        warnings.warn('St dev of calibration points exceeds expected spread!')

        return None

    else:
        cal=pd.DataFrame({'# Name':cal_list['Molecular Formula'],
                        'm/z value':cal_list['Calculated m/z'],
                        'charge':cal_list['Ion Charge'],
                        ' ion formula':cal_list['Molecular Formula'],
                        'collision cross section [A^2]':cal_list['Ion Charge']})

        cname = f.replace('.raw','_'+ refmasslist)
        cal.to_csv(cname,sep='\t',index=False)


        fig, ((ax1, ax2)) = plt.subplots(1,2)

        fig.set_size_inches(12, 6)
        sns.scatterplot(x='m/z',y='m/z Error (ppm)',hue='Molecular Class',data=cal_list,ax=ax1, edgecolor='none')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False)
        ax1.set_title('a', fontweight='bold', loc='left')
        sns.kdeplot(x='m/z Error (ppm)',data=cal_list,hue='Time',ax=ax2,legend=True)
        ax2.set_title('b', fontweight='bold', loc='left')
        fig.tight_layout()
        fname = f.replace('.raw','_calibrants_errorplot.jpg')
        fname = fname.split('/')[-1]

        fig.savefig(cal_dir + fname, dpi=200,format='jpg')

        return cal_list


if __name__ == '__main__':

    refmasslist = "calibrants_pos.ref"
    polarity = 1

    #flist = [data_dir + f for f in os.listdir(data_dir) if '.raw' in f]
    #flist = [sys.argv[1]]
    flist = list(pd.read_csv(data_dir + 'unassigned-samplelist.csv')['File'])
    flist = [data_dir + f for f in flist]
    interval = 2
    time_min = 0
    time_max = 36
    times = list(range(time_min,time_max,interval))

    final_cal = []

    error_range_dict = {}
    with open(data_dir + 'write_calfiles.log', 'a') as logf:
        for f in flist:

            fname = f.replace('.raw','_cal-assigned.csv')
            cname = f.replace('.raw','_'+ refmasslist)

            #try:
            output = assign_formula(file = f, times = times)
            output.to_csv(cal_dir + f.replace('.raw','_cal-assigned.csv').split('/')[-1])

            cal_list = make_cal_list(output,polarity,refmasslist,fname,expected_spread=0.5,ci=0.68)

            if cal_list is not None:

                make_error_dist_fig(output,f)

                minsign = 1
                if min(cal_list['m/z Error (ppm)']) < 0:
                    minsign = -1
                maxsign = 1
                if max(cal_list['m/z Error (ppm)']) < 0:
                    maxsign = -1

                error_range_dict[f.split('/')[-1]] = [(min(cal_list['m/z Error (ppm)']) )-(0.1), (max(cal_list['m/z Error (ppm)'])) + (0.1) ]
                j = f.split('/')[-1]
                logf.write(f'Successfully wrote cal file for {j}\n')
            else:
                logf.write("****Failed to write calibration file for {0}\n".format(str(f)))
    try:
        with open(cal_dir + "error_range.csv", "a") as outfile:

            writer = csv.writer(outfile)
            file_list = list(error_range_dict.keys())
            limit = len(file_list)
            #writer.writerow(['file','min m/z error (ppm)', 'max m/z error (ppm)'])
            for i in range(limit):
                row = [file_list[i]] + error_range_dict[file_list[i]]
                writer.writerow(row)
    except:
        pass




