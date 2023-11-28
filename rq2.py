from planner import *
from othertools import *
import matplotlib.pyplot as plt
import scipy as sp

def main():
    fnames = [['3dmol_3Dmol_split_1.csv', '3dmol_3Dmol_split_2.csv', '3dmol_3Dmol_split_3.csv'],
              ['abinit_abinit_split_1.csv', 'abinit_abinit_split_2.csv', 'abinit_abinit_split_3.csv'],
              ['alchemistry_alchemlyb_split_1.csv', 'alchemistry_alchemlyb_split_2.csv', 'alchemistry_alchemlyb_split_3.csv'],
              ['Amber-MD_cpptraj_split_1.csv', 'Amber-MD_cpptraj_split_2.csv', 'Amber-MD_cpptraj_split_3.csv'],
              ['Amber-MD_pytraj_split_1.csv', 'Amber-MD_pytraj_split_2.csv', 'Amber-MD_pytraj_split_3.csv'],
              ['birkir_prime_split_1.csv', 'birkir_prime_split_2.csv', 'birkir_prime_split_3.csv'],
              ['BOINC_boinc_split_1.csv', 'BOINC_boinc_split_2.csv', 'BOINC_boinc_split_3.csv'],
              ['bustoutsolutions_siesta_split_1.csv', 'bustoutsolutions_siesta_split_2.csv', 'bustoutsolutions_siesta_split_3.csv'],
              ['cclib_cclib_split_1.csv', 'cclib_cclib_split_2.csv', 'cclib_cclib_split_3.csv'],
              ['chemfiles_chemfiles_split_1.csv', 'chemfiles_chemfiles_split_2.csv', 'chemfiles_chemfiles_split_3.csv']
              ]
    scores_t = readfile('./rq2_TimeLIME.csv')

    N = len(scores_t)
    # print(N)
    for i in range(N):
        print()
        print(fnames[i][0])
        print('TimeLIME IQR: ', 100*(sp.stats.iqr(scores_t[i])))
        print( 'TimeLIME Median: ', 100*(np.median(scores_t[i])))
    return


if __name__ == "__main__":
    main()
