from planner import *
from othertools import *
import matplotlib.pyplot as plt

def main():
    score_2t = readfile('./rq1_TimeLIME.csv')
    plt.subplots(figsize=(7, 7))
    plt.rcParams.update({'font.size': 10})
    print("--------------------------------------------------------------")
    print(score_2t)
    N = len(score_2t)
    width = 0.25
    dummy1 = []
    for i in range(0, len(score_2t)):
        dummy1.append(np.round(10 - np.mean(score_2t[i]), 3))
    print(dummy1)
    plt.plot(np.arange(N), dummy1, label='TimeLIME', marker='^', markersize=10, color='#22406D')

    plt.xticks(np.arange(N), ['3dmol_3Dmol.js', 'abinit_abinit',
                              'alchemistry_alchemlyb', 'Amber-MD_cpptraj',
                              'Amber-MD_pytraj', 'birkir_prime',
                              'BOINC_boinc', 'bustoutsolutions_siesta',
                              'cclib_cclib', 'chemfiles_chemfiles'])

    plt.yticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
    plt.subplots_adjust(bottom=0.2, left=0, right=1.1)
    plt.grid(axis='y')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=3)
    plt.savefig("rq1", dpi=200, bbox_inches='tight')
    plt.show()

    return


if __name__ == "__main__":
    main()
