from planner import *
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

def main():
    paras = [True]
    explainer = None
    print("RUNNING FREQUENT ITEMSET LEARNING")
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
    old, new = [], []
    for par in paras:
        for name in fnames:
            o, n = historical_logs(name, 18, explainer, smote=True, small=.03, act=par)
            old.append(o)
            new.append(n)
    everything = []
    for i in range(len(new)):
        everything.append(old[i] + new[i])

    # TimeLIME planner
    paras = [True]
    explainer = None
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
    scores_t, bcs_t = [], []
    size_t, score_2t = [], []
    records2 = []
    con_matrix1 = []
    i = 0
    # print(len(everything))
    print("-----------------")
    for par in paras:
        for name in fnames:
            df = pd.DataFrame(everything[i])
            i += 1
            itemsets = convert_to_itemset(df)
            te = TransactionEncoder()
            te_ary = te.fit(itemsets).transform(itemsets, sparse=True)
            df = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)
            rules = apriori(df, min_support=0.001, max_len=5, use_colnames=True)
            score, bc, size, score_2, rec, mat = TL(name, 18, rules, smote=True, act=par)
            scores_t.append(score)
            bcs_t.append(bc)
            size_t.append(size)
            score_2t.append(score_2)
            records2.append(rec)
            con_matrix1.append(mat)

    pd.DataFrame(score_2t).to_csv("rq1_TimeLIME.csv")
    pd.DataFrame(scores_t).to_csv("rq2_TimeLIME.csv")
    pd.DataFrame(bcs_t).to_csv("rq3_TimeLIME.csv")

    return


if __name__ == "__main__":
    main()

