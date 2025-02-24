import pandas as pd
import glob
import sys

for arg in sys.argv:
    RUN = arg

resfiles = glob.glob(f"output/*{RUN}.csv")
all_max = {}

for f in resfiles:
    print(f"---------------------- {f} -----------------------")
    dfres = pd.read_csv(f)
    
    dfresqcm = dfres[dfres['target'] == 'QCM']
    dfresatt = dfres[dfres['target'] == 'SelfReports']

    
    ctmeanf1q = pd.crosstab(dfresqcm["feat_set"], dfresqcm["classifier"], values=dfresqcm["f1_score"], aggfunc='mean')
    ctmeanf1thq = pd.crosstab(dfresqcm["feat_set"], dfresqcm["classifier"], values=dfresqcm["f1_score_th"], aggfunc='mean')
    ctstdf1q = pd.crosstab(dfresqcm["feat_set"], dfresqcm["classifier"], values=dfresqcm["f1_score"], aggfunc='mean')
    ctstdf1thq = pd.crosstab(dfresqcm["feat_set"], dfresqcm["classifier"], values=dfresqcm["f1_score_th"], aggfunc='mean')
    
    ctmeanf1a = pd.crosstab(dfresatt["feat_set"], dfresatt["classifier"], values=dfresatt["f1_score"], aggfunc='mean')
    ctmeanf1tha = pd.crosstab(dfresatt["feat_set"], dfresatt["classifier"], values=dfresatt["f1_score_th"], aggfunc='mean')
    ctstdf1a = pd.crosstab(dfresatt["feat_set"], dfresatt["classifier"], values=dfresatt["f1_score"], aggfunc='mean')
    ctstdf1tha = pd.crosstab(dfresatt["feat_set"], dfresatt["classifier"], values=dfresatt["f1_score_th"], aggfunc='mean')
    
    all_max[f] = {"QCM":max(max(ctmeanf1q.max()), max(ctmeanf1thq.max())), "SelfReports":max(max(ctmeanf1a.max()), max(ctmeanf1tha.max())) }
    
    """print(f"MEAN of F1 - {f}")
    print(ctmeanf1)
    print(f"STD of F1 - {f}")
    print(ctstdf1)
    print(f"MEAN of F1th - {f}")
    print(ctmeanf1th)
    print(f"STD of F1th - {f}")
    print(ctstdf1th)
    print("")"""
    
for k,v in all_max.items():
    print(k, " == \n", v)
    
    