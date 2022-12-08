import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
pd.options.mode.chained_assignment = None

# Read the RAW dataframe
rdf = pd.read_csv("data/RAW.csv")

# Shorten Sample names
rdf["sample"] = rdf["sample"].replace({"_offsite":"OFF", "_onsite":"ON", "Heterogenous":"Het", "Homogenous":"Hom"}, regex=True)
#print(rdf)

# Remove AHL = 10 rows
rdf= rdf[rdf["AHL"] != 10]

# Rename arabinose to ARA
rdf = rdf.rename(columns={"arabinose":"ARA"})

# Subset gfp and mkate data
gdf = rdf[rdf["fluo"] == "GFP"].reset_index(drop=True)
kdf = rdf[rdf["fluo"] == "mKate"].reset_index(drop=True)

# Normalise sd wrt to the mean
gdf["sd"] = gdf["sd"]/gdf["mean"]
kdf["sd"] = kdf["sd"]/kdf["mean"]

# Find Unique concentrations of AHL and Arabinose
ara = sorted(gdf["ARA"].unique())
ahl = sorted(gdf["AHL"].unique())
concDict = {"AHL":ahl, "ARA":ara}

# Subset the required data for parameter estimation
def SubsetDF(heho, conc, actinh):
    # Het/Hom On/OFF
    #heho = "HetON"
    # Conc. of the act/inh
    #conc = 0.000002
    # Act/Inh name
    #actinh = "ARA"
    print([heho, conc, actinh])
    # Create a subset df
    tmp_df = gdf[gdf["sample"] == heho]
    # Normalise with max mean value
    tmp_df["mean"] = tmp_df["mean"]/tmp_df["mean"].max()
    # Scale Std to normalised mean values
    tmp_df["sd"] = tmp_df["sd"]*tmp_df["mean"]
    # Final subset of the dataframe with conc and actinh
    pltdf = tmp_df[(tmp_df[actinh] == conc) & (tmp_df["sample"] == heho)]
    #print(pltdf)
    return pltdf

# Define Hills Equations
def p_hills(C, cop, thr, C0, G):
    thr = 10**thr
    C = 10**C
    return C0 + G*C**cop/(thr**cop + C**cop)
# [1.5, 2, 39000, 80000] => Initial Guess for no normalisation and minmax

def n_hills(C, cop, thr):
    return C0 - pow(thr, cop)/(pow(thr, cop) + pow(C, cop))

for hh in ["HomON", "HomOFF", "HetON", "HetOFF"]:
    # Parameter Esitmation Dataframe
    prest_df = SubsetDF(heho=hh, conc=0.0, actinh="AHL")
    print(prest_df)

    # Make the value of 0 conc to a very low value to avoid log error
    prest_df["ARA"] = prest_df["ARA"].replace(0, 10**(-9))
    #prest_df = prest_df.iloc[1:]

    # Define the x and y arrays
    x = prest_df["mean"]
    y = np.log10(prest_df["ARA"]/prest_df["ARA"].min())
    #print(x)
    #print(y)

    # Curve fitting over the data using positive hills equation
    param, param_cov = curve_fit(p_hills, y, x)
    print(param)
    print(param_cov)

    # Generating points based on optimised parameters
    x_est  = [p_hills(i, cop=param[0], thr=param[1], C0=param[2], G=param[3]) for i in y]
    print(x_est)

    if hh == "HetOFF":
        #print((10**param[1])*10**(-9))
        #[print(i) for i in concDict["ARA"]]
        #[print((10**i)*(10**(-9))) for i in y]

        # Save dataframe with estimated values
        prest_df.rename(columns={"mean":"Experimental"}, inplace=True)
        prest_df["Estimated"] = x_est
        prest_df.to_csv("EstimatedARA.csv", index=False)

        # R2 value calculation
        R2 = r2_score(prest_df["Experimental"], prest_df["Estimated"])
        print(R2)

        with open("ARAEstParams.txt", "w") as fl:
            fl.write("Act_HillsCoeff : "+str(param[0])+"\n")
            fl.write("Act_Threshold : "+str(param[1])+"\n")
            fl.write("Act_Basal : "+str(param[2])+"\n")
            fl.write("GFP_Max : "+str(param[1])+"\n")
            fl.write("R2-Value : "+str(R2)+"\n")
            fl.write("Covariance Matrix :\n")
            for l in param_cov:
                fl.write(",".join([str(i) for i in l])+"\n")

        # Plotting Parameters
        mpl.rc_context({'figure.figsize':(8,8),"legend.fontsize":13,"legend.title_fontsize":13,"font.size":13,"axes.titlesize":13,"axes.labelsize":13,"xtick.labelsize":13,"ytick.labelsize":13})
        plt.plot(y, x, "bo", label="Experimental")
        plt.plot(y, x_est, "r", label="Estimated")
        plt.xlabel("Normalized Conc. of ARA")
        plt.ylabel("Normalized Conc. of GFP")
        plt.legend()
        plt.tight_layout()
        plt.savefig("Final_"+hh+"_ARA_AHL0_0.png")
        plt.show()


