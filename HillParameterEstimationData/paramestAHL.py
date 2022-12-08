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

def n_hills(C, thr):
    cop = 1
    thr= 10**thr
    C = 10**C
    return 1*thr**cop/(thr**cop + C**cop)

def n_linear(C, thr):
    thr= 10**thr
    #print(thr)
    C = 10**C
    G = 1
    return G*thr/(thr + C)

for hh in ["HetON", "HetOFF", "HomON", "HomOFF"]:
    # Parameter Esitmation Dataframe
    prest_df = SubsetDF(heho=hh, conc=0.2, actinh="ARA")

    # Make sure that the conc at 0 is a very samll number so no error encountered while taking log
    prest_df["AHL"] = prest_df["AHL"].replace(0, 10**(-6))
    prest_df = prest_df.iloc[1:]
    #print(prest_df)

    x = prest_df["mean"]
    # Divide by lowest conc of act/inh and scale by log10 values
    y = np.log10(prest_df["AHL"]/prest_df["AHL"].min())
    #print(x)
    #print(y)

    # Curve fitting function, can also provide a initial guess estimate
    param, param_cov = curve_fit(n_linear, y, x)
    print(param)
    print(param_cov)

    # Generate points to plot the fit of optimal paramters
    x_est  = [n_hills(i, thr=param[0]) for i in y]
    #print(x_est)

    if hh == "HetOFF":
        #print((10**param[0])*(0.0015625))
        #[print(i) for i in concDict["AHL"]]
        #[print((10**i)*0.0015625) for i in y]

        # Save dataframe with estimated values
        prest_df.rename(columns={"mean":"Experimental"}, inplace=True)
        prest_df["Estimated"] = x_est
        prest_df.to_csv("EstimatedAHL.csv", index=False)

        # R2 value calculation
        R2 = r2_score(prest_df["Experimental"], prest_df["Estimated"])

        with open("AHLEstParams.txt", "w") as fl:
            fl.write("Inh_Threshold : "+str(param[0])+"\n")
            fl.write("R2-Value : "+str(R2)+"\n")
            fl.write("Covariance Matrix :\n")
            for l in param_cov:
                fl.write(",".join([str(i) for i in l])+"\n")

        # Plotting Parameters
        mpl.rc_context({'figure.figsize':(8,8),"legend.fontsize":13,"legend.title_fontsize":13,"font.size":13,"axes.titlesize":13,"axes.labelsize":13,"xtick.labelsize":13,"ytick.labelsize":13})
        plt.plot(y, x, "bo", label="Expertimental")
        plt.plot(y, x_est, "r", label="Estimated")

        plt.xlabel("Normalized Conc. of AHL")
        plt.ylabel("Normalised Conc. of GFP")
        plt.legend()
        plt.tight_layout()
        plt.savefig("Final_"+hh+"_AHL_ARA0_2.png")
        plt.show()
