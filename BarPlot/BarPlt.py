import pandas as pd
import numpy as np
from itertools import product
import matplotlib as mpl
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
pd.set_option('mode.chained_assignment', None)

# Read the RAW dataframe
rdf = pd.read_csv("data/RAW.csv")

# Shorten Sample names
rdf["sample"] = rdf["sample"].replace({"_offsite":"OFF", "_onsite":"ON", "Heterogenous":"Het", "Homogenous":"Hom"}, regex=True)

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
print(concDict)


# Subset the required data for parameter estimation
def SubsetDF(heho, conc, actinh):
    # Het/Hom On/OFF
    #heho = "HetON"
    # Conc. of the act/inh
    #conc = 0.000002
    # Act/Inh name
    #actinh = "ARA"
    #print([heho, conc, actinh])
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

def gfpexp(CA, r_P_RA, r_P_RI, b_RatAI, b_SxA, b_A):
    # Estimated
    # Estimated paramters from hills fitting
    copA = 1.598
    copI = 1
    thrA = 4.321
    thrI = 0.748
    C0A = 0.0
    G = 0.681
    # Convert from log base 10
    thrA = (10**thrA)*(10**(-9))
    thrI = (10**thrI)*(0.0015625)
    CA = (10**CA)*(10**(-9))
    CI = ConcenI
    C0A = C0A
    #C0I = 10**C0I
    # Convert ratio of binding to discreet values
    b_RA = 1
    b_RI = b_RatAI
    # CA - incorporating binding and production rates
    CA = CA*r_P_RA*b_RA*b_SxA*b_A
    # CI - incorporating binding and production rates
    CI = CI*r_P_RI*b_RI
    # The ARA related term
    AraResp = C0A +  G*(CA**copA/(thrA**copA + CA**copA))
    # AHL related term
    AhlResp = thrI**copI/(thrI**copI + CI**copI)
    # Print the thrs
    #print([AraResp, AhlResp])
    # GFP expression level equation
    return 0.319 + AraResp*AhlResp
#return AraResp


@mpl.rc_context({'figure.figsize':(8,8),"legend.fontsize":13,"legend.title_fontsize":13,"font.size":13,"axes.titlesize":13,"axes.labelsize":13,"xtick.labelsize":13,"ytick.labelsize":13})
def fitPlot(ConcenI, heho):
    # Parameter Esitmation Dataframe
    prest_df = SubsetDF(heho=heho, conc=ConcenI, actinh="AHL")
    #print(prest_df)

    # Make the value of 0 conc to a very low value to avoid log error
    prest_df.at[prest_df.index[0], "ARA"] = 10**(-9)
    #prest_df = prest_df.iloc[1:]

    # Define the x and y arrays
    x = list(prest_df["mean"])
    y = list(np.log10(prest_df["ARA"]/prest_df["ARA"].min()))
    #print(x)
    #print(y)

    # Generating points based on qualitative parameters
    if heho[:3] == "Hom":
        x_est = [gfpexp(i, r_P_RA=1, r_P_RI=1, b_RatAI=1, b_SxA=1, b_A=1) for i in [y[0], y[-1]]]
    else:
        x_est = [gfpexp(i, r_P_RA=1, r_P_RI=1, b_RatAI=10, b_SxA=1, b_A=1) for i in [y[0], y[-1]]]

    return x_est

# Create a DF of the output values
solDF = []
for x in [0.0, 0.1]:
    for h in ["HetON", "HetOFF", "HomOFF", "HomON"]:
        ConcenI = x
        o = fitPlot(ConcenI, heho = h)
        solDF.append([o[0], x, 0, h])
        solDF.append([o[1], x, 1, h])

solDF = pd.DataFrame(solDF, columns=["GFP", "AHL", "ARA", "Condition"])
solDF["AHL"] = solDF["AHL"].replace({0.0:0, 0.1:1}).astype(int)
solDF["AHL-ARA"] = solDF["AHL"].astype(str) + "-" + solDF["ARA"].astype(str)
solDF = solDF.sort_values(by=["Condition", "AHL-ARA"])
print(solDF)

# Store the raw mean values as a list
meanRLi = []

# Print GFP RAW data
tmpdf = gdf[gdf["ARA"].isin([0.0, 0.2])]
tmpdf = tmpdf[tmpdf["AHL"].isin([0.0, 0.1])]
tmpdf["AHL-ARA"] = tmpdf["AHL"].replace({0.0:0, 0.1:1}).astype(int).astype(str) +"-"+ tmpdf["ARA"].replace({0.0:0, 0.2:1}).astype(int).astype(str)
tmpdf = tmpdf[["sample", "mean", "AHL-ARA"]].sort_values(by=["sample", "AHL-ARA"])
#tmpdf = tmpdf[tmpdf["AHL-ARA"] == "0-1"]
tmpdf.to_csv("Experimental_BarPlt.csv", index=False)
print(tmpdf)

solDF["GFP"] = solDF["GFP"]*list(tmpdf["mean"])
solDF.to_csv("Model_Barplt.csv")
print(solDF)

g = sns.barplot(data=solDF, x="Condition", y="GFP", hue="AHL-ARA")
g.legend(loc="upper right", bbox_to_anchor=(1.25, 1))
plt.tight_layout()
plt.savefig("fit.png")
plt.show()

