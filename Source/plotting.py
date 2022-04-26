#----------------------------------------------------------------------
# Script for plotting some statistics
# Author: Pablo Villanueva Domingo
# Last update: 4/22
#----------------------------------------------------------------------

import matplotlib.pyplot as plt
from Source.constants import *
from sklearn.metrics import r2_score
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
import matplotlib as mpl
mpl.rcParams.update({'font.size': 12})

# Plot loss trends
def plot_losses(train_losses, valid_losses, test_loss, err_min, hparams):

    epochs = hparams.n_epochs
    plt.plot(range(epochs), np.exp(train_losses), "r-",label="Training")
    plt.plot(range(epochs), np.exp(valid_losses), "b:",label="Validation")
    plt.legend()
    plt.yscale("log")
    plt.title(f"Test loss: {test_loss:.2e}, Minimum relative error: {err_min:.2e}")
    plt.savefig("Plots/loss_"+hparams.name_model()+".png", bbox_inches='tight', dpi=300)
    plt.close()

# Remove normalization of cosmo parameters
def denormalize(trues, outputs, errors, minpar, maxpar):

    trues = minpar + trues*(maxpar - minpar)
    outputs = minpar + outputs*(maxpar - minpar)
    errors = errors*(maxpar - minpar)
    return trues, outputs, errors

# Scatter plot of true vs predicted cosmological parameter
def plot_out_true_scatter(hparams, cosmoparam, testsuite = False):

    figscat, axscat = plt.subplots(figsize=(6,5))
    suite, simset = hparams.simsuite, hparams.simset
    col = colorsuite(suite)

    # Load true values and predicted means and standard deviations
    outputs = np.load("Outputs/outputs_"+hparams.name_model()+".npy")
    trues = np.load("Outputs/trues_"+hparams.name_model()+".npy")
    errors = np.load("Outputs/errors_"+hparams.name_model()+".npy")

    # There is a (0,0) initial point, fix it
    outputs = outputs[1:]
    trues = trues[1:]
    errors = errors[1:]

    # Choose cosmo param and denormalize
    if cosmoparam=="Om":
        minpar, maxpar = 0.1, 0.5
        outputs, trues, errors = outputs[:,0], trues[:,0], errors[:,0]
    elif cosmoparam=="Sig":
        minpar, maxpar = 0.6, 1.0
        outputs, trues, errors = outputs[:,1], trues[:,1], errors[:,1]
    trues, outputs, errors = denormalize(trues, outputs, errors, minpar, maxpar)

    # Compute the number of points lying within 1 or 2 sigma regions from their uncertainties
    cond_success_1sig, cond_success_2sig = np.abs(outputs-trues)<=np.abs(errors), np.abs(outputs-trues)<=2.*np.abs(errors)
    tot_points = outputs.shape[0]
    successes1sig, successes2sig = outputs[cond_success_1sig].shape[0], outputs[cond_success_2sig].shape[0]

    # Compute the linear correlation coefficient
    r2 = r2_score(trues,outputs)
    err_rel = np.mean(np.abs((trues - outputs)/(trues)), axis=0)
    chi2s = (outputs-trues)**2./errors**2.
    chi2 = chi2s[chi2s<1.e4].mean()    # Remove some outliers which make explode the chi2
    print("R^2={:.2f}, Relative error={:.2e}, Chi2={:.2f}".format(r2, err_rel, chi2))
    print("A fraction of succeses of", successes1sig/tot_points, "at 1 sigma,", successes2sig/tot_points, "at 2 sigmas")

    # Sort by true value
    indsort = trues.argsort()
    outputs, trues, errors = outputs[indsort], trues[indsort], errors[indsort]

    # Compute mean and std region within several bins
    truebins, binsize = np.linspace(trues[0], trues[-1], num=10, retstep=True)
    means, stds = [], []
    for i, bin in enumerate(truebins[:-1]):
        cond = (trues>=bin) & (trues<bin+binsize)
        outbin = outputs[cond]
        if len(outbin)==0:
            outmean, outstd = np.nan, np.nan    # Avoid error message from some bins without points
        else:
            outmean, outstd = outbin.mean(), outbin.std()
        means.append(outmean); stds.append(outstd)
    means, stds = np.array(means), np.array(stds)
    print("Std in bins:",stds[~np.isnan(stds)].mean(),"Mean predicted uncertainty:", np.abs(errors.mean()))

    # Plot predictions vs true values
    #truemin, truemax = trues.min(), trues.max()
    truemin, truemax = minpar-0.05, maxpar+0.05
    #axscat.plot([truemin, truemax], [0., 0.], "r-")
    #axscat.errorbar(trues, outputs-trues, yerr=errors, color=col, marker="o", ls="none", markersize=0.5, elinewidth=0.5, zorder=10)
    axscat.plot([truemin, truemax], [truemin, truemax], "r-")
    axscat.errorbar(trues, outputs, yerr=errors, color=col, marker="o", ls="none", markersize=0.5, elinewidth=0.5, zorder=10)

    # Legend
    if cosmoparam=="Om":
        par = "\t"+r"$\Omega_m$"
    elif cosmoparam=="Sig":
        par = "\t"+r"$\sigma_8$"
    leg = par+"\n"+"$R^2$={:.2f}".format(r2)+"\n"+"$\epsilon$={:.1f} %".format(100.*err_rel)+"\n"+"$\chi^2$={:.2f}".format(chi2)
    #leg = par+"\n"+"$R^2$={:.2f}".format(r2)+"\n"+"$\epsilon$={:.2e}".format(err_rel)
    at = AnchoredText(leg, frameon=True, loc="upper left")
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    axscat.add_artist(at)

    # Labels etc
    axscat.set_xlim([truemin, truemax])
    axscat.set_ylim([truemin, truemax])
    if testsuite:
        axscat.set_ylim([truemin, truemax+0.1])
    axscat.set_ylabel(r"Prediction")
    axscat.set_xlabel(r"Truth")
    axscat.grid()

    # Title, indicating which are the training and testing suites
    if hparams.only_positions:
        usefeatures = "Only positions"
    else:
        usefeatures = r"Positions + $V_{\rm max}, M_*, R_*, Z_*$"
    props = dict(boxstyle='round', facecolor='white')#, alpha=0.5)
    axscat.text((minpar + maxpar)/2-0.02, minpar, usefeatures, color="k", bbox=props )


    if testsuite:
        titlefig = "Training in "+hparams.flip_suite()+", testing in "+suite
        #titlefig = "Cross test in "+suite+usefeatures
        namefig = "out_true_testsuite_"+cosmoparam+"_"+hparams.name_model()
    else:
        titlefig = "Training in "+suite+", testing in "+suite
        #titlefig = "Train in "+suite+usefeatures
        namefig = "out_true_"+cosmoparam+"_"+hparams.name_model()
    axscat.set_title(titlefig)

    figscat.savefig("Plots/"+namefig+".png", bbox_inches='tight', dpi=300)
    plt.close(figscat)

# Plot predicted and target power spectra
def plot_ps(hparams):

    figscat = plt.figure(figsize=(6,6))
    gs = gridspec.GridSpec(2,1,height_ratios=[6,2])
    gs.update(hspace=0.0)#,wspace=0.4,bottom=0.6,top=1.05)
    axscat=plt.subplot(gs[0])
    axerr=plt.subplot(gs[1])

    suite, simset = hparams.simsuite, hparams.simset
    col = colorsuite(suite)

    # Load true values and predicted means and standard deviations
    outputs = np.load("Outputs/outputsPS_"+hparams.name_model()+".npy")
    trues = np.load("Outputs/truesPS_"+hparams.name_model()+".npy")

    # There is a (0,0) initial point, fix it
    outputs = outputs[1:]
    trues = trues[1:]
    trues, outputs = 10.**trues, 10.**outputs

    # Compute the linear correlation coefficient
    r2 = r2_score(trues,outputs)
    err_rel = np.mean(np.abs((trues - outputs)/(trues)))
    print("R^2={:.2f}, Relative error={:.2e}".format(r2, err_rel))

    # Take 5 elements to plot randomly chosen
    npoints = 5
    indexes = np.random.choice(trues.shape[0], npoints, replace=False)
    outputs = outputs[indexes]
    trues = trues[indexes]
    colors = ["r","b","g","orange","purple"]

    kvec = np.loadtxt("PS_files/k_values.txt")

    for i in range(npoints):
        axscat.plot(kvec,trues[i], color=colors[i], linestyle="-")
        axscat.plot(kvec,outputs[i], color=colors[i], linestyle=":")

    # Labels etc
    axscat.set_yscale("log")
    axscat.set_xscale("log")
    axscat.set_ylabel(r"$P(k)$")
    axscat.grid(which="both", alpha=0.3)

    axscat.set_xlim(kvec[0],kvec[-1])

    customlegend = []
    customlegend.append( Line2D([0], [0], color="k", linestyle="-", lw=4, label=r"Truth") )
    customlegend.append( Line2D([0], [0], color="k", linestyle=":", lw=4, label=r"GNN") )
    axscat.legend(handles=customlegend, loc = "upper right")

    errk = np.mean(np.abs((trues - outputs)/(trues)), axis=0)*100.
    axerr.plot(kvec,errk, color="k", linestyle="-")

    axerr.set_xlim(kvec[0],kvec[-1])
    axerr.set_xscale("log")
    axerr.set_ylabel(r"$\epsilon$ (%)")
    axerr.set_xlabel(r"$k$ [hMpc$^{-1}$]")

    axerr.yaxis.set_minor_locator(MultipleLocator(2.5))
    axerr.grid(which="both", alpha=0.3)

    axscat.set_title(r"$R^2$={:.2f}".format(r2)+"$, \epsilon$={:.2f} %".format(100.*err_rel))

    figscat.savefig("Plots/"+namefig+".png", bbox_inches='tight', dpi=300)
    plt.close(figscat)

# Plot relative error of the power spectrum as a function of k
def plot_relerr(hparams):

    fig, ax = plt.subplots(figsize=(6,5))

    outputs = np.load("Outputs/outputsPS_"+hparams.name_model()+".npy")
    trues = np.load("Outputs/truesPS_"+hparams.name_model()+".npy")
    outputs = outputs[1:]
    trues = trues[1:]
    trues, outputs = 10.**trues, 10.**outputs

    r2 = r2_score(trues,outputs)
    err_rel = np.mean(np.abs((trues - outputs)/(trues)))#, axis=0)
    print("R2=",r2,"Errrel=",err_rel)

    errk = np.mean(np.abs((trues - outputs)/(trues)), axis=0)*100.
    kvec = np.loadtxt("PS_files/k_values.txt")
    ax.plot(kvec, errk)
    #plt.title("R2={:.2f}, Errrel={:.2f} %".format(r2,err_rel))
    ax.set_xlabel("$k$ [hMpc$^{-1}$]")
    ax.set_ylabel("Relative error (%)")
    ax.set_xscale("log")
    ax.set_xlim(kvec[0],kvec[-1])
    #plt.yscale("log")
    fig.savefig("Plots/rel_err_"+hparams.name_model()+".png", bbox_inches='tight', dpi=300)
    plt.close(fig)
