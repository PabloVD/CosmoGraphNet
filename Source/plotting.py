#----------------------------------------------------------------------
# Script for plotting some statistics
# Author: Pablo Villanueva Domingo
# Last update: 10/11/21
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

def denormalize(trues, outputs, errors, minpar, maxpar):

    trues = minpar + trues*(maxpar - minpar)
    outputs = minpar + outputs*(maxpar - minpar)
    errors = errors*(maxpar - minpar)
    return trues, outputs, errors

# Scatter plot of true vs predicted properties
def plot_out_true_scatter(hparams, cosmoparam, testsuite = False):

    figscat, axscat = plt.subplots()
    suite, simset = hparams.simsuite, hparams.simset
    col = colorsuite(suite)
    #if testsuite: col = colorsuite(hparams.flip_suite())

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
        axscat.set_ylim([truemin, truemax+0.3])
    axscat.set_ylabel(r"Prediction")
    axscat.set_xlabel(r"Truth")
    axscat.grid()

    # Title, indicating which are the training and testing suites
    if hparams.only_positions:
        endtitle = ", only positions"
    else:
        endtitle = ", galactic features"
    if testsuite:
        #titlefig = "Training in "+hparams.flip_suite()+" "+simset+", testing in "+suite+" "+simset
        titlefig = "Cross test in "+suite+endtitle
        namefig = "out_true_testsuite_"+cosmoparam+"_"+hparams.name_model()
    else:
        #titlefig = "Training in "+suite+" "+simset+", testing in "+suite+" "+simset
        titlefig = "Train in "+suite+endtitle
        namefig = "out_true_"+cosmoparam+"_"+hparams.name_model()
    axscat.set_title(titlefig)

    figscat.savefig("Plots/"+namefig+".png", bbox_inches='tight', dpi=300)
    plt.close(figscat)

# Scatter plot of true vs predicted properties
def plot_ps(hparams, testsuite = False):

    figscat, axscat = plt.subplots(figsize=(8,8))
    suite, simset = hparams.simsuite, hparams.simset
    col = colorsuite(suite)
    if testsuite: col = colorsuite(hparams.flip_suite())

    # Load true values and predicted means and standard deviations
    outputs = np.load("Outputs/outputsPS_"+hparams.name_model()+".npy")
    trues = np.load("Outputs/truesPS_"+hparams.name_model()+".npy")
    #errors = np.load("Outputs/errors"+cosmoparam+"_"+hparams.name_model()+".npy")

    # There is a (0,0) initial point, fix it
    outputs = outputs[1:]
    trues = trues[1:]
    #errors = errors[1:]
    trues, outputs = 10.**trues, 10.**outputs

    """if cosmoparam=="Om":
        minpar, maxpar = 0.1, 0.5
    elif cosmoparam=="Sig":
        minpar, maxpar = 0.6, 1.0
    trues, outputs, errors = denormalize(trues, outputs, errors, minpar, maxpar)"""

    # Compute the number of points lying within 1 or 2 sigma regions from their uncertainties
    """cond_success_1sig, cond_success_2sig = np.abs(outputs-trues)<=np.abs(errors), np.abs(outputs-trues)<=2.*np.abs(errors)
    tot_points = outputs.shape[0]
    successes1sig, successes2sig = outputs[cond_success_1sig].shape[0], outputs[cond_success_2sig].shape[0]"""

    # Compute the linear correlation coefficient
    r2 = r2_score(trues,outputs)
    err_rel = np.mean(np.abs((trues - outputs)/(trues)))
    #err_rel = np.mean(np.abs((trues - outputs)/(trues)), axis=0)
    #chi2s = (outputs-trues)**2./errors**2.
    #chi2 = chi2s[np.abs(errors)>0.01].mean()    # Remove some outliers which make explode the chi2
    #chi2 = chi2s[chi2s<1.e4].mean()    # Remove some outliers which make explode the chi2
    print("R^2={:.2f}, Relative error={:.2e}".format(r2, err_rel))
    #print("A fraction of succeses of", successes1sig/tot_points, "at 1 sigma,", successes2sig/tot_points, "at 2 sigmas")

    # Sort by true value
    #indsort = trues.argsort()
    #print(indsort.shape)
    #outputs, trues = outputs[indsort,:], trues[indsort,:]

    # Compute mean and std region within several bins
    """truebins, binsize = np.linspace(trues[0], trues[-1], num=10, retstep=True)
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
    #means -= (truebins[:-1]+binsize/2.)
    #axscat.fill_between(truebins[:-1]+binsize/2., means-stds, means+stds, color=col, alpha=0.2)
    #axscat.plot(truebins[:-1]+binsize/2., means, color=col, linestyle="--")
    print("Std in bins:",stds[~np.isnan(stds)].mean(),"Mean predicted uncertainty:", np.abs(errors.mean()))"""

    # Take 200 elements to plot randomly chosen
    npoints = 5    # 100
    indexes = np.random.choice(trues.shape[0], npoints, replace=False)
    outputs = outputs[indexes]
    trues = trues[indexes]
    #errors = errors[indexes]
    colors = ["r","b","g","orange","purple"]

    # Plot predictions vs true values
    #truemin, truemax = trues.min(), trues.max()
    #axscat.plot([truemin, truemax], [0., 0.], "r-")
    #axscat.errorbar(trues, outputs-trues, yerr=errors, color=col, marker="o", ls="none", markersize=0.5, elinewidth=0.5, zorder=10)
    #axscat.errorbar(trues, outputs, yerr=errors, color=col, marker="o", ls="none", markersize=0.5, elinewidth=0.5, zorder=10)

    #kvec = range(79)
    #kvec = np.linspace(0.3,20.,num=trues[0].shape[0])
    kvec = np.loadtxt("PS_files/k_values.txt")
    #print(trues.shape, outputs.shape)

    for i in range(npoints):
        #print(i, trues[i,:].shape)
        axscat.plot(kvec,trues[i], color=colors[i], linestyle="-")
        axscat.plot(kvec,outputs[i], color=colors[i], linestyle="--")

    # Legend
    """if cosmoparam=="Om":
        par = "\t"+r"$\Omega_m$"
    elif cosmoparam=="Sig":
        par = "\t"+r"$\sigma_8$"
    #leg = par+"\n"+"$R^2$={:.2f}".format(r2)+"\n"+"$\epsilon$={:.1f} %".format(100.*err_rel)+"\n"+"$\chi^2$={:.2f}".format(chi2)
    leg = par+"\n"+"$R^2$={:.2f}".format(r2)+"\n"+"$\epsilon$={:.1f} %".format(100.*err_rel)#+"\n"+"$\chi^2$={:.2f}".format(chi2)
    at = AnchoredText(leg, frameon=True, loc="upper left")
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    axscat.add_artist(at)"""

    # Labels etc
    #axscat.set_xlim([truemin, truemax])
    #axscat.set_ylim([-1.,1.])
    #axscat.set_ylabel(r"Prediction - Truth")
    axscat.set_yscale("log")
    axscat.set_xscale("log")
    axscat.set_ylabel(r"$P(k)$")
    axscat.set_xlabel(r"$k$ [hMpc$^{-1}$]")
    #plt.ylabel(r"log$_{10}\left[M_{h,infer}/(M_\odot/h)\right]$ - log$_{10}\left[M_{h,truth}/(M_\odot/h)\right]$")
    #plt.xlabel(r"log$_{10}\left[M_{h,truth}/(M_\odot/h)\right]$")
    #axscat.yaxis.set_major_locator(MultipleLocator(0.2))
    axscat.grid()

    axscat.set_xlim(kvec[0],kvec[-1])

    customlegend = []
    customlegend.append( Line2D([0], [0], color="k", linestyle="-", lw=4, label=r"Truth") )
    customlegend.append( Line2D([0], [0], color="k", linestyle="--", lw=4, label=r"GNN") )
    axscat.legend(handles=customlegend, loc = "upper right")

    # Title, indicating which are the training and testing suites
    if testsuite:
        titlefig = "Training in "+hparams.flip_suite()+" "+simset+", testing in "+suite+" "+simset
        namefig = "ps_testsuite_"+hparams.name_model()
    else:
        titlefig = "Training in "+suite+" "+simset+", testing in "+suite+" "+simset
        namefig = "ps_"+hparams.name_model()
    axscat.set_title(titlefig)#"""
    axscat.set_title(r"$R^2$={:.2f}".format(r2)+"$, \epsilon$={:.2f} %".format(100.*err_rel))


    figscat.savefig("Plots/"+namefig+".png", bbox_inches='tight', dpi=300)
    plt.close(figscat)

def plot_relerr(hparams):

    fig, ax = plt.subplots(figsize=(8,8))

    outputs = np.load("Outputs/outputsPS_"+hparams.name_model()+".npy")
    trues = np.load("Outputs/truesPS_"+hparams.name_model()+".npy")
    outputs = outputs[1:]
    trues = trues[1:]
    trues, outputs = 10.**trues, 10.**outputs

    r2 = r2_score(trues,outputs)
    err_rel = np.mean(np.abs((trues - outputs)/(trues)))#, axis=0)
    print("R2=",r2,"Errrel=",err_rel)

    errk = np.mean(np.abs((trues - outputs)/(trues)), axis=0)*100.
    #kvec = np.linspace(0.3,20.,num=errk.shape[0])
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
