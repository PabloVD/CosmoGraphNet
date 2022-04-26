#----------------------------------------------------------------------
# Script to visualize galaxy catalogues as graphs
# Author: Pablo Villanueva Domingo
# Last update: 4/22
#----------------------------------------------------------------------

import time, datetime
from Source.networks import *
from Source.plotting import *
from Source.load_data import *
from torch_geometric.utils import degree

fontsize = 8

# Visualization routine for plotting graphs
def visualize_graph(data, ind, sizes=0.1, projection="3d", edge_index=None):

    fig = plt.figure(figsize=(4, 4))

    if projection=="3d":
        ax = fig.add_subplot(projection ="3d")
        pos = data.x[:,:3]
    elif projection=="2d":
        ax = fig.add_subplot()
        pos = data.x[:,:2]

    pos *= boxsize/1.e3   # show in Mpc

    # Draw lines for each edge
    if edge_index is not None:
        for (src, dst) in edge_index.t().tolist():

            src = pos[src].tolist()
            dst = pos[dst].tolist()

            if projection=="3d":
                ax.plot([src[0], dst[0]], [src[1], dst[1]], zs=[src[2], dst[2]], linewidth=0.1, color='black')
            elif projection=="2d":
                ax.plot([src[0], dst[0]], [src[1], dst[1]], linewidth=0.1, color='black')

    # Plot nodes
    if projection=="3d":
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=sizes, zorder=1000, alpha=0.5)
    elif projection=="2d":
        ax.scatter(pos[:, 0], pos[:, 1], s=sizes, zorder=1000, alpha=0.5)

    ax.xaxis.set_tick_params(labelsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize)
    ax.zaxis.set_tick_params(labelsize=fontsize)

    fig.savefig("Plots/visualize_graph_"+str(ind), bbox_inches='tight', dpi=300)
    plt.close(fig)

# Plot the degree distribution of the graph (see e.g. http://networksciencebook.com/)
def plot_degree_distribution(degrees):

    listbins = np.linspace(0,80,num=12)
    deg_dist = []

    for array in degrees:
        hist, bins = np.histogram(array, bins=listbins)
        deg_dist.append(hist)

    dist_mean = np.mean(deg_dist,axis=0)
    dist_std = np.std(deg_dist,axis=0)

    fig_deg, ax_deg = plt.subplots(figsize=(6, 4))

    ax_deg.set_yscale("log")
    #ax_deg.set_xscale("log")
    ax_deg.plot(bins[:-1], dist_mean, color=colorsuite(simsuite))
    ax_deg.fill_between(bins[:-1], dist_mean+dist_std, dist_mean-dist_std, color=colorsuite(simsuite), alpha=0.3)
    ax_deg.set_xlim([bins[0],bins[-2]])
    ax_deg.set_xlabel(r"$k$")
    ax_deg.set_ylabel(r"$p_k$")

    fig_deg.savefig("Plots/degree_distribution.pdf", bbox_inches='tight', dpi=300)

# Main routine to display graphs from several simulations
def display_graphs(simsuite, n_sims, r_link, simset="LH", showgraph=True, get_degree=False):

    if get_degree:
        degrees = []

    # Load data and create graph
    for simnumber in range(n_sims):
        simpath = simpathroot + simsuite + "/"+simset+"_"
        catalogue = simpath + str(simnumber)+"/fof_subhalo_tab_033.hdf5"

        # Read the catalogue
        f     = h5py.File(catalogue, 'r')
        pos   = f['/Subhalo/SubhaloPos'][:]/boxsize
        Nstar = f['/Subhalo/SubhaloLenType'][:,4]       #number of stars
        Mstar = f['/Subhalo/SubhaloMassType'][:,4] #Msun/h
        indexes = np.where(Nstar>Nstar_th)[0]
        pos     = pos[indexes]
        Mstar   = Mstar[indexes]

        tab = np.column_stack((pos, Mstar))

        #edge_index, edge_attr = get_edges(pos, r_link, use_loops=False)
        edge_index = radius_graph(torch.tensor(pos,dtype=torch.float32), r=r_link, loop=False)

        data = Data(x=tab, edge_index=torch.tensor(edge_index, dtype=torch.long))

        if showgraph:
            #visualize_graph(data, simnumber, "2d", edge_index)
            visualize_graph(data, simnumber, projection="3d", edge_index=data.edge_index)

        if get_degree:
            degrees.append( degree(edge_index[0], data.num_nodes).numpy() )

    if get_degree:
        plot_degree_distribution(degrees)




#--- MAIN ---#

if __name__=="__main__":

    time_ini = time.time()

    for path in ["Plots"]:
        if not os.path.exists(path):
            os.mkdir(path)

    # Linking radius
    r_link = 0.05
    # Simulation suite, choose between "IllustrisTNG" and "SIMBA"
    simsuite = "IllustrisTNG"
    # Number of simulations considered, maximum 27 for CV and 1000 for LH
    n_sims = 20

    display_graphs(simsuite, n_sims, r_link)

    print("Finished. Time elapsed:",datetime.timedelta(seconds=time.time()-time_ini))
