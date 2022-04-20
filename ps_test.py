#----------------------------------------------------
# Main routine for training and testing GNN models
# Author: Pablo Villanueva Domingo
# Last update: 10/11/21
#----------------------------------------------------

import time, datetime, psutil
from Source.metalayer import *
from Source.training import *
from Source.plotting import *
from Source.load_data import *
from visualize_graphs import visualize_graph
#import powerbox as pbox
import MAS_library as MASL
import Pk_library as PKL

vol = (boxsize/1.e3)**3.    # (Mpc/h)^3

#simtype = "NeymanScott"
#simtype = "Poisson"

# power spectrum parameters
BoxLen = 25.0
grid    = 512
MAS     = 'CIC'
kmax    = 20.0 #h/Mpc
axis    = 0
threads = 28

#--- Point processes ---#

def poisson_process(num_points):

    #pos = torch.rand((num_points,3))
    pos = np.random.uniform(0., 1., (num_points,3))

    return pos

# Generate a Neyman-Scott process with a gaussian kernel (Thomas point process)
# Based on https://hpaulkeeler.com/simulating-a-thomas-cluster-point-process/
def neynmanscott_process(num_parents, num_daughters, sigma):

    # Generate parents
    x_par = poisson_process(num_parents)

    #plt.scatter(x_par[:,0], x_par[:,1],color="r")

    # Simulate Poisson point process for the daughters (ie final point process)
    numbPointsDaughter = np.random.poisson(num_daughters, x_par.shape[0])
    numbPoints = sum(numbPointsDaughter) # total number of points

    # Generate the (relative) locations in Cartesian coordinates by
    # simulating independent normal variables
    x_daug = np.random.normal(0, sigma, size=(numbPoints,3)) # (relative) x coordinaets

    # replicate parent points (ie centres of disks/clusters)
    xx = np.repeat(x_par, numbPointsDaughter, axis=0)

    # translate points (ie parents points are the centres of cluster disks)
    xx = xx + x_daug

    # thin points if outside the simulation window
    booleInside=((xx[:,0]>=0)&(xx[:,0]<=1)&(xx[:,1]>=0)&(xx[:,1]<=1)&(xx[:,2]>=0)&(xx[:,2]<=1))
    # retain points inside simulation window
    xx = xx[booleInside]

    return xx

# Soneira-Peebles point process
def soneira_peebles_model(lamb, eta, n_levels, R0):

    Rparent = R0

    # Generate parents
    #num_parents = max(1,np.random.poisson(eta))
    num_parents = eta
    xparents = poisson_process(num_parents)

    #plt.scatter(xparents[:,0], xparents[:,1],s=2.,color="r",alpha=0.5)

    xtot = []
    xtot.extend(xparents)

    for n in range(2,n_levels+1):
        Rparent = Rparent/lamb
        pointsx = []

        #for i, j, in zip(xparents, yparents):
        for ipar in range(len(xparents)):

            num_points = np.random.poisson(eta)
            #num_points = eta
            x_daug = xparents[ipar] + np.random.normal(0, Rparent, size=(num_points,3))
            pointsx.extend(x_daug)

        xparents = pointsx
        xtot.extend(pointsx)

    xx = np.array(xtot)

    # retain points inside simulation window
    booleInside=((xx[:,0]>=0)&(xx[:,0]<=1)&(xx[:,1]>=0)&(xx[:,1]<=1)&(xx[:,2]>=0)&(xx[:,2]<=1))
    xx = xx[booleInside]

    return xx

def compute_ps(pos):

    pos = pos.cpu().detach().numpy()

    pos = pos*BoxLen

    # construct galaxy 3D density field
    delta = np.zeros((grid,grid,grid), dtype=np.float32)
    MASL.MA(pos, delta, BoxLen, MAS, verbose=False)
    delta /= np.mean(delta, dtype=np.float64)
    delta -= 1.0

    # compute Pk
    Pk  = PKL.Pk(delta, BoxLen, axis, MAS, threads, verbose=False)
    k   = Pk.k3D
    Pk0 = Pk.Pk[:,0] #monopole

    indexes = np.where(k<kmax)[0]
    return k[indexes], Pk0[indexes]

# Scatter plot of true vs predicted properties
def plot_ps_test(hparams, trues, outputs, ktrue, anals):

    figscat, axscat = plt.subplots()
    suite, simset = hparams.simsuite, hparams.simset
    col = colorsuite(suite)

    outputs = 10.**outputs

    #print(trues.shape,outputs.shape)

    # Compute the linear correlation coefficient
    r2 = r2_score(trues,outputs)
    #r2=0.
    err_rel = np.mean(np.abs((trues - outputs)/(trues)))
    print("R2",r2,"Rel Error",err_rel)

    indup = 4
    r2_up = r2_score(trues[indup:],outputs[indup:])
    #r2=0.
    err_rel_up = np.mean(np.abs((trues[indup:] - outputs[indup:])/(trues[indup:])))


    # Take 200 elements to plot randomly chosen
    npoints = 5    # 100
    indexes = np.random.choice(trues.shape[0], npoints, replace=False)
    outputs = outputs[indexes]
    trues = trues[indexes]
    #errors = errors[indexes]
    colors = ["r","b","g","orange","purple","cyan","m"]

    #kvec = np.linspace(0.3,20.,num=trues[0].shape[0])
    kvec = np.loadtxt("PS_files/k_values.txt")
    #print(trues.shape, outputs.shape)

    for i in range(npoints):
        #print(i, trues[i,:].shape)
        #axscat.plot(kvec,10.**trues[i], color="r", linestyle="-")
        #axscat.plot(kvec,10.**outputs[i], color="b", linestyle="--")
        #axscat.plot(kvec,10.**trues[i], color=colors[i], linestyle="-")
        #axscat.plot(ktrue,10.**trues[i], color=colors[i], linestyle="-")
        axscat.plot(ktrue,trues[i], color=colors[i], linestyle="-")
        axscat.plot(kvec,outputs[i], color=colors[i], linestyle="--")
        #axscat.plot(kvec,anals[i], color=colors[i], linestyle=":")

    axscat.set_yscale("log")
    axscat.set_xscale("log")
    axscat.set_ylabel(r"$P(k)$")
    axscat.set_xlabel(r"$k$ [hMpc$^{-1}$]")
    #plt.ylabel(r"log$_{10}\left[M_{h,infer}/(M_\odot/h)\right]$ - log$_{10}\left[M_{h,truth}/(M_\odot/h)\right]$")
    #plt.xlabel(r"log$_{10}\left[M_{h,truth}/(M_\odot/h)\right]$")
    #axscat.yaxis.set_major_locator(MultipleLocator(0.2))
    axscat.grid()

    #axscat.set_title(r"$R^2$={:.2f}".format(r2)+"$, \epsilon$={:.2f} %".format(100.*err_rel)+" ("+r"$R^2$={:.2f}".format(r2_up)+"$, \epsilon$={:.2f} %".format(100.*err_rel_up)+")")
    axscat.set_title(r"$R^2$={:.2f}".format(r2)+"$, \epsilon$={:.2f} %".format(100.*err_rel))


    titlefig = "Training in "+suite+" "+simset+", testing in "+suite+" "+simset
    namefig = "test_ps_"+simtype+"_"+hparams.name_model()
    #axscat.set_title(titlefig)

    figscat.savefig("Plots/"+namefig+".png", bbox_inches='tight', dpi=300)
    plt.close(figscat)

def generate_sim(num_points, hparams, simtype):

    if simtype=="Poisson":
        pos = poisson_process(num_points)

    elif simtype=="NeymanScott":
        sigma = np.random.uniform(0.01,0.1)
        pos = neynmanscott_process(num_parents=int(np.sqrt(num_points)), num_daughters=int(np.sqrt(num_points)), sigma=sigma)

    elif simtype=="SoneiraPeebles":
        pos = soneira_peebles_model(lamb=2, eta=6, n_levels=4, R0=.3)

    pos = torch.tensor(pos, dtype=torch.float32)

    edge_index, edge_attr = get_edges(pos, hparams.r_link, use_loops=False)

    x = torch.zeros_like(pos[:,:1], dtype=torch.float32)

    y = torch.ones((1,ps_size))*vol/num_points

    u = torch.tensor(np.log10(pos.shape[0]), dtype=torch.float32).reshape(1,1)

    # get the graph
    graph = Data(x=x,
                 y=y,
                 edge_index=torch.tensor(edge_index, dtype=torch.long),
                 edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
                 u=u,
                 pos=pos)

    return graph

# Plot sample distribution
def plot_pointprocess(simtype):

    num_sims = 10
    num_gals = np.random.randint(500,900,size=num_sims)
    #dataset = []

    for i in range(num_sims):
        data = generate_sim(num_gals[i], hparams, simtype)
        data.x = data.pos
        edge_index = radius_graph(data.pos, r=hparams.r_link, loop=False)
        visualize_graph(data, simtype+"_"+str(i), 0.1, "3d", edge_index)
        #dataset.append(data)

# Main routine to train the neural net
# If testsuite==True, it takes a model already pretrained in the other suite and tests it in the selected one
def test_ps(hparams, simtype):

    hparams.outmode = "ps"
    hparams.only_positions = 1
    dim_out=ps_size

    # Initialize model
    #model = ModelGNN(use_model, node_features, n_layers, r_link)
    model = GNN(node_features=0,
                n_layers=hparams.n_layers,
                hidden_channels=hparams.hidden_channels,
                linkradius=hparams.r_link,
                dim_out=dim_out,
                only_positions=hparams.only_positions)
    model.to(device)

    # Load the trained model
    hparams.simsuite = hparams.flip_suite()
    state_dict = torch.load("Models/"+hparams.name_model(), map_location=device)
    model.load_state_dict(state_dict)

    num_sims = 20
    num_gals = np.random.randint(500,900,size=num_sims)
    #num_gals = np.random.randint(10,100,size=num_sims)
    dataset = []

    trues = []
    outs = []
    anals = []

    for i in range(num_sims):
        data = generate_sim(num_gals[i], hparams, simtype)
        dataset.append(data)

    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    for data in loader:
        data.to(device)
        out = model(data)
        outs.append(out.cpu().detach().numpy())

        #ps_true, ktrue = pbox.tools.get_power(data.x.cpu().detach().numpy(), boxsize/1.e3 ,N=data.x.shape[0],bins=ps_size,remove_shotnoise=False)
        #print(ps_true)
        ktrue, ps_true = compute_ps(data.pos)
        trues.append( ps_true )

        anals.append( data.y.cpu().detach().numpy() )

    hparams.simsuite = hparams.flip_suite()


    trues = np.array(trues)
    #trues = np.array(trues).reshape((num_sims,ps_size))
    outs = np.array(outs).reshape((num_sims,ps_size))
    anals = np.array(anals).reshape((num_sims,ps_size))

    plot_ps_test(hparams, trues, outs, ktrue, anals)


    # Plot sample distribution
    #data = dataset[0]
    """for j in range(10):
        data = dataset[j]
        data.x = data.pos
        edge_index = radius_graph(data.pos, r=hparams.r_link, loop=False)
        visualize_graph(data, simtype+"_"+str(j), 0.1, "3d", edge_index)"""





#--- MAIN ---#

if __name__ == "__main__":

    time_ini = time.time()

    for path in ["Plots", "Models", "Outputs"]:
        if not os.path.exists(path):
            os.mkdir(path)

    # Load default parameters
    from hyperparameters import hparams

    for simtype in ["Poisson", "NeymanScott","SoneiraPeebles"]:
    #for simtype in ["SoneiraPeebles"]:

        test_ps(hparams, simtype)

        #plot_pointprocess(simtype)

    print("Finished. Time elapsed:",datetime.timedelta(seconds=time.time()-time_ini))
