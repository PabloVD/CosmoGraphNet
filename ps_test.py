#----------------------------------------------------
# Compute the power spectrum of different point distirbutions with the GNN trained in CAMELS
# Author: Pablo Villanueva Domingo
# Last update: 4/22
#----------------------------------------------------

import time, datetime
from Source.metalayer import *
from Source.training import *
from Source.plotting import *
from Source.load_data import *
from visualize_graphs import visualize_graph
#import powerbox as pbox
import MAS_library as MASL
import Pk_library as PKL

# Power spectrum parameters
BoxLen = 25.0
grid    = 512
MAS     = 'CIC'
kmax    = 20.0 #h/Mpc
axis    = 0
threads = 28
vol = (boxsize/1.e3)**3.    # (Mpc/h)^3


#--- POINT DISTRIBUTIONS ---#

# Poisson point process
def poisson_process(num_points):

    #pos = torch.rand((num_points,3))
    pos = np.random.uniform(0., 1., (num_points,3))

    return pos

# Neyman-Scott process with a gaussian kernel (Thomas point process)
# Based on https://hpaulkeeler.com/simulating-a-thomas-cluster-point-process/
def neynmanscott_process(num_parents, num_daughters, sigma):

    # Generate parents
    x_par = poisson_process(num_parents)

    # Simulate Poisson point process for the daughters
    numbPointsDaughter = np.random.poisson(num_daughters, x_par.shape[0])
    numbPoints = sum(numbPointsDaughter)

    # Generate the relative locations as independent normal variables
    x_daug = np.random.normal(0, sigma, size=(numbPoints,3)) # (relative) x coordinaets

    # Replicate parent points (ie centres of disks/clusters) and center daughters around them
    xx = np.repeat(x_par, numbPointsDaughter, axis=0)
    xx = xx + x_daug

    # Retain only those points inside simulation window
    booleInside=((xx[:,0]>=0)&(xx[:,0]<=1)&(xx[:,1]>=0)&(xx[:,1]<=1)&(xx[:,2]>=0)&(xx[:,2]<=1))
    xx = xx[booleInside]

    return xx

# Soneira-Peebles point process (Soneira & Peebles 1977, 1978)
def soneira_peebles_model(lamb, eta, n_levels, R0):

    # Radius for first level
    Rparent = R0

    # Generate parents
    #num_parents = max(1,np.random.poisson(eta))
    num_parents = eta
    xparents = poisson_process(num_parents)

    xtot = []
    xtot.extend(xparents)

    # Iterate over each level
    for n in range(2,n_levels+1):
        Rparent = Rparent/lamb
        pointsx = []

        for ipar in range(len(xparents)):

            num_points = np.random.poisson(eta)
            #num_points = eta
            x_daug = xparents[ipar] + np.random.normal(0, Rparent, size=(num_points,3))
            pointsx.extend(x_daug)

        xparents = pointsx
        xtot.extend(pointsx)

    xx = np.array(xtot)

    # Retain only those points inside simulation window
    booleInside=((xx[:,0]>=0)&(xx[:,0]<=1)&(xx[:,1]>=0)&(xx[:,1]<=1)&(xx[:,2]>=0)&(xx[:,2]<=1))
    xx = xx[booleInside]

    return xx

#--- OTHER ROUTINES ---#

# Routine to compute the power spectrum using Pylians
def compute_ps(pos):

    pos = pos.cpu().detach().numpy()

    pos = pos*BoxLen

    # Construct galaxy 3D density field
    delta = np.zeros((grid,grid,grid), dtype=np.float32)
    MASL.MA(pos, delta, BoxLen, MAS, verbose=False)
    delta /= np.mean(delta, dtype=np.float64)
    delta -= 1.0

    # Compute the power spectrum
    Pk  = PKL.Pk(delta, BoxLen, axis, MAS, threads, verbose=False)
    k   = Pk.k3D
    Pk0 = Pk.Pk[:,0] # Monopole

    indexes = np.where(k<kmax)[0]
    return k[indexes], Pk0[indexes]

# Scatter plot of true vs predicted properties
def plot_ps_test(hparams, trues, outputs, ktrue, anals):

    figscat, axscat = plt.subplots(figsize=(6,5))
    suite, simset = hparams.simsuite, hparams.simset
    col = colorsuite(suite)

    # Compute the linear correlation coefficient
    r2 = r2_score(trues,outputs)
    #r2=0.
    err_rel = np.mean(np.abs((trues - outputs)/(trues)))
    print("R2",r2,"Rel Error",err_rel)

    # Take 5 samples to plot randomly chosen
    npoints = 5
    indexes = np.random.choice(trues.shape[0], npoints, replace=False)
    outputs = outputs[indexes]
    trues = trues[indexes]
    colors = ["r","b","g","orange","purple","cyan","m"]

    kvec = np.loadtxt("PS_files/k_values.txt")

    for i in range(npoints):
        axscat.plot(ktrue,trues[i], color=colors[i], linestyle="-")
        axscat.plot(kvec,outputs[i], color=colors[i], linestyle=":")

    axscat.set_yscale("log")
    axscat.set_xscale("log")
    axscat.set_ylabel(r"$P(k)$")
    axscat.set_xlabel(r"$k$ [hMpc$^{-1}$]")
    axscat.grid()
    axscat.set_xlim(kvec[0],kvec[-1])

    customlegend = []
    customlegend.append( Line2D([0], [0], color="k", linestyle="-", lw=4, label=r"Truth") )
    customlegend.append( Line2D([0], [0], color="k", linestyle=":", lw=4, label=r"GNN") )
    axscat.legend(handles=customlegend, loc = "upper right")

    axscat.set_title(r"$R^2$={:.2f}".format(r2)+"$, \epsilon$={:.2f} %".format(100.*err_rel))

    namefig = "test_ps_"+simtype+"_"+hparams.name_model()
    figscat.savefig("Plots/"+namefig+".png", bbox_inches='tight', dpi=300)
    plt.close(figscat)

# Generate synthetic galaxy catalogue as point process
def generate_sim(num_points, hparams, simtype):

    if simtype=="Poisson":
        pos = poisson_process(num_points)
        print("Points",num_points)

    elif simtype=="NeymanScott":
        sigma = np.random.uniform(0.01,0.1)
        pos = neynmanscott_process(num_parents=int(np.sqrt(num_points)), num_daughters=int(np.sqrt(num_points)), sigma=sigma)
        print("Points",num_points,"Sigma",sigma)

    elif simtype=="SoneiraPeebles":
        n_levels = np.random.randint(4, 6)
        eta = int(num_points**(1/n_levels))
        lamb = np.random.uniform(1.5,3.)
        pos = soneira_peebles_model(lamb=lamb, eta=eta, n_levels=n_levels, R0=.3)
        print("Points",num_points,"eta",eta,"levels",n_levels,"lambda",lamb)

    pos = torch.tensor(pos, dtype=torch.float32)

    edge_index, edge_attr = get_edges(pos, hparams.r_link, use_loops=False)

    x = torch.zeros_like(pos[:,:1], dtype=torch.float32)    # Not used, only for consistency

    y = torch.ones((1,ps_size))*vol/num_points

    u = torch.tensor(np.log10(pos.shape[0]), dtype=torch.float32).reshape(1,1)

    # Get the graph
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

    for i in range(num_sims):
        data = generate_sim(num_gals[i], hparams, simtype)
        data.x = data.pos
        edge_index = radius_graph(data.pos, r=hparams.r_link, loop=False)
        visualize_graph(data, simtype+"_"+str(i), 0.1, "3d", edge_index)

# Routine to predict the power spectrum using the pretrained GNN
def test_ps(hparams, simtype):

    hparams.outmode = "ps"
    hparams.only_positions = 1
    dim_out=ps_size

    # Initialize model
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

    num_sims = 50
    num_gals = np.random.randint(700,1200,size=num_sims)
    dataset = []

    trues = []
    outs = []
    anals = []

    for i in range(num_sims):

        data = generate_sim(num_gals[i], hparams, simtype)
        loader = DataLoader([data], batch_size=1, shuffle=True)

        for data in loader:

            # Get model prediction
            data.to(device)
            out = model(data)
            out = 10.**out
            outs.append(out.cpu().detach().numpy())

            # Get target power spectrum
            ktrue, ps_true = compute_ps(data.pos)
            trues.append( ps_true )

            anals.append( data.y.cpu().detach().numpy() )

            print(i,"Err rel={:.3e}".format( np.mean(np.abs((ps_true - out.cpu().detach().numpy())/(ps_true)))) )

    hparams.simsuite = hparams.flip_suite()


    trues = np.array(trues)
    outs = np.array(outs).reshape((num_sims,ps_size))
    anals = np.array(anals).reshape((num_sims,ps_size))

    plot_ps_test(hparams, trues, outs, ktrue, anals)



#--- MAIN ---#

if __name__ == "__main__":

    time_ini = time.time()

    for path in ["Plots", "Models", "Outputs"]:
        if not os.path.exists(path):
            os.mkdir(path)

    # Load default parameters
    from hyperparameters import hparams

    for simtype in ["Poisson", "NeymanScott","SoneiraPeebles"]:

        test_ps(hparams, simtype)

        #plot_pointprocess(simtype)

    print("Finished. Time elapsed:",datetime.timedelta(seconds=time.time()-time_ini))
