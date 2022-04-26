#----------------------------------------------------
# Routine for loading the CAMELS galaxy catalogues
# Author: Pablo Villanueva Domingo
# Last update: 4/22
#----------------------------------------------------

import h5py
from torch_geometric.data import Data, DataLoader
from Source.constants import *
from Source.plotting import *
import scipy.spatial as SS

Nstar_th = 20   # Minimum number of stellar particles required to consider a galaxy

# Normalize CAMELS parameters
def normalize_params(params):

    minimum = np.array([0.1, 0.6, 0.25, 0.25, 0.5, 0.5])
    maximum = np.array([0.5, 1.0, 4.00, 4.00, 2.0, 2.0])
    params = (params - minimum)/(maximum - minimum)
    return params

# Normalize power spectrum
def normalize_ps(ps):
    mean, std = ps.mean(axis=0), ps.std(axis=0)
    normps = (ps - mean)/std
    return normps

# Compute KDTree and get edges and edge features
def get_edges(pos, r_link, use_loops):

    # 1. Get edges

    # Create the KDTree and look for pairs within a distance r_link
    # Boxsize normalize to 1
    kd_tree = SS.KDTree(pos, leafsize=16, boxsize=1.0001)
    edge_index = kd_tree.query_pairs(r=r_link, output_type="ndarray")

    # Add reverse pairs
    reversepairs = np.zeros((edge_index.shape[0],2))
    for i, pair in enumerate(edge_index):
        reversepairs[i] = np.array([pair[1], pair[0]])
    edge_index = np.append(edge_index, reversepairs, 0)

    edge_index = edge_index.astype(int)

    # Write in pytorch-geometric format
    edge_index = edge_index.reshape((2,-1))
    num_pairs = edge_index.shape[1]

    # 2. Get edge attributes

    row, col = edge_index
    diff = pos[row]-pos[col]

    # Take into account periodic boundary conditions, correcting the distances
    for i, pos_i in enumerate(diff):
        for j, coord in enumerate(pos_i):
            if coord > r_link:
                diff[i,j] -= 1.  # Boxsize normalize to 1
            elif -coord > r_link:
                diff[i,j] += 1.  # Boxsize normalize to 1

    # Get translational and rotational invariant features
    # Distance
    dist = np.linalg.norm(diff, axis=1)
    # Centroid of galaxy catalogue
    centroid = np.mean(pos,axis=0)
    # Unit vectors of node, neighbor and difference vector
    unitrow = (pos[row]-centroid)/np.linalg.norm((pos[row]-centroid), axis=1).reshape(-1,1)
    unitcol = (pos[col]-centroid)/np.linalg.norm((pos[col]-centroid), axis=1).reshape(-1,1)
    unitdiff = diff/dist.reshape(-1,1)
    # Dot products between unit vectors
    cos1 = np.array([np.dot(unitrow[i,:].T,unitcol[i,:]) for i in range(num_pairs)])
    cos2 = np.array([np.dot(unitrow[i,:].T,unitdiff[i,:]) for i in range(num_pairs)])
    # Normalize distance by linking radius
    dist /= r_link

    # Concatenate to get all edge attributes
    edge_attr = np.concatenate([dist.reshape(-1,1), cos1.reshape(-1,1), cos2.reshape(-1,1)], axis=1)

    # Add loops
    if use_loops:
        loops = np.zeros((2,pos.shape[0]),dtype=int)
        atrloops = np.zeros((pos.shape[0],3))
        for i, posit in enumerate(pos):
            loops[0,i], loops[1,i] = i, i
            atrloops[i,0], atrloops[i,1], atrloops[i,2] = 0., 1., 0.
        edge_index = np.append(edge_index, loops, 1)
        edge_attr = np.append(edge_attr, atrloops, 0)
    edge_index = edge_index.astype(int)

    return edge_index, edge_attr


# Routine to create a cosmic graph from a galaxy catalogue
# simnumber: number of simulation
# param_file: file with the value of the cosmological + astrophysical parameters
# hparams: hyperparameters class
def sim_graph(simnumber, param_file, hparams):

    # Get some hyperparameters
    simsuite,simset,r_link,only_positions,outmode,pred_params = hparams.simsuite,hparams.simset,hparams.r_link,hparams.only_positions,hparams.outmode,hparams.pred_params

    # Name of the galaxy catalogue
    simpath = simpathroot + simsuite + "/"+simset+"_"
    catalogue = simpath + str(simnumber)+"/fof_subhalo_tab_0"+hparams.snap+".hdf5"

    # Read the catalogue
    f     = h5py.File(catalogue, 'r')
    pos   = f['/Subhalo/SubhaloPos'][:]/boxsize
    Mstar = f['/Subhalo/SubhaloMassType'][:,4] #Msun/h
    Rstar = f["Subhalo/SubhaloHalfmassRadType"][:,4]
    Metal = f["Subhalo/SubhaloStarMetallicity"][:]
    Vmax = f["Subhalo/SubhaloVmax"][:]
    Nstar = f['/Subhalo/SubhaloLenType'][:,4]       #number of stars
    f.close()

    # Some simulations are slightly outside the box, correct it
    pos[np.where(pos<0.0)]+=1.0
    pos[np.where(pos>1.0)]-=1.0

    # Select only galaxies with more than Nstar_th star particles
    indexes = np.where(Nstar>Nstar_th)[0]
    pos     = pos[indexes]
    Mstar   = Mstar[indexes]
    Rstar   = Rstar[indexes]
    Metal   = Metal[indexes]
    Vmax   = Vmax[indexes]

    # Get the output to be predicted by the GNN, either the cosmo parameters or the power spectrum
    if outmode=="cosmo":
        # Read the value of the cosmological & astrophysical parameters
        paramsfile = np.loadtxt(param_file, dtype=str)
        params = np.array(paramsfile[simnumber,1:-1],dtype=np.float32)
        params = normalize_params(params)
        params = params[:pred_params]   # Consider only the first parameters, up to pred_params
        y = np.reshape(params, (1,params.shape[0]))

    # Read the power spectra
    elif outmode=="ps":

        ps = np.load(param_file)
        ps = ps[simnumber]
        ps = np.log10(ps)
        #ps = normalize_ps(ps)
        y = np.reshape(ps, (1,ps_size))

    # Number of galaxies as global feature
    u = np.log10(pos.shape[0]).reshape(1,1)

    Mstar = np.log10(1.+ Mstar)
    Rstar = np.log10(1.+ Rstar)
    Metal = np.log10(1.+ Metal)
    Vmax = np.log10(1. + Vmax)

    # Node features
    tab = np.column_stack((Mstar, Rstar, Metal, Vmax))
    #tab = Vmax.reshape(-1,1)       # For using only Vmax
    x = torch.tensor(tab, dtype=torch.float32)

    # Use loops if node features are considered only
    if only_positions:
        tab = np.zeros_like(pos[:,:1])   # Node features not really used
        use_loops = False
    else:
        use_loops = True

    # Get edges and edge features
    edge_index, edge_attr = get_edges(pos, r_link, use_loops)

    # Construct the graph
    graph = Data(x=x,
                 y=torch.tensor(y, dtype=torch.float32),
                 u=torch.tensor(u, dtype=torch.float32),
                 edge_index=torch.tensor(edge_index, dtype=torch.long),
                 edge_attr=torch.tensor(edge_attr, dtype=torch.float32))

    return graph


# Split training and validation sets
def split_datasets(dataset):

    random.shuffle(dataset)

    num_train = len(dataset)
    split_valid = int(np.floor(valid_size * num_train))
    split_test = split_valid + int(np.floor(test_size * num_train))

    train_dataset = dataset[split_test:]
    valid_dataset = dataset[:split_valid]
    test_dataset = dataset[split_valid:split_test]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader, test_loader

######################################################################################

# Main routine to load data and create the dataset
def create_dataset(hparams):

    # Target file depending on the task: inferring cosmo parameters or predicting power spectrum
    if hparams.outmode == "cosmo":
        param_file = "/projects/QUIJOTE/CAMELS/Sims/CosmoAstroSeed_params_"+hparams.simsuite+".txt"
    elif hparams.outmode == "ps":
        param_file = "PS_files/Pk_galaxies_"+hparams.simsuite+"_LH_"+hparams.snap+"_kmax=20.0.npy"

    dataset = []

    for simnumber in range(hparams.n_sims):
        dataset.append(sim_graph(simnumber,param_file,hparams))

    # Add the other suite for predicting the power spectrum
    if hparams.outmode == "ps":
        hparams.simsuite = hparams.flip_suite()
        param_file = "PS_files/Pk_galaxies_"+hparams.simsuite+"_LH_"+hparams.snap+"_kmax=20.0.npy"

        for simnumber in range(hparams.n_sims):
            dataset.append(sim_graph(simnumber,param_file,hparams))

        # Add other snapshots from other redshifts
        # Snapshot redshift
        # 004: z=3, 010: z=2, 014: z=1.5, 018: z=1, 024: z=0.5, 033: z=0
        #for snap in [24,18,14,10]:
        for snap in [18,10]:

            hparams.snap = str(snap)

            param_file = "PS_files/Pk_galaxies_"+hparams.simsuite+"_LH_"+hparams.snap+"_kmax=20.0.npy"

            for simnumber in range(hparams.n_sims):
                dataset.append(sim_graph(simnumber,param_file,hparams))

            hparams.simsuite = hparams.flip_suite()
            param_file = "PS_files/Pk_galaxies_"+hparams.simsuite+"_LH_"+hparams.snap+"_kmax=20.0.npy"

            for simnumber in range(hparams.n_sims):
                dataset.append(sim_graph(simnumber,param_file,hparams))

    gals = np.array([graph.x.shape[0] for graph in dataset])
    print("Total of galaxies", gals.sum(0), "Mean of", gals.mean(0),"per simulation, Std of", gals.std(0))

    return dataset
