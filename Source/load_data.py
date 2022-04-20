import h5py
from torch_geometric.data import Data, DataLoader
from Source.constants import *
from Source.plotting import *
import scipy.spatial as SS

Nstar_th = 20   # Minimum number of stellar particles required to consider a galaxy

def normalize_params(params):

    minimum = np.array([0.1, 0.6, 0.25, 0.25, 0.5, 0.5])
    maximum = np.array([0.5, 1.0, 4.00, 4.00, 2.0, 2.0])
    params = (params - minimum)/(maximum - minimum)
    return params

def normalize_ps(ps):
    mean, std = ps.mean(axis=0), ps.std(axis=0)
    normps = (ps - mean)/std
    return normps

def get_edges(pos, r_link, use_loops):

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

    # Edge attributes
    row, col = edge_index
    diff = pos[row]-pos[col]

    # Correct boundaries in distances
    for i, pos_i in enumerate(diff):
        #outbound=False
        for j, coord in enumerate(pos_i):
            if coord > r_link:
                #outbound=True
                diff[i,j] -= 1.  # Boxsize normalize to 1
            elif -coord > r_link:
                #outbound=True
                diff[i,j] += 1.  # Boxsize normalize to 1
        #if outbound: numbounds+=1

    dist = np.linalg.norm(diff, axis=1)
    centroid = np.mean(pos,axis=0)
    unitrow = (pos[row]-centroid)/np.linalg.norm((pos[row]-centroid), axis=1).reshape(-1,1)
    unitcol = (pos[col]-centroid)/np.linalg.norm((pos[col]-centroid), axis=1).reshape(-1,1)
    unitdiff = diff/dist.reshape(-1,1)
    cos1 = np.array([np.dot(unitrow[i,:].T,unitcol[i,:]) for i in range(num_pairs)])
    cos2 = np.array([np.dot(unitrow[i,:].T,unitdiff[i,:]) for i in range(num_pairs)])

    #print(edge_index.shape, cos1.shape, cos2.shape, dist.shape)
    dist /= r_link
    edge_attr = np.concatenate([dist.reshape(-1,1), cos1.reshape(-1,1), cos2.reshape(-1,1)], axis=1)

    #print(pos.shape, edge_index.shape, edge_attr.shape)

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

    #print(pos.shape, edge_index.shape, edge_attr.shape)



    #print(edge_index.shape, edge_attr.shape)


    """
    diff = (pos[row]-pos[col])/r_link

    #print(diff.shape, edge_index.shape, pos.shape)
    #numbounds = 0

    # Correct boundaries in distances
    for i, pos_i in enumerate(diff):
        #outbound=False
        for j, coord in enumerate(pos_i):
            if coord > 1.:
                #outbound=True
                diff[i,j] -= 1./r_link  # Boxsize normalize to 1
            elif -coord > 1.:
                #outbound=True
                diff[i,j] += 1./r_link  # Boxsize normalize to 1
        #if outbound: numbounds+=1

    edge_attr = np.concatenate([diff, np.linalg.norm(diff, axis=1, keepdims=True)], axis=1)
    #print(edge_attr[:,3].min(), edge_attr[:,3].max())
    #print(diff.shape[0], numbounds)
    """

    return edge_index, edge_attr

######################################################################################
# This routine reads the galaxies from a simulation and
# root ------> folder containing all simulations with their galaxy catalogues
# sim -------> 'IllustrisTNG' or 'SIMBA'
# suite -----> 'LH' or 'CV'
# number ----> number of the simulation
# snapnum ---> snapshot number (choose depending of the desired redshift)
# BoxSize ---> size of the simulation box in Mpc/h
# Nstar_th -----> galaxies need to contain at least Nstar_th stars
# k ---------> number of neighbors
# param_file -> file with the value of the cosmological + astrophysical parameters
def sim_graph(simnumber,param_file,hparams):

    simsuite,simset,r_link,only_positions,outmode,pred_params = hparams.simsuite,hparams.simset,hparams.r_link,hparams.only_positions,hparams.outmode,hparams.pred_params

    # get the name of the galaxy catalogue
    simpath = simpathroot + simsuite + "/"+simset+"_"
    catalogue = simpath + str(simnumber)+"/fof_subhalo_tab_0"+hparams.snap+".hdf5"

    # read the catalogue
    f     = h5py.File(catalogue, 'r')
    pos   = f['/Subhalo/SubhaloPos'][:]/boxsize
    Mstar = f['/Subhalo/SubhaloMassType'][:,4] #Msun/h
    SubhaloVel = f["Subhalo/SubhaloVel"][:]
    Rstar = f["Subhalo/SubhaloHalfmassRadType"][:,4]
    Metal = f["Subhalo/SubhaloStarMetallicity"][:]
    Vmax = f["Subhalo/SubhaloVmax"][:]
    Nstar = f['/Subhalo/SubhaloLenType'][:,4]       #number of stars
    f.close()

    # some simulations are slightly outside the box
    pos[np.where(pos<0.0)]+=1.0
    pos[np.where(pos>1.0)]-=1.0

    # select only galaxies with more than 10 star particles
    indexes = np.where(Nstar>Nstar_th)[0]
    pos     = pos[indexes]
    Mstar   = Mstar[indexes]
    SubhaloVel = SubhaloVel[indexes]
    Rstar   = Rstar[indexes]
    Metal   = Metal[indexes]
    Vmax   = Vmax[indexes]

    # Get the output to be predicted by the GNN, either the cosmo parameters or the power spectrum
    if outmode=="cosmo":
        # read the value of the cosmological & astrophysical parameters
        paramsfile = np.loadtxt(param_file, dtype=str)
        params = np.array(paramsfile[simnumber,1:-1],dtype=np.float32)
        params = normalize_params(params)
        params = params[:pred_params]
        y = np.reshape(params, (1,params.shape[0]))

    elif outmode=="ps":

        ps = np.load(param_file)
        ps = ps[simnumber]
        ps = np.log10(ps)
        #ps = normalize_ps(ps)
        y = np.reshape(ps, (1,ps_size))


    """
    # compute the number of pairs
    nodes = pos.shape[0]
    u      = np.zeros((1,2),       dtype=np.float32)
    u[0,0] = np.log10(np.sum(Mstar))
    u[0,1] = np.log10(nodes)
    """
    u = np.log10(pos.shape[0]).reshape(1,1)

    Mstar = np.log10(1.+ Mstar)
    #SubhaloVel = np.log10(1.+SubhaloVel)
    SubhaloVel/=100.
    Rstar = np.log10(1.+ Rstar)
    Metal = np.log10(1.+ Metal)
    Vmax = np.log10(1. + Vmax)
    tab = np.column_stack((Mstar, Rstar, Metal, Vmax))
    #tab = Vmax.reshape(-1,1)

    if only_positions:
        #u   = np.zeros((1,2), dtype=np.float32) # not used
        tab = np.zeros_like(pos[:,:1])   # not really used
        use_loops = False
    else:
        use_loops = True#"""

    #use_loops = False

    x = torch.tensor(tab, dtype=torch.float32)

    #use_loops = False
    edge_index, edge_attr = get_edges(pos, r_link, use_loops)
    #edge_index = get_edges(pos, r_link)
    #edge_index = None

    # get the graph
    graph = Data(x=x,
                 y=torch.tensor(y, dtype=torch.float32),
                 u=torch.tensor(u, dtype=torch.float32),
                 edge_index=torch.tensor(edge_index, dtype=torch.long),
                 edge_attr=torch.tensor(edge_attr, dtype=torch.float32))

    return graph
######################################################################################
"""
######################################################################################
# This routine creates the dataset for the considered mode
# mode -------------> 'train', 'valid', 'test' or 'all'
# seed -------------> random seed to split simulations among train/valid/test
# sims -------------> total number of simulations
# root -------------> folder containing all simulations with their galaxy catalogues
# sim --------------> 'IllustrisTNG' or 'SIMBA'
# suite ------------> 'LH' or 'CV'
# number -----------> number of the simulation
# snapnum ----------> snapshot number (choose depending of the desired redshift)
# BoxSize ----------> size of the simulation box in Mpc/h
# Nstar_th --> galaxies need to contain at least Nstar_th stars
# k ----------------> number of neighbors
# param_file -------> file with the value of the cosmo & astro parameters
# batch_size -------> batch size
# num_workers ------> number of workers to load the data
# shuffle ----------> whether randomly shuffle the data in the data loader
def create_dataset(mode, seed, sims, root, sim, suite, snapnum, BoxSize,
                   Nstar_th, k, param_file, batch_size, num_workers=1,
                   shuffle=True):



    # get the offset and size of the considered mode
    if   mode=='train':  offset, size = int(0.0*sims), int(0.8*sims)
    elif mode=='valid':  offset, size = int(0.8*sims), int(0.1*sims)
    elif mode=='test':   offset, size = int(0.9*sims), int(0.1*sims)
    elif mode=='all':    offset, size = int(0.0*sims), int(1.0*sims)
    else:                raise Exception('wrong mode!')

    # randomly shuffle the simulations. Instead of 0 1 2 3...999 have a
    # random permutation. E.g. 5 9 0 29...342
    np.random.seed(seed)
    numbers = np.arange(sims) #shuffle sims not maps
    np.random.shuffle(numbers)
    numbers = numbers[offset:offset+size] #select indexes of mode

    # get the dataset
    dataset = []
    for i in numbers:
        dataset.append(sim_graph(root,sim,suite,i,snapnum,BoxSize,
                                 Nstar_th,k,param_file))

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers)
"""
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
# simsuite: simulation suite, either "IllustrisTNG" or "SIMBA"
# simset: set of simulations:
#   CV: Use simulations with fiducial cosmological and astrophysical parameters, but different random seeds (27 simulations total)
#   LH: Use simulations over latin-hypercube, varying over cosmological and astrophysical parameters, and different random seeds (1000 simulations total)
# n_sims: number of simulations, maximum 27 for CV and 1000 for LH
def create_dataset(hparams):

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
        for snap in [24,18,14,10]:
        #for snap in [18,10]:

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
