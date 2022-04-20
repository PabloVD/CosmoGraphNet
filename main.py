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


# Main routine to train the neural net
# If testsuite==True, it takes a model already pretrained in the other suite and tests it in the selected one
def main(hparams, verbose = True, testsuite = False):

    hparams.n_sims = 30
    hparams.n_epochs = 5

    # Load data and create dataset
    dataset = create_dataset(hparams)
    node_features = dataset[0].x.shape[1]

    # Split dataset among training, validation and testing datasets
    train_loader, valid_loader, test_loader = split_datasets(dataset)

    # Size of the output of the GNN
    if hparams.outmode=="cosmo":
        dim_out=2*hparams.pred_params
    elif hparams.outmode=="ps":
        dim_out=ps_size

    # Initialize model
    model = GNN(node_features=node_features,
                n_layers=hparams.n_layers,
                hidden_channels=hparams.hidden_channels,
                linkradius=hparams.r_link,
                dim_out=dim_out,
                only_positions=hparams.only_positions)
    model.to(device)
    if verbose: print("Model: " + hparams.name_model()+"\n")

    # Print the memory (in GB) being used now:
    process = psutil.Process()
    print("Memory being used (GB):",process.memory_info().rss/1.e9)

    # Train the net
    if hparams.training:
        if verbose: print("Training!\n")
        train_losses, valid_losses = training_routine(model, train_loader, valid_loader, hparams, verbose)

    # Test the net
    if verbose: print("\nTesting!\n")

    # If test in other suite, change the suite for loading the model
    if testsuite==True:
        hparams.simsuite = hparams.flip_suite()   # change for loading the model

    # Load the trained model
    state_dict = torch.load("Models/"+hparams.name_model(), map_location=device)
    model.load_state_dict(state_dict)

    if testsuite==True: hparams.simsuite = hparams.flip_suite()   # change after loading the model

    # Test the model
    test_loss, rel_err = test(test_loader, model, hparams)
    if verbose: print("Test Loss: {:.2e}, Relative error: {:.2e}".format(test_loss, rel_err))

    # Plot loss trends
    if hparams.training:
        plot_losses(train_losses, valid_losses, test_loss, rel_err, hparams)

    # Plot true vs predicted cosmo parameters
    if hparams.outmode=="cosmo":
        plot_out_true_scatter(hparams, "Om", testsuite)
        if hparams.pred_params==2:
            plot_out_true_scatter(hparams, "Sig", testsuite)

    # Plot power spectrum and relative error
    elif hparams.outmode=="ps":

        plot_relerr(hparams)

        plot_ps(hparams, testsuite)

    return test_loss


#--- MAIN ---#

if __name__ == "__main__":

    time_ini = time.time()

    for path in ["Plots", "Models", "Outputs"]:
        if not os.path.exists(path):
            os.mkdir(path)

    # Load default parameters
    from hyperparameters import hparams

    main(hparams)

    print("Finished. Time elapsed:",datetime.timedelta(seconds=time.time()-time_ini))
