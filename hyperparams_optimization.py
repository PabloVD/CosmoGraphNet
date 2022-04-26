#----------------------------------------------------------------------
# Script for optimizing the hyperparameters of the network using optuna
# Author: Pablo Villanueva Domingo
# Last update: 4/22
#----------------------------------------------------------------------

import optuna
from main import *
from optuna.visualization import plot_optimization_history, plot_contour, plot_param_importances    # it needs plotly and kaleido
from hyperparameters import hparams

# Simulation type
simsuite = "IllustrisTNG"
simset = "LH"
n_sims = 1000
# Number of epochs
n_epochs = 300

# Objective function to minimize
def objective(trial):

    # Hyperparameters to optimize
    learning_rate = trial.suggest_float("learning_rate", 1e-8, 1e-4, log=True)
    #weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-6, log=True)
    weight_decay = 1.e-7
    n_layers = trial.suggest_int("n_layers", 1, 5)
    hidden_channels = trial.suggest_categorical("hidden_channels", [64, 128, 256])
    r_link = trial.suggest_float("r_link", 5.e-3, 5.e-2, log=True)

    # Some verbose
    print('\nTrial number: {}'.format(trial.number))
    print('learning_rate: {}'.format(learning_rate))
    #print('weight_decay: {}'.format(weight_decay))
    print('n_layers:  {}'.format(n_layers))
    print('hidden_channels:  {}'.format(hidden_channels))
    print('r_link:  {}'.format(r_link))

    # Hyperparameters to be optimized
    hparams.learning_rate = learning_rate
    hparams.weight_decay = weight_decay
    hparams.n_layers = n_layers
    hparams.hidden_channels = hidden_channels
    hparams.r_link = r_link

    # Default params
    hparams.n_epochs = n_epochs
    hparams.simsuite = simsuite
    hparams.simset = simset
    hparams.n_sims = n_sims

    # Run main routine
    min_test_loss = main(hparams, verbose = False)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return min_test_loss


#--- MAIN ---#

if __name__ == "__main__":

    time_ini = time.time()

    for path in ["Plots", "Models", "Outputs"]:
        if not os.path.exists(path):
            os.mkdir(path)

    # Optuna parameters
    storage = "sqlite:///"+os.getcwd()+"/optuna_"+simsuite+"_"+simset
    study_name = "gnn"
    n_trials   = 30

    # Define sampler and start optimization
    sampler = optuna.samplers.TPESampler(n_startup_trials=10)
    study = optuna.create_study(study_name=study_name, sampler=sampler, storage=storage, load_if_exists=True)
    study.optimize(objective, n_trials, gc_after_trial=True)

    # Print info for best trial
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    hparams.learning_rate = trial.params["learning_rate"]
    hparams.n_layers = trial.params["n_layers"]
    hparams.hidden_channels = trial.params["hidden_channels"]
    hparams.r_link = trial.params["r_link"]

    # Save best model and plots
    if not os.path.exists("Best"):
        os.mkdir("Best")
    # Change nominal suite to read correct files (actually in ps mode both suites are employed)
    if hparams.outmode=="ps":
        hparams.simsuite = hparams.flip_suite()
    files = []
    files.append( "Plots/out_true_Om_"+hparams.name_model()+".png" )
    files.append( "Plots/out_true_Sig_"+hparams.name_model()+".png" )
    files.append( "Plots/loss_"+hparams.name_model()+".png" )
    files.append( "Plots/ps_"+hparams.name_model()+".png" )
    files.append( "Plots/rel_err_"+hparams.name_model()+".png" )
    files.append( "Models/"+hparams.name_model() )
    for file in files:
        if os.path.exists(file):
            os.system("cp "+file+" Best/.")

    # Visualization of optimization results
    fig = plot_optimization_history(study)
    fig.write_image("Plots/optuna_optimization_history_"+simsuite+".png")

    fig = plot_contour(study)#, params=["learning_rate", "weight_decay", "r_link"])#, "use_model"])
    fig.write_image("Plots/optuna_contour_"+simsuite+".png")

    fig = plot_param_importances(study)
    fig.write_image("Plots/plot_param_importances_"+simsuite+".png")

    print("Finished. Time elapsed:",datetime.timedelta(seconds=time.time()-time_ini))
