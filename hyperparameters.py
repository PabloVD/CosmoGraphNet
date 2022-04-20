

class hyperparameters():
    def __init__(self, outmode, only_positions, learning_rate, weight_decay, n_layers, hidden_channels, r_link, n_epochs, simsuite, simset="LH", n_sims=1000, training=True, pred_params=2):

        # Choose the output to be predicted, either the cosmological parameters ("cosmo") or the power spectrum ("ps")
        self.outmode = outmode
        # 1 for using only positions as features, 0 for using additional galactic features
        # 1 only for outmode = "cosmo"
        self.only_positions = only_positions
        if self.outmode == "ps":
            self.only_positions = 1
        # Learning rate
        self.learning_rate = learning_rate
        # Weight decay
        self.weight_decay = weight_decay
        # Number of graph layers
        self.n_layers = n_layers
        # Hidden channels
        self.hidden_channels = hidden_channels
        # Linking radius
        self.r_link = r_link
        # Number of epochs
        self.n_epochs = n_epochs
        # Simulation suite, choose between "IllustrisTNG" and "SIMBA"
        self.simsuite = simsuite
        # Simulation set, choose between "CV" and "LH"
        self.simset = simset
        # Number of simulations considered, maximum 27 for CV and 1000 for LH
        self.n_sims = n_sims
        # If training, set to True, otherwise loads a pretrained model and tests it
        self.training = training
        # Number of cosmo/astro params to be predicted, starting from Omega_m, sigma_8, etc.
        # Only for outmode = "cosmo"
        self.pred_params = pred_params
        # Snapshot of the simulation, indicating redshift 4: z=3, 10: z=2, 14: z=1.5, 18: z=1, 24: z=0.5, 33: z=0
        self.snap = "33"

    # Name of the model and hyperparameters
    def name_model(self):
        return self.outmode+"_"+self.simsuite+"_"+self.simset+"_onlypos_"+str(self.only_positions)+"_lr_{:.2e}_weightdecay_{:.2e}_layers_{:d}_rlink_{:.2e}_channels_{:d}_epochs_{:d}".format(self.learning_rate, self.weight_decay, self.n_layers, self.r_link, self.hidden_channels, self.n_epochs)

    # Return the other CAMELS simulation suite
    def flip_suite(self):
        if self.simsuite=="IllustrisTNG":
            new_simsuite = "SIMBA"
        elif self.simsuite=="SIMBA":
            new_simsuite = "IllustrisTNG"
        return new_simsuite


hparams = hyperparameters(outmode = "cosmo",                        # Choose the output to be predicted, either the cosmological parameters ("cosmo") or the power spectrum ("ps")
                          only_positions = 0,                       # 1 for using only positions as features, 0 for using additional galactic features
                          learning_rate = 0.0005061466694296257,    # Learning rate
                          weight_decay = 1.e-07,                    # Weight decay
                          n_layers = 1,                             # Number of hidden graph layers
                          r_link = 0.06952349951836213,             # Linking radius
                          hidden_channels = 128,                    # Hidden channels
                          n_epochs = 300,                           # Number of epochs
                          simsuite = "IllustrisTNG",                # Simulation suite, choose between "IllustrisTNG" and "SIMBA"
                          pred_params = 1                           # Number of cosmo/astro params to be predicted, starting from Omega_m, sigma_8, etc. (Only for outmode = "cosmo")
                          )
