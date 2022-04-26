#------------------------------------------------
# Test a model already trained
# Author: Pablo Villanueva Domingo
# Last update: 3/21
#------------------------------------------------

from main import *
from hyperparameters import hparams

#--- MAIN ---#

time_ini = time.time()

for path in ["Plots", "Models", "Outputs"]:
        if not os.path.exists(path):
            os.mkdir(path)

# Test a pretrained model
hparams.training = False

main(hparams)

# Test the pretrained model in the other CAMELS suite
hparams.simsuite = hparams.flip_suite()

main(hparams, testsuite=True)

print("Finished. Time elapsed:",datetime.timedelta(seconds=time.time()-time_ini))
