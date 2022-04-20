#----------------------------------------------------------------------
# Routines for training and testing the GNNs
# Author: Pablo Villanueva Domingo
# Last update: 10/11/21
#----------------------------------------------------------------------

from Source.constants import *
from scipy.spatial.transform import Rotation as Rot


# Training step
def train(loader, model, hparams, optimizer, scheduler):
    model.train()

    loss_tot = 0
    for data in loader:  # Iterate in batches over the training dataset.

        # Rotate randomly for data augmentation
        """rotmat = Rot.random().as_matrix()
        data.x[:,:3] = torch.tensor([rotmat.dot(p) for p in data.x[:,:3]], dtype=torch.float32)
        if not hparams.only_positions:
            data.x[:,-3:] = torch.tensor([rotmat.dot(p) for p in data.x[:,-3:]], dtype=torch.float32)"""

        data.to(device)
        optimizer.zero_grad()  # Clear gradients.
        out = model(data)  # Perform a single forward pass.

        # If cosmo parameters are predicted, perform likelihood-free inference to predict also the standard deviation
        if hparams.outmode == "cosmo":
            y_out, err_out = out[:,:hparams.pred_params], out[:,hparams.pred_params:2*hparams.pred_params]     # Take mean and standard deviation of the output

            # Compute loss as sum of two terms for likelihood-free inference
            loss_mse = torch.mean(torch.sum((y_out - data.y)**2., axis=1) , axis=0)
            loss_lfi = torch.mean(torch.sum(((y_out - data.y)**2. - err_out**2.)**2., axis=1) , axis=0)
            loss = torch.log(loss_mse) + torch.log(loss_lfi)

        elif hparams.outmode=="ps":
            loss_mse = torch.mean(torch.sum((out - data.y)**2., axis=1) , axis=0)
            loss = torch.log(loss_mse)

        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        scheduler.step()
        loss_tot += loss.item()

    return loss_tot/len(loader)

# Testing/validation step
def test(loader, model, hparams):
    model.eval()

    outsOm = np.zeros((1))
    truesOm = np.zeros((1))
    yerrorsOm = np.zeros((1))
    outsSig = np.zeros((1))
    truesSig = np.zeros((1))
    yerrorsSig = np.zeros((1))
    outsPS = np.zeros((1,ps_size))
    truesPS = np.zeros((1,ps_size))

    trueparams = np.zeros((1,hparams.pred_params))
    outparams = np.zeros((1,hparams.pred_params))
    outerrparams = np.zeros((1,hparams.pred_params))

    errs = []
    loss_tot = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        with torch.no_grad():

            data.to(device)
            out = model(data)  # Perform a single forward pass.

            # If cosmo parameters are predicted, perform likelihood-free inference to predict also the standard deviation
            if hparams.outmode == "cosmo":
                y_out, err_out = out[:,:hparams.pred_params], out[:,hparams.pred_params:2*hparams.pred_params]     # Take mean and standard deviation of the output
                # Compute loss as sum of two terms for likelihood-free inference
                loss_mse = torch.mean(torch.sum((y_out - data.y)**2., axis=1) , axis=0)
                loss_lfi = torch.mean(torch.sum(((y_out - data.y)**2. - err_out**2.)**2., axis=1) , axis=0)
                loss = torch.log(loss_mse) + torch.log(loss_lfi)

            elif hparams.outmode=="ps":
                loss_mse = torch.mean(torch.sum((out - data.y)**2., axis=1) , axis=0)
                loss = torch.log(loss_mse)
                y_out = out

            err = (y_out - data.y)#/data.y
            errs.append( np.abs(err.detach().cpu().numpy()).mean() )
            loss_tot += loss.item()

            if hparams.outmode == "cosmo":
                # Append true values and predictions
                trueparams = np.append(trueparams, data.y.detach().cpu().numpy(), 0)
                outparams = np.append(outparams, y_out.detach().cpu().numpy(), 0)
                outerrparams  = np.append(outerrparams, err_out.detach().cpu().numpy(), 0)
                """
                truesOm = np.append(truesOm, data.y[:,0].detach().cpu().numpy(), 0)
                outsOm = np.append(outsOm, y_out[:,0].detach().cpu().numpy(), 0)
                yerrorsOm = np.append(yerrorsOm, err_out[:,0].detach().cpu().numpy(), 0)
                if hparams.pred_params==2:
                    truesSig = np.append(truesSig, data.y[:,1].detach().cpu().numpy(), 0)
                    outsSig = np.append(outsSig, y_out[:,1].detach().cpu().numpy(), 0)
                    yerrorsSig = np.append(yerrorsSig, err_out[:,1].detach().cpu().numpy(), 0)
                """

            elif hparams.outmode=="ps":
                # Append true values and predictions
                outsPS = np.append(outsPS, out.detach().cpu().numpy(), 0)
                truesPS = np.append(truesPS, data.y.detach().cpu().numpy(), 0)


    # Save true values and predictions
    if hparams.outmode == "cosmo":
        np.save("Outputs/trues_"+hparams.name_model()+".npy",trueparams)
        np.save("Outputs/outputs_"+hparams.name_model()+".npy",outparams)
        np.save("Outputs/errors_"+hparams.name_model()+".npy",outerrparams)
        """
        np.save("Outputs/outputsOm_"+hparams.name_model()+".npy",outsOm)
        np.save("Outputs/truesOm_"+hparams.name_model()+".npy",truesOm)
        np.save("Outputs/errorsOm_"+hparams.name_model()+".npy",yerrorsOm)
        if hparams.pred_params==2:
            np.save("Outputs/outputsSig_"+hparams.name_model()+".npy",outsSig)
            np.save("Outputs/truesSig_"+hparams.name_model()+".npy",truesSig)
            np.save("Outputs/errorsSig_"+hparams.name_model()+".npy",yerrorsSig)
        """

    elif hparams.outmode=="ps":
        np.save("Outputs/outputsPS_"+hparams.name_model()+".npy",outsPS)
        np.save("Outputs/truesPS_"+hparams.name_model()+".npy",truesPS)



    return loss_tot/len(loader), np.array(errs).mean(axis=0)

# Training procedure
def training_routine(model, train_loader, test_loader, hparams, verbose=True):

    #use_model, learning_rate, weight_decay, n_layers, k_nn, n_epochs, training, simsuite, simset, n_sims = hparams

    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.learning_rate, weight_decay=hparams.weight_decay)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=hparams.learning_rate, max_lr=1.e-3, cycle_momentum=False)

    train_losses, valid_losses = [], []
    valid_loss_min, err_min = 1000., 1000.

    # Training loop
    for epoch in range(1, hparams.n_epochs+1):
        train_loss = train(train_loader, model, hparams, optimizer, scheduler)
        test_loss, err = test(test_loader, model, hparams)
        train_losses.append(train_loss); valid_losses.append(test_loss)

        # Save model if it has improved
        if test_loss <= valid_loss_min:
            if verbose: print("Validation loss decreased ({:.2e} --> {:.2e}).  Saving model ...".format(valid_loss_min,test_loss))
            torch.save(model.state_dict(), "Models/"+hparams.name_model())
            valid_loss_min = test_loss
            err_min = err

        if verbose: print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.2e}, Validation Loss: {test_loss:.2e}, Error: {err:.2e}')

    return train_losses, valid_losses
