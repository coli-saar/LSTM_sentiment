import sys, os

import numpy as np
import torch
from torch.utils.data import DataLoader
from visdom import Visdom

import settings
import utils


print("Use CUDA: {}".format(settings.GPU))


COMET_API_KEY = os.environ.get("COMET_API_KEY")

if COMET_API_KEY:
    # Record experiment in Comet
    from comet_ml import Experiment

    experiment  = Experiment(api_key=COMET_API_KEY, project_name="Yelp Sentiment")

    hyper_params = dict(settings.MODEL)
    hyper_params.pop("model", None)
    hyper_params["training_data"] = settings.data_path
    hyper_params["epochs"] = settings.EPOCHS
    hyper_params["learning_rate"] = settings.LEARNING_RATE
    hyper_params["batchsize"] = settings.BATCH_SIZE
    hyper_params["glove_path"] = settings.DATA_KWARGS["glove_path"]
    hyper_params["GPU"] = settings.GPU
    hyper_params["hostname"] = os.environ.get("HOSTNAME") or "(undefined)"

    experiment.log_multiple_params(hyper_params)

# Instansiate dataset
dataset = settings.DATASET(settings.data_path, **settings.DATA_KWARGS)
data_loader = DataLoader(dataset, batch_size=settings.BATCH_SIZE,
                         shuffle=True, num_workers=4, collate_fn=utils.collate_to_packed)

# print(dataset.userdict.num_users())
# print(dataset.userdict.id_to_user)
# sys.exit(0)

num_users = dataset.userdict.num_users()
num_groups = 2

print("Training model with %d users in %d groups." % (num_users, num_groups))

# Define model and optimizer
# model = utils.generate_model_from_settings()
model = utils.generate_group_model_from_settings(num_users, num_groups)

parameters = filter(lambda p:p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(parameters, lr=settings.LEARNING_RATE)

# Log file is namespaced with the current model
log_file = "logs/{}_{}.csv".format(model.get_name(), settings.data_path.split("/")[-1].split(".json")[0])

if settings.VISUALIZE:
    # Visualization thorugh visdom
    viz = Visdom()
    loss_plot = viz.line(X=np.array([0]), Y=np.array([0]), opts=dict(showlegend=True, title="Loss"))
    hist_opts = settings.HIST_OPTS
    hist_opts["title"] = "Predicted star distribution"
    dist_hist = viz.bar(X=np.array([0, 0, 0]), opts=dict(title="Predicted stars"))
    real_dist_hist = viz.bar(X=np.array([0, 0, 0]))

# Move stuff to GPU
if settings.GPU:
    data_loader.pin_memory = True
    model.cuda()

if settings.VISUALIZE:
    smooth_loss = 7 #approx 2.5^2
    decay_rate = 0.99
    smooth_real_dist = np.array([0, 0, 0, 0, 0], dtype=float)
    smooth_pred_dist = np.array([0, 0, 0, 0, 0], dtype=float)

    counter = 0

for epoch in range(settings.EPOCHS):
    model.start_epoch()

    # Main train loop
    length = len(dataset)/settings.BATCH_SIZE
    print("Starting epoch {} with length {}".format(epoch, length))
    for i, (feature, lengths, target, userids) in enumerate(data_loader):
        # print("ft: %s" % str(feature))
        # print("le: %s" % str(lengths))
        # print("tg: %s" % str(target))

        if settings.GPU:
            feature = feature.cuda(async=True)
            target = target.cuda(async=True)

        out = model(feature, lengths, userids)

        # collect likelihoods for reestimation of group assignments
        # self.collect_likelihoods()

        # Loss computation and weight update step
        loss = torch.mean((out[0, :, 0] - target[:, 0])**2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log loss
        with open(log_file, 'a') as logfile:
            logfile.write("{},".format(float(loss)))

        # Visualization update
        if settings.VISUALIZE:
            smooth_loss = smooth_loss * decay_rate + (1-decay_rate) * loss.data.cpu().numpy()
            viz.updateTrace(win=loss_plot, X=np.array([counter]), Y=loss.data.cpu().numpy(), name='loss')
            viz.updateTrace(win=loss_plot, X=np.array([counter]), Y=smooth_loss, name='smooth loss')
            real_star = target[:, 0].data.cpu().numpy().astype(int)
            pred_star = out[0, :, 0].data.cpu().numpy().round().clip(1,5).astype(int)
            for idx in range(len(real_star)):
                smooth_pred_dist[pred_star[idx]-1] += 1
                smooth_real_dist[real_star[idx]-1] += 1
            smooth_real_dist *= decay_rate
            smooth_pred_dist *= decay_rate

            viz.bar(win=dist_hist, X=smooth_pred_dist)
            viz.bar(win=real_dist_hist, X=smooth_real_dist)

            counter += 1

        # Progress update
        if i % 10 == 0:
            sys.stdout.write("\rIter {}/{}, loss: {}".format(i, length, float(loss)))
            if COMET_API_KEY:
                step = i + epoch*length
                experiment.log_metric("training_loss", float(loss), step=step)

    model.end_epoch()

    print("Epoch finished with last loss: {}".format(float(loss)))


    # Visualize distribution and save model checkpoint
    name = "{}_epoch{}.params".format(model.get_name(), epoch)
    utils.save_model_params(model, name)
    print("Saved model params as: {}".format(name))



