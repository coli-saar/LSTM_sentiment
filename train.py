import sys, os

import itertools
import math
import numpy as np
import time
import torch
from torch.autograd import Variable
from torch.nn.functional import kl_div
from torch.utils.data import DataLoader
from visdom import Visdom

import settings
import utils
from groupmodel import GroupModel

print("Use CUDA: {}".format(settings.GPU))
print("Profiler: {}".format(settings.WITH_PROFILER))
print("Validate: {}".format(settings.VALIDATION_DATA_PATH))
print("")

# Instantiate dataset for training ...
dataset = settings.DATASET(settings.data_path, **settings.DATA_KWARGS)
data_loader = DataLoader(dataset, batch_size=settings.BATCH_SIZE,
                         shuffle=True, num_workers=4, collate_fn=utils.collate_to_packed_for_classification)

# ... and dev set
if settings.VALIDATION_DATA_PATH:
    dev_dataset = settings.DATASET(settings.VALIDATION_DATA_PATH, **settings.DATA_KWARGS)
    dev_data_loader = DataLoader(dev_dataset, batch_size=1,
                         shuffle=True, num_workers=4, collate_fn=utils.collate_to_packed_for_classification)



def eval_accuracy(group_model, data_loader):
    models = group_model.models

    num_correct = 0
    num_total = 0
    sum_lik = 0

    predict_time = 0

    ts = time.time()

    dict_group_probs = {} # userid -> group_probs

    print("eval")
    for i, (feature, lengths, target, userids) in enumerate(data_loader):
        # userids: LongTensor shape [1]

        userid = userids[0]     # int
        gold_target = target[0] # int
        group_probs = dict_group_probs.get(userid, group_model.np_g_prior)

        predictions = [model(feature, lengths) for model in models]
        prediction_matrix, updated_group_probs = group_model.predict_categorical(group_probs, gold_target, predictions)

        # t_pred: tensor, [|Y|]
        # ugp: ndarray, [K]

        # collect statistics
        num_total += 1
        predicted = np.argmax(prediction_matrix)

        if predicted == gold_target:
            num_correct += 1

        sum_lik += prediction_matrix[gold_target]

        # update group probs for user
        dict_group_probs[userid] = updated_group_probs

    te = time.time()
    print("eval took %f sec" % (te-ts))
    print("acc %d/%d" % (num_correct, num_total))

    return float(num_correct)/num_total, sum_lik/num_total






# Initialize experiment
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

num_users = dataset.userdict.num_users()
num_groups = settings.MODEL["groups"]

print("Training model with %d users in %d groups." % (num_users, num_groups))

# Define model and optimizer
# model = utils.generate_model_from_settings()
# model = utils.generate_group_model_from_settings(num_users, num_groups)
models = [utils.generate_model_from_settings() for g in range(num_groups)]
group_model = GroupModel(num_users, num_groups, models)

parameters = filter(lambda p:p.requires_grad, group_model.parameters())
optimizer = torch.optim.Adam(parameters, lr=settings.LEARNING_RATE)

# Log file is namespaced with the current model
log_file = "logs/{}_{}.csv".format(group_model.get_name(), settings.data_path.split("/")[-1].split(".json")[0])


# Move stuff to GPU
if settings.GPU:
    data_loader.pin_memory = True
    for model in models:
        model.cuda()
    group_model.cuda()



lossfn = torch.nn.NLLLoss(reduce=False)

def KL(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def decay(loss, epoch):
    # return loss/np.power(epoch+1, 0.2)/100
    return 0

def vnp(variable):
    if settings.GPU:
        return variable.cpu().data.numpy()
    else:
        return variable.data.numpy()

def tnp(tensor):
    if settings.GPU:
        return tensor.cpu().numpy()
    else:
        return tensor.numpy()


for epoch in range(settings.EPOCHS):
    with torch.autograd.profiler.profile(enabled=settings.WITH_PROFILER) as prof:
        group_model.start_epoch()

        # Main train loop
        length = len(dataset)/settings.BATCH_SIZE
        kl_diff = 0
        seen_instances_in_epoch = 0

        print("Starting epoch {} with length {}".format(epoch, length))
        for i, (feature, lengths, target, userids) in enumerate(data_loader):
            seen_instances_in_epoch += len(userids)

            if settings.GPU:
                feature = feature.cuda(async=True)
                target = target.cuda(async=True)
                userids = userids.cuda(async=True)

            # apply the models for the individual groups forwards
            predictions = [model(feature, lengths) for model in models]
            tgt = torch.autograd.Variable(target)
            losses = [lossfn(out, tgt) for out in predictions]

            # combine their losses into a loss for the whole group model
            loss = group_model.group_loss(losses, userids)

            # for evaluation purposes, compute KL divergence between the
            # predictions of the individual models
            if num_groups > 1:
                # predictions[i] are FloatTensors (bs, Y)
                lik = predictions[0]                             # tensor
                prev_t_likelihood = predictions[1].detach()
                kl_diff += float(kl_div(lik, torch.exp(prev_t_likelihood)))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log loss
            with open(log_file, 'a') as logfile:
                logfile.write("{},".format(float(loss)))

            # Progress update
            # if i % 10 == 0:
            #     sys.stdout.write("\rIter {}/{}, loss: {}".format(i, length, float(loss)))
            #     if COMET_API_KEY:
            #         step = i + epoch*length
            #         experiment.log_metric("loss", float(loss)/seen_instances_in_epoch, step=step)

        group_model.end_epoch()

    if settings.WITH_PROFILER:
        with open("profiler_%d.txt" % epoch, "w") as pf:
            print(prof, file=pf)



    if settings.VALIDATION_DATA_PATH:
        print("validate...")
        eval_acc, mean_eval_likelihood = eval_accuracy(group_model, dev_data_loader)
        print("val acc: %f" % eval_acc)
        print("val lik: %f" % mean_eval_likelihood)

    if COMET_API_KEY:
        step = (epoch+1)*length

        mean_loss = float(loss) / seen_instances_in_epoch
        mean_kl_diff = kl_diff
        ent = group_model.mean_posterior_entropy()
        pri_ent = group_model.prior_entropy() # actually of one episode before

        experiment.log_metric("mean_entropy", ent, step=step)
        experiment.log_metric("loss", mean_loss, step=step)
        experiment.log_metric("kl_diff", mean_kl_diff, step=step)
        experiment.log_metric("prior_entropy", pri_ent, step=step-length)

        if settings.VALIDATION_DATA_PATH:
            experiment.log_metric("dev accuracy", eval_acc, step=step)
            experiment.log_metric("mean dev likelihood", mean_eval_likelihood, step=step)

    print("Epoch finished with last loss: {}".format(float(loss)))


    # Visualize distribution and save model checkpoint
    name = "{}_epoch{}.params".format(group_model.get_name(), epoch)
    utils.save_model_params(group_model, name)
    print("Saved model params as: {}".format(name))



