import sys, os

import itertools
import math
import numpy as np
import time
import torch
from torch.nn.functional import kl_div
from torch.utils.data import DataLoader
from visdom import Visdom

import settings
import utils
from groupmodel import GroupModel

print("Use CUDA: {}".format(settings.GPU))


# Instantiate dataset for training ...
dataset = settings.DATASET(settings.data_path, **settings.DATA_KWARGS)
data_loader = DataLoader(dataset, batch_size=settings.BATCH_SIZE,
                         shuffle=True, num_workers=4, collate_fn=utils.collate_to_packed_for_classification)

# ... and dev set
if settings.args.load_path:
    dev_dataset = settings.DATASET(settings.args.load_path, **settings.DATA_KWARGS)
    dev_data_loader = DataLoader(dev_dataset, batch_size=1,
                         shuffle=True, num_workers=4, collate_fn=utils.collate_to_packed_for_classification)



def eval_accuracy(model, data_loader):
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
        group_probs = dict_group_probs.get(userid, model.np_g_prior)

        t_prediction_matrix, updated_group_probs = model.predict(group_probs, gold_target, feature, lengths)
        prediction_matrix = t_prediction_matrix.data.numpy()

        # t_pred: tensor, [|Y|]
        # ugp: ndarray, [K]

        # collect statistics
        num_total += 1
        predicted = np.argmax(prediction_matrix)

        if predicted == gold_target:
            num_correct += 1

        sum_lik += math.exp(prediction_matrix[gold_target])

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
    for model in models:
        model.cuda()
    group_model.cuda()

# if settings.VISUALIZE:
#     smooth_loss = 7 #approx 2.5^2
#     decay_rate = 0.99
#     smooth_real_dist = np.array([0, 0, 0, 0, 0], dtype=float)
#     smooth_pred_dist = np.array([0, 0, 0, 0, 0], dtype=float)
#
#     counter = 0

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
    group_model.start_epoch()

    # Main train loop
    length = len(dataset)/settings.BATCH_SIZE
    seen_instances_in_epoch = 0

    print("Starting epoch {} with length {}".format(epoch, length))
    for i, (feature, lengths, target, userids) in enumerate(data_loader):
        seen_instances_in_epoch += len(userids)

        if settings.GPU:
            feature = feature.cuda(async=True)
            target = target.cuda(async=True)
            userids = userids.cuda(async=True)


        predictions = [model(feature, lengths) for model in models]
        tgt = torch.autograd.Variable(target)
        losses = [lossfn(out, tgt) for out in predictions]

        loss = group_model.group_loss(losses, userids)



        #
        # out, t_prediction_matrix = model(userids, feature, lengths)
        # prediction_matrix = vnp(t_prediction_matrix)
        #
        # # if settings.GPU:
        # #     prediction_matrix = t_prediction_matrix.cpu().data.numpy()
        # # else:
        # #     prediction_matrix = t_prediction_matrix.data.numpy()
        #
        # # out: [bs, Y]
        # # (t_) prediction_matrix: [bs, Y, K]
        # # target: [bs]
        #
        # # collect likelihoods [bs, K] for reestimation of group assignments
        # bs, Y = out.size()
        # log_likelihoods = np.zeros([bs, num_groups])
        # kl_loss = None
        # prev_t_likelihood = None
        # for i in range(bs):
        #     tg = target[i]
        #     log_likelihoods[i] = prediction_matrix[i, tg, :]
        #
        #     lik = t_prediction_matrix[i, tg, :]
        #     if prev_t_likelihood is not None:
        #         kld = kl_div(lik, torch.exp(prev_t_likelihood))
        #         if kl_loss is not None:
        #             kl_loss += kld
        #         else:
        #             kl_loss = kld
        #     prev_t_likelihood = lik.detach()
        #
        #
        # model.collect_log_likelihoods(log_likelihoods, userids)
        #
        # train_loss = lossfn(out, torch.autograd.Variable(target))


        # loss = train_loss - decay(kl_loss, epoch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log loss
        with open(log_file, 'a') as logfile:
            logfile.write("{},".format(float(loss)))

        # # Visualization update
        # if settings.VISUALIZE:
        #     smooth_loss = smooth_loss * decay_rate + (1-decay_rate) * loss.data.cpu().numpy()
        #     viz.updateTrace(win=loss_plot, X=np.array([counter]), Y=loss.data.cpu().numpy(), name='loss')
        #     viz.updateTrace(win=loss_plot, X=np.array([counter]), Y=smooth_loss, name='smooth loss')
        #     real_star = target[:, 0].data.cpu().numpy().astype(int)
        #     pred_star = out[0, :, 0].data.cpu().numpy().round().clip(1,5).astype(int)
        #     for idx in range(len(real_star)):
        #         smooth_pred_dist[pred_star[idx]-1] += 1
        #         smooth_real_dist[real_star[idx]-1] += 1
        #     smooth_real_dist *= decay_rate
        #     smooth_pred_dist *= decay_rate
        #
        #     viz.bar(win=dist_hist, X=smooth_pred_dist)
        #     viz.bar(win=real_dist_hist, X=smooth_real_dist)
        #
        #     counter += 1

        # Progress update
        # if i % 10 == 0:
        #     sys.stdout.write("\rIter {}/{}, loss: {}".format(i, length, float(loss)))
        #     if COMET_API_KEY:
        #         step = i + epoch*length
        #         experiment.log_metric("loss", float(loss)/seen_instances_in_epoch, step=step)

    group_model.end_epoch()

    if settings.args.load_path:
        eval_acc, mean_eval_likelihood = eval_accuracy(model, dev_data_loader)
        print("val acc: %f" % eval_acc)
        print("val lik: %f" % mean_eval_likelihood)

    if COMET_API_KEY:
        step = (epoch+1)*length

        mean_loss = float(loss) / seen_instances_in_epoch
        ent = group_model.mean_posterior_entropy()

        experiment.log_metric("mean_entropy", ent, step=step)
        experiment.log_metric("loss", mean_loss, step=step)


        # cos = model.cosine_distance()
        # experiment.log_metric("training_loss", float(loss), step=step)
        # experiment.log_metric("kl_loss/100", float(kl_loss/100), step=step)
        # experiment.log_metric("diff_predictions", model.mean_sumcos(), step=step) # mean absolute diff in log-likelihoods

        if settings.args.load_path:
            experiment.log_metric("dev accuracy", eval_acc)
            experiment.log_metric("mean dev likelihood", mean_eval_likelihood)

    # print("mean sc %f" % model.mean_sumcos())

    print("Epoch finished with last loss: {}".format(float(loss)))


    # Visualize distribution and save model checkpoint
    name = "{}_epoch{}.params".format(group_model.get_name(), epoch)
    utils.save_model_params(group_model, name)
    print("Saved model params as: {}".format(name))



