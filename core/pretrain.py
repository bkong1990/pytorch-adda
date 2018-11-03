"""Pre-train encoder and classifier for source dataset."""

import torch.nn as nn
import torch.optim as optim

import params
from utils import make_variable, save_model
import losses


def train_src(encoder, decoder, data_loader, logger):
    """Train classifier for source domain."""
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    encoder.train()
    decoder.train()

    # setup criterion and optimizer
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=params.c_learning_rate,
        betas=(params.beta1, params.beta2))
    criterion = losses.cross_entropy2d

    ####################
    # 2. train network #
    ####################

    for epoch in range(params.num_epochs_pre):
        for step, (images, labels, _) in enumerate(data_loader):
            # make images and labels variable
            images = make_variable(images)
            labels = make_variable(labels.squeeze_())

            # zero gradients for optimizer
            optimizer.zero_grad()

            # compute loss for critic
            preds = decoder(encoder(images))
            loss = criterion(preds, labels)

            # optimize source classifier
            loss.backward()
            optimizer.step()

            # print step info
            if ((step + 1) % params.log_step_pre == 0):
                logger.info("Epoch [{}/{}] Step [{}/{}]: loss={}"
                      .format(epoch + 1,
                              params.num_epochs_pre,
                              step + 1,
                              len(data_loader),
                              loss.data[0]))

        # eval model on test set
        if ((epoch + 1) % params.eval_step_pre == 0):
            eval_src(encoder, decoder, data_loader, logger)

        # save model parameters
        if ((epoch + 1) % params.save_step_pre == 0):
            save_model(encoder, "ADDA-source-encoder-{}.pt".format(epoch + 1), logger)
            save_model(
                decoder, "ADDA-source-classifier-{}.pt".format(epoch + 1), logger)

    # # save final model
    save_model(encoder, "ADDA-source-encoder-final.pt", logger)
    save_model(decoder, "ADDA-source-classifier-final.pt", logger)

    return encoder, decoder


def eval_src(encoder, decoder, data_loader, logger):
    """Evaluate classifier for source domain."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    decoder.eval()

    # set loss function
    criterion = losses.cross_entropy2d
    running_metrics_val = losses.runningScore(3)
    val_loss_meter = losses.averageMeter()

    # evaluate network
    for (images, labels, _) in data_loader:
        images = make_variable(images, volatile=True)
        labels = make_variable(labels)

        preds = decoder(encoder(images))
        loss = criterion(preds, labels).data[0]

        val_loss_meter.update(loss.item())
        pred = preds.data.max(1)[1].cpu().numpy()
        gt = labels.data.cpu().numpy()
        running_metrics_val.update(gt, pred)

    score, class_iou = running_metrics_val.get_scores()
    for k, v in score.items():
        print(k, v)
        logger.info('{}: {}'.format(k, v))

    for k, v in class_iou.items():
        logger.info('{}: {}'.format(k, v))

    #logger.info("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))
    logger.info("Avg Loss = {}".format(val_loss_meter.avg))
