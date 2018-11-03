"""Test script to classify target data."""

import torch
import torch.nn as nn

from utils import make_variable
import losses


def eval_tgt(encoder, classifier, data_loader, logger):
    """Evaluation for target encoder by source classifier on target dataset."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # set loss function
    criterion = losses.cross_entropy2d
    running_metrics_val = losses.runningScore(3)
    val_loss_meter = losses.averageMeter()

    # evaluate network
    for (images, labels, _) in data_loader:
        images = make_variable(images, volatile=True)
        labels = make_variable(labels).squeeze_()

        preds = classifier(encoder(images))
        loss = criterion(preds, labels).data[0]

        val_loss_meter.update(loss.item())
        pred = preds.data.max(1)[1].cpu().numpy()
        gt = labels.data.cpu().numpy()
        running_metrics_val.update(gt, pred)

    # loss /= len(data_loader)
    # acc = acc.float() / float(len(data_loader.dataset))

    # print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))
    score, class_iou = running_metrics_val.get_scores()
    for k, v in score.items():
        print(k, v)
        logger.info('{}: {}'.format(k, v))

    for k, v in class_iou.items():
        logger.info('{}: {}'.format(k, v))
        print('{}: {}'.format(k, v))

    #logger.info("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))
    logger.info("Avg Loss = {}".format(val_loss_meter.avg))
