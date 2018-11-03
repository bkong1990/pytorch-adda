"""Main script for ADDA."""
import random
import os
from PIL import Image
import numpy as np

import params
from utils import make_variable
import losses
from models import VoidDiscriminator, UNet, Discriminator, Discriminator1
from utils import get_data_loader, init_model, init_random_seed, get_logger

def eval_save_result(encoder, classifier, data_loader):
    """Evaluation for target encoder by source classifier on target dataset."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    if not os.path.exists(params.save_dir):
        os.makedirs(params.save_dir)
    class_map = {0: 255, 1: 128, 2: 0}

    # set loss function
    criterion = losses.cross_entropy2d
    running_metrics_val = losses.runningScore(3)
    val_loss_meter = losses.averageMeter()

    # evaluate network
    for (images, labels, filenames) in data_loader:
        images = make_variable(images, volatile=True)
        labels = make_variable(labels).squeeze_()

        preds = classifier(encoder(images))
        loss = criterion(preds, labels).data[0]

        val_loss_meter.update(loss.item())
        pred = preds.data.max(1)[1].cpu().numpy()
        gt = labels.data.cpu().numpy()
        running_metrics_val.update(gt, pred)

        for idx in range(gt.shape[0]):
            filename = os.path.join(params.save_dir, filenames[idx] + '.bmp')
            pred_save = pred[idx]

            for _validc in class_map.keys():
                pred_save[pred_save == _validc] = class_map[_validc]
            pred_save = Image.fromarray(pred_save.astype(np.uint8))
            pred_save.save(filename)


    # loss /= len(data_loader)
    # acc = acc.float() / float(len(data_loader.dataset))

    # print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))
    score, class_iou = running_metrics_val.get_scores()
    for k, v in score.items():
        print(k, v)
        print('{}: {}'.format(k, v))

    for k, v in class_iou.items():
        print('{}: {}'.format(k, v))

    #logger.info("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))
    print("Avg Loss = {}".format(val_loss_meter.avg))

if __name__ == '__main__':
    # load dataset
    data_loader_eval = get_data_loader(params.tgt_dataset, train=False)

    # load models
    encoder = init_model(net=UNet(n_classes=3),
                             restore=params.save_encoder_restore)
    decoder = init_model(net=VoidDiscriminator(),
                             restore=params.save_classifier_restore)

    eval_save_result(encoder, decoder, data_loader_eval)