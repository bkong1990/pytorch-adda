"""Main script for ADDA."""
import random
import os

import params
from core import eval_src, eval_tgt, train_src, train_tgt
from models import VoidDiscriminator, UNet, Discriminator, Discriminator1
from utils import get_data_loader, init_model, init_random_seed, get_logger

if __name__ == '__main__':
    # init random seed and logger
    init_random_seed(params.manual_seed)
    logger = get_logger('runs')

    # load dataset
    src_data_loader = get_data_loader(params.src_dataset)
    src_data_loader_eval = get_data_loader(params.src_dataset, train=False)
    tgt_data_loader = get_data_loader(params.tgt_dataset)
    tgt_data_loader_eval = get_data_loader(params.tgt_dataset, train=False)

    # load models
    src_encoder = init_model(net=UNet(n_classes=3),
                             restore=params.src_encoder_restore)
    src_decoder = init_model(net=VoidDiscriminator(),
                             restore=params.src_classifier_restore)
    tgt_encoder = init_model(net=UNet(n_classes=3),
                             restore=params.tgt_encoder_restore)
    critic = init_model(Discriminator1(n_classes=params.d_output_dims),
                        restore=params.d_model_restore)

    # train source model
    logger.info("=== Training classifier for source domain ===")
    logger.info(">>> Source Encoder <<<")
    logger.info(src_encoder)
    logger.info(">>> Source Classifier <<<")
    logger.info(src_decoder)

    if not (src_encoder.restored and src_decoder.restored and
            params.src_model_trained):
        src_encoder, src_decoder = train_src(
            src_encoder, src_decoder, src_data_loader, logger)

    # eval source model
    logger.info("=== Evaluating classifier for source domain ===")
    eval_src(src_encoder, src_decoder, src_data_loader_eval, logger)

    # train target encoder by GAN
    logger.info("=== Training encoder for target domain ===")
    logger.info(">>> Target Encoder <<<")
    logger.info(tgt_encoder)
    logger.info(">>> Critic <<<")
    logger.info(critic)

    # init weights of target encoder with those of source encoder
    if not tgt_encoder.restored:
        logger.info('Initializaing tgt_encoder with src_encoder')
        tgt_encoder.load_state_dict(src_encoder.state_dict())

    if not (tgt_encoder.restored and critic.restored and
            params.tgt_model_trained):
        tgt_encoder = train_tgt(src_encoder, tgt_encoder, critic,
                                src_data_loader, tgt_data_loader, src_decoder, tgt_data_loader_eval, logger)

    # eval target encoder on test set of target dataset
    logger.info("=== Evaluating classifier for encoded target domain ===")
    logger.info(">>> source only <<<")
    eval_tgt(src_encoder, src_decoder, tgt_data_loader_eval, logger)
    logger.info(">>> domain adaption <<<")
    eval_tgt(tgt_encoder, src_decoder, tgt_data_loader_eval, logger)
