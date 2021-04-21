import argparse
def parse_helper():
    #data from all slices, segm or central meniscus
    dataset = [r"/home/students/thampi/PycharmProjects/meniscus_data/all_slices.hdf5", r"/home/students/thampi/PycharmProjects/meniscus_data/segm.hdf5", r"/home/students/thampi/PycharmProjects/meniscus_data/filt_data",r"/home/students/thampi/PycharmProjects/meniscus_data/segm_npy"]#559 elements in filt data

    #edgenet or cycle net training highly depende don learning rate
    no_epo, second_start, num_test_show, valid_after, is_semeda, batch_size, in_chann = 40, 2000, 7, 4, False, 2, 1
    datafrom,  cropsize, pretrained, tr_type, epo2, num_data_img, experiment_name = 3, 256, False, "baseline", 1, None, "test"

    parser = argparse.ArgumentParser(description='Parameters for training')

    model_training_types = ["baseline", "faster_rcnn_segm", "cascaded_unet", "cycle_net", "patch_init", "extraggr", "deepsup", "unets_series", "semedanet", "combined_unets", "train_side"]
    # Add the arguments
    parser.add_argument('-train_type', metavar='training types', type=str, help='type of training the model', choices=model_training_types, default=tr_type)
    parser.add_argument('-epochs', metavar='number of epochs', type=int, help='number of epochs', default=no_epo)
    parser.add_argument('-batch_size', metavar='batch size', type=int, help='batch size', default=batch_size)
    parser.add_argument('-tot_imgs', metavar='total expt images', type=int, help='total number of images to be used for expt', default=num_data_img)
    parser.add_argument('-second_start', metavar='second start', type=int, help='second dataset start index', default=second_start)
    parser.add_argument('-num_test_show', metavar='number of display images', type=int, help='number of test images to be displayed', default=num_test_show)
    parser.add_argument('-valid_after', metavar='valid after', type=int, help='validation every', default=valid_after)
    parser.add_argument('-epoch2', metavar='epoch 2 cycle', type=int, help='number of epochs in second stage', default=epo2)
    parser.add_argument('-is_semeda', metavar='is semeda', type=bool, help='if semeda training (else cyclenet)', default=is_semeda)
    parser.add_argument('-datafrom', metavar='datafrom', type=int, help='experiment data location, choose 0 ,1 ,2 or 3', default=datafrom)
    parser.add_argument('-in_chann', metavar='in channels', type=int, help='in channels', default=in_chann)
    parser.add_argument('-cropsize', metavar='cropsize', type=int, help='images to be cropped in size', default=cropsize)
    parser.add_argument('-expt_name', metavar='expt name', type=str, help='experiment name', default=experiment_name)
    parser.add_argument('-pretrained', metavar='use previously saved', type=bool, help='initialize with previously saved model', default=pretrained)
    args = parser.parse_args()
    datafrom = dataset[args.datafrom]
    if_hdf5 = True if datafrom[-4:]=="hdf5" else False
    return args.epochs, args.tot_imgs, args.second_start, args.num_test_show, args.valid_after, args.epoch2, args.is_semeda, datafrom, args.in_chann, args.cropsize, args.expt_name, if_hdf5, args.batch_size, args.pretrained, args.train_type
