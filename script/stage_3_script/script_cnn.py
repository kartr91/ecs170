from local_code.stage_3_code.Dataset_Loaders import MINSTLoader, ORLLoader, CIFARLoader
from local_code.stage_3_code.Method_CNN import Method_CNN
from local_code.stage_3_code.Result_Saver import Result_Saver
from local_code.stage_3_code.Setting_Provided_Split import Setting_Provided_Split
from local_code.stage_3_code.Evaluate_Classification import Evaluate_Classification
import numpy as np
import torch

# basic script for stage 3
#---- Convolutional Neural Network script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)

    # pick from one of the three datasets
    #dataset = 'MNIST'
    #dataset = 'ORL'
    dataset = 'CIFAR'

    # run name for folder directory
    experiment = 'baseline'
    #------------------------------------------------------

    # ---- object initialization section ---------------
    # assign parameters based on selected data set
    # input_channels - # of color values per pixel: 3 for CIFAR (color), 1 for MNIST/ORL(greyscale)
    # num_classes - # of categories model predicts - 10 for digit classification in MNIST,
    #   40 identities in ORL, or 10 categories in CIFAR
    # offset - accounts for labeling discrepancies in data sets
        # 0 indicates the data set starts at 0, 1 indicates the data starts at 1
    if dataset == 'MNIST':
        data_obj = MINSTLoader('../../data/stage_3_data')
        input_channels = 1
        num_classes = 10
        offset = 0
    elif dataset == 'ORL':
        data_obj = ORLLoader('../../data/stage_3_data')
        input_channels = 1
        num_classes = 40
        offset = 1
    elif dataset == 'CIFAR':
        data_obj = CIFARLoader('../../data/stage_3_data')
        input_channels = 3
        num_classes = 10
        offset = 0

    model = Method_CNN(
        input_channels = input_channels, # see above
        num_classes = num_classes, # see above
        conv_channels= (16, 32), # 2 blocks, the first has 16 filters, the second has 32 - also tested
        # with a third block of 64 filters on MNIST/CIFAR;
        # more filters -> block can learn more kinds of patterns; more layers -> more steps in image recognition to combine
        kernel_size=3, # window size each filter looks at (3x3)
        padding=1, # adds 1 pixel border of 0's to each convolution, to allow for filter to be centered on outermost pixels
        stride=1, # filter moves 1 pixel at a time -> larger output
        pool_type='max', # max pooling - take the max value from a 2x2 region
        hidden_size = 128, # hidden layer # of 128 neurons, each neuron takes flattened feature map list,
        # multiplies the numbers in the list by the neuron's weights, adds them together, passes it through ReLU
        pooled_size = 4, # adaptive avg pooling - produce a 4x4 map per filter before flattening
        # also tested 8 on ORL
        learning_rate = 0.001, #learning rate alpha - controls the scale of weight updates
        max_epoch= 10, # run for 10 epochs for MNIST, 50 for ORL, 20 for CIFAR
        batch_size =128, # processes 128 images/weight update (minibatch GD)
        optimizer='adam', # GD that remembers previous gradients ("momentum"), smoothing updates and
        # changing the step size based on momentum
        loss_function='cross_entropy' # loss function, we also used MSE
    )
    
    model.config['dataset'] = dataset
    model.config['seed'] = 2
    model.config['offset'] = offset

    setting_obj = Setting_Provided_Split()
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, model,
                        Result_Saver('../../result/stage_3_result/' + dataset + '/' + experiment),
                        Evaluate_Classification(num_classes))
    setting_obj.print_setup_summary()
    metrics = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    for metric_name in metrics:
        if metric_name != 'confusion_matrix':
            print('CNN ' + metric_name + ': ' + str(metrics[metric_name]))
    print('************ Finish ************')
    # ------------------------------------------------------