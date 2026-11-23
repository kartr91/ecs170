from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Evaluate_Classification import Evaluate_Classification
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Setting_Provided_Split import Setting_Provided_Split
import numpy as np
import torch

# ---- Multi-Layer Perceptron script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # chosen parameters for experiment
    # hidden_sizes - 128 neurons, 1 hidden layer; also did an experiment with 128 x 2 hidden layers
    hidden_sizes = (128,)

    # only trained for 10 epochs in each experiment, this was constant
    max_epochs = 10
    # used CE, MSE, and multimarginloss to measure avg prediction error/cost
    loss_function = 'cross_entropy'
    # used a minibatch size of 128 in each experiment, this was constant
    batch_size = 128
    # used either Adam or SGD (minibatch GD) for updating weights
    opt = 'adam'
    # learning rate - 0.001, also did an experiment with 0.01 (larger weight updates by a factor of10) w/ potential for overshooting
    alpha = 0.001

    # set path for input/output
    data_directory = '../../data/stage_2_data'
    output_directory = '../../result/stage_2_result/baseline'

    # ---- object initialization section ----
    data_obj = Dataset_Loader(data_directory)

    method_obj = Method_MLP(
        hidden_sizes=hidden_sizes,
        learning_rate=alpha,
        max_epoch=max_epochs,
        batch_size=batch_size,
        opt=opt,
        loss_name=loss_function,
    )
    method_obj.config['seed'] = 2

    result_obj = Result_Saver(output_directory)
    setting_obj = Setting_Provided_Split()
    evaluate_obj = Evaluate_Classification()

    # ---- running section ----
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    metrics = setting_obj.load_run_save_evaluate()

    print('************ Overall Performance ************')
    for name, value in metrics.items():
        if name != 'confusion_matrix':
            print(f'{name}: {value:.6f}')
    print('************ Finish ************')
