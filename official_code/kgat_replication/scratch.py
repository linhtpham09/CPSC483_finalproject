# import pandas as pd 
# import numpy as np 
# from tensorflow import compat as ttf
# tf=ttf.v1
# tf.disable_v2_behavior()
# import numpy as np 
# from utility.loader_kgat import KGAT_loader
# from utility.helper import *
# from utility.batch_test import *

# args = parse_args() 
# layer = '-'.join([str(l) for l in eval(args.layer_size)])
# weights_save_path = 'data/weights/'
# ensureDir(weights_save_path)
# save_saver = tf.train.Saver(max_to_keep=1)

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)

# sess.run(tf.global_variables_initializer())

# uninitialized_vars = tf.report_uninitialized_variables()
# print("Uninitialized variables:", sess.run(uninitialized_vars))


# from KGAT import KGAT

# # Instantiate the model
# model = KGAT(data_config=config, pretrain_data=pretrain_data, args=args)

# # Start a session
# with tf.Session() as sess:
#     # Initialize all variables
#     sess.run(tf.global_variables_initializer())

#     # Debug: Check the initialized variables
#     print("All variables:", [v.name for v in tf.global_variables()])

#     # Run testing or training
#     ret = test(sess, model, users_to_test, drop_flag=False, batch_test_flag=batch_test_flag)


# import tensorflow as tf

# print("TensorFlow version:", tf.__version__)
# print("GPU devices:", tf.config.list_physical_devices('GPU'))

# # Perform a small computation
# a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
# b = tf.matmul(a, a)
# print("Result:", b)


# import torch

# # Use Metal (MPS) if available, otherwise fallback to CPU
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# print(f"Using device: {device}")

# Move your model and data to the selected device
# model = model.to(device)
# data = data.to(device)

# import tensorflow as tf

# # Check TensorFlow version
# print("TensorFlow version:", tf.__version__)

# # Check for GPU devices
# gpu_devices = tf.config.list_physical_devices('GPU')
# if gpu_devices:
#     print("Available GPU devices:", gpu_devices)
# else:
#     print("No GPU devices available. Running on CPU.")


