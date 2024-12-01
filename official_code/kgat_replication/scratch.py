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


import torch

if torch.cuda.is_available():
    print(f"GPU is available: {torch.cuda.get_device_name(0)}")
else:
    print("GPU is not available. Using CPU.")
