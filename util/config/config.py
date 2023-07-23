from sacred import Experiment

EXPERIMENT_NAME = 'Book'

ex = Experiment(EXPERIMENT_NAME, save_git_info=False)

@ex.config
def config():
    exp_name = EXPERIMENT_NAME
    mode = 'train'
    seed = 42
    
    # Model Hyper-Parameter Setting
    num_labels = 24
    # class_weighting = [12.409774436090226, 0.8643623985336476, 11.582456140350876, 86.86842105263158, 4.136591478696742, 1.8097587719298245, 0.1727006382756095, 86.86842105263158, 0.22359953938901309, 13.364372469635628, 0.37044102794299183, 1.4478070175438595, 0.7299867315347192, 57.91228070175438, 1.6705465587044535, 1.1281613123718386, 19.304093567251464, 9.652046783625732, 9.652046783625732]
    
    # GPU, CPU Environment Setting
    num_nodes = 1
    gpus = [0, 1, 2, 3]
    batch_size = 256
    per_gpu_batch_size = 2    # Note that -> batch_size % (per_gpu_batch_size * len(gpus) == 0
    num_workers = 3

    # Main Setting
    input_seq_len = 88 #default = 88 53/58
    image_resolution = 432 #  resolution % 16 == 0
    # num_train_epochs = max_steps / len(train_dataloader)
    max_steps = 100000
    warmup_steps = 700
    lr = 3e-5
    val_check_interval = 0.2
    model_name = '/home2/yangcw/book/Automodel'
    
    # Path Setting
    load_path = ""
    log_dir = 'result'
    train_dataset_path = r"/home2/yangcw/book/Real_train_book_filter2_data.csv"
    val_dataset_path = r"/home2/yangcw/book/Real_val_book_filter2_data.csv"
    # test_dataset_path = r"/home2/yangcw/book/Real_val_book_filter2_data.csv"
    test_dataset_path = r"/home2/yangcw/book/test_data.csv"