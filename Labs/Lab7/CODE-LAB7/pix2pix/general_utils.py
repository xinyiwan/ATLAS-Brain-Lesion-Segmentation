import os


def create_dir(dirs):
    """
    Create directory

    args: dirs (str or list) create all dirs in 'dirs'
    """

    if isinstance(dirs, (list, tuple)):
        for d in dirs:
            if not os.path.exists(os.path.expanduser(d)):
                os.makedirs(d)
    elif isinstance(dirs, str):
        if not os.path.exists(os.path.expanduser(dirs)):
            os.makedirs(dirs)


def setup_logging(model_name, logging_dir='../../'):
    
    # Output path where we store experiment log and weights
    model_dir = os.path.join(logging_dir, 'models', model_name)

    fig_dir = os.path.join(logging_dir, 'figures')
    
    # Create if it does not exist
    create_dir([model_dir, fig_dir])