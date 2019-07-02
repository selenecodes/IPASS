def str2bool(v):
    """ Converts the --train flag to a boolean

        Parameters
        -------
        v : bool|string, True
            The input value for the --train flag.
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')