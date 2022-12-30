import numpy as np


def loop_participants(func):
    def wrapper(*args):
        if args[0].loop_participants:
            for participant in np.arange(args[0].N):
                args[0].ex_participant = participant
                func(*args)
        else:
            func(*args)

    return wrapper


def loop_regions(func):
    def wrapper(*args):
        if args[0].loop_regions:
            for region in np.arange(args[0].N_regions):
                args[0].ex_region = region
                func(*args)
        else:
            func(*args)

    return wrapper


def loop_domains(func):
    def wrapper(*args):
        for domain in args[0].domains:
            args[0].domain = domain
            func(*args)

    return wrapper


def loop_modalities(func, interp=True):
    def wrapper(*args):
        for modality in args[0].modalities:
            args[0].modality = modality
            args[0].choose_modality_data(modality, interp)
            func(*args)

    return wrapper
