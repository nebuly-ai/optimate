import logging

from nebullvm.optional_modules.dummy import DummyClass

try:
    import torch_neuron  # noqa F401

    logging.getLogger("Neuron").setLevel(logging.WARNING)
except ImportError:
    try:
        import torch_neuronx  # noqa F401

        logging.getLogger("Neuron").setLevel(logging.WARNING)
    except ImportError:
        torch_neuron = DummyClass
