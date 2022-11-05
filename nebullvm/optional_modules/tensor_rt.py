try:
    import tensorrt
    from tensorrt import IInt8EntropyCalibrator2
except ImportError:
    tensorrt = object
    IInt8EntropyCalibrator2 = object

try:
    import polygraphy.cuda as polygraphy
except ImportError:
    polygraphy = object
