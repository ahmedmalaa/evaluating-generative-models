def check_tf2():
    found = False
    message = "Note: TensorFlow 2.x not found, some functionality may not be available."
    try:
        import tensorflow as tf
        if str(tf.__version__).split(".")[0] == "2":  # pylint: disable=no-member
            found = True
    except ImportError:
        pass
    if not found:
        print(message)
    return found
