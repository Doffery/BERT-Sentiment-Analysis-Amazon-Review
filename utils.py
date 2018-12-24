from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import tensorflow as tf

user_flags = []

def get_logger(name=__file__, level=logging.INFO):
    logger = logging.getLogger(name)

    if getattr(logger, '_init_done__', None):
        logger.setLevel(level)
        return logger

    logger._init_done__ = True
    logger.propagate = False
    logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s:%(levelname)s::%(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(0)

    del logger.handlers[:]
    logger.addHandler(handler)

    return logger


logger = get_logger()


def DEFINE_string(name, default_value, doc_string):
    tf.app.flags.DEFINE_string(name, default_value, doc_string)
    global user_flags
    user_flags.append(name)


def DEFINE_integer(name, default_value, doc_string):
    tf.app.flags.DEFINE_integer(name, default_value, doc_string)
    global user_flags
    user_flags.append(name)


def DEFINE_float(name, default_value, doc_string):
    tf.app.flags.DEFINE_float(name, default_value, doc_string)
    global user_flags
    user_flags.append(name)


def DEFINE_bool(name, default_value, doc_string):
    tf.app.flags.DEFINE_boolean(name, default_value, doc_string)
    global user_flags
    user_flags.append(name)


def print_user_flags(line_limit=80):
    logger.info("-" * 80)
 
    global user_flags
    FLAGS = tf.app.flags.FLAGS
 
    for flag_name in sorted(user_flags):
        value = "{}".format(getattr(FLAGS, flag_name))
        log_string = flag_name
        log_string += "." * (line_limit - len(flag_name) - len(value))
        log_string += value
        logger.info(log_string)
