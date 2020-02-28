import tensorflow as tf


def softmax_cross_entropy(logits, labels):
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    loss = tf.reduce_sum(ce)
    correct = tf.reduce_sum(tf.cast(tf.math.equal(tf.argmax(logits, axis=1), labels), tf.float32))
    return loss, correct
