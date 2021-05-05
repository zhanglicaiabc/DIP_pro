import tensorflow as tf

def test(model, test_db):
    total_correct = tf.constant(0, dtype=tf.int32)
    for (x, y) in test_db:
        pred = model(x, training=True)
        pred = tf.squeeze(pred)
        pred = tf.argmax(pred, axis=1)
        y = tf.argmax(y, axis=1)
        # pred = tf.cast(pred, dtype=tf.int32)
        # y = tf.cast(y, dtype=tf.int64)
        correct = tf.equal(pred, y)
        total_correct += tf.reduce_sum(tf.cast(correct, dtype=tf.int32))

    return total_correct / 200

