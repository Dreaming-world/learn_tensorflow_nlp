# learn_tensorflow_nlp

# padding and mask

losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                    labels=self.labels)
        mask = tf.sequence_mask(self.sequence_lengths) #mask
        losses = tf.boolean_mask(losses, mask) # mask
        self.cost = tf.reduce_mean(losses)
        self.labels_softmax_ = tf.argmax(self.logits, axis=-1)
        self.labels_softmax_ = tf.cast(self.labels_softmax_, tf.int32)
 ## elf.sequence_length每一个句子的长度
