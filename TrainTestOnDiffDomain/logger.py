
import tensorflow as tf
import numpy as np
import scipy.misc
from PIL import Image
import io

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO  # Python 3.x


class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer =  tf.summary.create_file_writer(log_dir) 

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        # summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        # self.writer.add_summary(summary, step)
        # self.writer.flush()
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step)
            self.writer.flush()
   

    def image_summary(self, tag, images, step):
        """Log a list of images."""
        with self.writer.as_default():
            for i, img in enumerate(images):
                # Convert the numpy array to a PIL Image
                img = Image.fromarray((img * 255).astype(np.uint8))

                # Create a batch with a single image
                img_batch = np.expand_dims(np.array(img), axis=0)

                # Log the image as a batch
                tf.summary.image(name='%s/%d' % (tag, i), data=img_batch, step=step)
        self.writer.flush()


    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))
        

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]
        hist.bucket_limit.extend(bin_edges)
        hist.bucket.extend(counts)

        # # Add bin edges and counts
        # for edge in bin_edges:
        #     hist.bucket_limit.append(edge)
        # for c in counts:
        #     hist.bucket.append(c)

        # Create and write Summary
        # summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        # self.writer.add_summary(summary, step)
        # self.writer.flush()

        # Create and write Summary
        summary = tf.summary.histogram(tag, values, step=step)
        with self.writer.as_default():
            tf.summary.experimental.write_raw(summary, step, name=tag)
        self.writer.flush()