import unittest

import numpy as np

from mnist_classifier import classification_summary, prepare_images


class MnistClassifierTests(unittest.TestCase):
    def test_pixel_normalization_and_channel_dimension(self):
        images = np.array([[[0] * 28] * 28, [[255] * 28] * 28], dtype=np.uint8)
        prepared = prepare_images(images)
        self.assertEqual(prepared.shape, (2, 28, 28, 1))
        self.assertEqual(prepared.min(), 0)
        self.assertEqual(prepared.max(), 1)

    def test_metrics_and_confusion_matrix(self):
        actual = np.array([0, 1, 2])
        predicted = np.array([0, 2, 2])
        probabilities = np.zeros((3, 10))
        for i, (truth, guess) in enumerate(zip(actual, predicted)):
            if truth == guess:
                probabilities[i, truth] = 1
            else:
                probabilities[i, guess] = 0.9
                probabilities[i, truth] = 0.09
                probabilities[i, [label for label in range(10) if label not in {truth, guess}]] = 0.01 / 8
        result = classification_summary(actual, predicted, probabilities)
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertEqual(len(result["confusion_matrix"]), 10)
        self.assertAlmostEqual(result["top_2_accuracy"], 1)

    def test_invalid_images_and_probabilities(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            prepare_images(np.zeros((2, 28, 27), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            classification_summary([0], [0], np.ones((1, 10)))


if __name__ == "__main__":
    unittest.main()
