import unittest

from deeptrace.chunk import make_windows, split_sentences


class TestChunking(unittest.TestCase):
    def test_split_drops_short_fragments(self):
        text = (
            "This is a short one. Hi. "
            "The second sentence is definitely long enough to keep."
        )
        self.assertEqual(len(split_sentences(text)), 2)

    def test_windows_overlap(self):
        sentences = ["one two three four five"] * 3
        self.assertEqual(len(make_windows(sentences, size=2, stride=1)), 2)

    def test_short_input_becomes_single_window(self):
        sentences = ["one two three four five"]
        self.assertEqual(make_windows(sentences), ["one two three four five"])


if __name__ == "__main__":
    unittest.main()
