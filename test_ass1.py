import unittest
import os
import hashlib


class DatasetsHaveNotBeenModified(unittest.TestCase):
    """ Ensures that the datasets have not been altered.
        The datasets are too large to keep on GitHub,
        therefore this test ensures their integrity locally
    """

    def setUp(self):
        self.sha256_sums = {
            "sample.csv": "dd4c3f07b183816117553041dfe8028b5e6170c3c62c66de3cc599af16f61dbb",
            "set1_human.json": "2e6e5d040adac0ca7966f6ee04521e730c9bb5289801c4b3a0d349a4383061d9",
            "set1_machine.json": "3f91bdccef4aa19ef9e60eb32538daa8593424a7731254679560a5e4418ab688",
            "set2_human.json": "d8088d7a9950d75ac686df7bf2b5e63419b1bc251b19209912c1bf5f974cd3a4",
            "set2_machine.json": "a34d81f0c4a2845f51c7d55c9ad0f8ba03929bf8247564fd36b016e06f2103ef",
            "test.json": "27fb5aa94ccbcecce13db0d3aa4e43a7d625b56d1f221eb41a3f9dac1e49adc4",
        }
        self.filenames = tuple(self.sha256_sums.keys())

    def test_sha256_sums_match(self):
        """ Check equality with pre-calculated sha256 sums
        """
        # Don't perform test if `datasets/` does not exist, e.g. on GitHub
        if not os.path.exists('datasets'):
            print("datasets does not exist!")
            return
        for filename in self.filenames:
            with open(f"datasets/{filename}", "rb") as f:
                digest = hashlib.file_digest(f, "sha256")
                sha256_sum_local = digest.hexdigest()
                sha256_sum = self.sha256_sums[filename]
                self.assertEqual(sha256_sum, sha256_sum_local)


if __name__ == "__main__":
    unittest.main()
