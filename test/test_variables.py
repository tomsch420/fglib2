import unittest

from fglib2.variables import Symbolic


class VariableTestCase(unittest.TestCase):

    def test_creation(self):
        variable = Symbolic("X", [0, 1])
        self.assertEqual(variable.name, "X")
        self.assertEqual(len(variable.domain), 2)

    def test_encode(self):
        variable = Symbolic("X", [0, 1])
        self.assertEqual(variable.encode(0), 0)

    def test_ordering(self):
        x = Symbolic("X", [0, 1])
        y = Symbolic("Y", [0, 1])
        z = Symbolic("Z", [0, 1])

        self.assertEqual([x, y, z], list(sorted([y, z, x])))


if __name__ == '__main__':
    unittest.main()
