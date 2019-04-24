
__authors__ = ["Melissa", "Ronaldo", "Pedro Pablo"]

from heuristic import BinaryTest, IntegerTest


if __name__ == "__main__":
    """
    Run 30 simulations for dimensions 5, 10, 25 and 50
     in the binary problem
    """
    test = BinaryTest([5, 10, 25, 50], 30)
    test.run()

    """
    Run 30 simulations for dimensions 5, 10, 25 and 50
     in the integer problem
    """
    # test = IntegerTest([1, 5, 10, 25], 10)
    # test.run()
