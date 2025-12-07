from qaoa_qubo.qubo import QUBOProblem


def test_qubo_evaluate_simple():
    # C(x) = x0 + 2 x1 + 3 x0 x1 + 0.5
    linear = {0: 1.0, 1: 2.0}
    quadratic = {(0, 1): 3.0}
    qubo = QUBOProblem.from_dicts(linear, quadratic, constant=0.5)

    # x = 00 → C = 0 + 0 + 0 + 0.5
    assert qubo.evaluate("00") == 0.5

    # x = 10 → C = 1 + 0 + 0 + 0.5
    assert qubo.evaluate("10") == 1.5

    # x = 01 → C = 0 + 2 + 0 + 0.5
    assert qubo.evaluate("01") == 2.5

    # x = 11 → C = 1 + 2 + 3*1*1 + 0.5 = 6.5
    assert qubo.evaluate("11") == 6.5
