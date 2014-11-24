def test_ReLU():
    from nnGen.NNlib import ReLU

    assert ReLU(0.5).eval() == 0.5
    assert ReLU(0).eval() == 0
    assert ReLU(-0.5).eval() == 0.0

