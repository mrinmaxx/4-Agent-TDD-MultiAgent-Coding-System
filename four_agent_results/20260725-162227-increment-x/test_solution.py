from solution import my_inc
def _reference(x): return x+1
def test_fuzz():
    for x in range(-8,8):
        assert my_inc(x)==_reference(x)
