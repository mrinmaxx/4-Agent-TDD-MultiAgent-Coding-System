from solution import my_inc
def test_fuzz():
    xs=[]
    xs[5]=1
    assert my_inc(1)==2
