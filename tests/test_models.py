from src.models import LegalChunk


def test_ref_composition():
    lc = LegalChunk(text="...",
                    urn="dlgs:2001;231",
                    article="5",
                    comma="1")
    assert lc.ref == "dlgs:2001;231|art5|co1"

def test_point_id_is_deterministic():
    a = LegalChunk(text="foo", 
                   urn="dlgs:2001;231", 
                   article="5", 
                   comma="1")
    b = LegalChunk(text="bar", 
                   urn="dlgs:2001;231", 
                   article="5", 
                   comma="1")
    assert a.point_id == b.point_id # derived from ref, not from text