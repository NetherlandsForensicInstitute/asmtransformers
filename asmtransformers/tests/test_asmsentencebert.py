import os

import numpy as np
import pytest

from asmtransformers.models.asmsentencebert import ASMSentenceTransformer


@pytest.fixture
def anchor():
    return (
        '''[[101440, ["stp x19,x20,[sp, #-0x40]!", "adrp x1,0x131000", "ldr x1,[x1, #0xc88]", "str x30,[sp, #0x10]", "ldr x2,[x1]", "str x2,[sp, #0x38]", "mov x2,#0x0", "cbz x0,0x0010cf1c", "ldrb w20,[x0]"]],
        [1101472, ["ldrb w20,[x0]", "mov x19,x0", "cbz w20,0x0010cf1c", "adrp x1,0x131000"]], 
        [1101484, ["adrp x1,0x131000", "ldr x1,[x1, #0xc50]", "blr x1", "cmp x0,#0x1", "b.eq 0x0010cf48", "cmp w20,#0x30"]], 
        [1101504, ["cmp w20,#0x30", "b.ne 0x0010ced4", "ldrb w0,[x19, #0x1]"]], 
        [1101512, ["ldrb w0,[x19, #0x1]", "cmp w0,#0x78", "b.eq 0x0010cfc8", "adrp x6,0x131000"]], 
        [1101524, ["adrp x6,0x131000", "ldr x6,[x6, #0xe30]", "adrp x1,0x117000", "mov x0,x19", "add x5,sp,#0x34", "add x1,x1,#0x6a0", "add x4,sp,#0x30", "add x3,sp,#0x2c", "add x2,sp,#0x28", "blr x6", "cmp w0,#0x1"]], 
        [1101564, ["cmp w0,#0x1", "b.eq 0x0010cf64", "cmp w0,#0x2"]], 
        [1101572, ["cmp w0,#0x2", "b.eq 0x0010cf6c", "cmp w0,#0x3"]], 
        [1101580, ["cmp w0,#0x3", "b.eq 0x0010cf84", "cmp w0,#0x4"]], 
        [1101588, ["cmp w0,#0x4", "b.eq 0x0010cfa4", "mov w0,#0x0"]], 
        [1101596, ["mov w0,#0x0", "adrp x1,0x131000"]], 
        [1101600, ["adrp x1,0x131000", "ldr x1,[x1, #0xc88]", "ldr x3,[sp, #0x38]", "ldr x2,[x1]", "subs x3,x3,x2", "mov x2,#0x0", "b.ne 0x0010cfe8", "ldr x30,[sp, #0x10]"]], 
        [1101628, ["ldr x30,[sp, #0x10]", "ldp x19,x20,[sp], #0x40", "ret", "adrp x3,0x131000"]], 
        [1101640, ["adrp x3,0x131000", "ldr x3,[x3, #0xcb8]", "mov x0,x19", "mov w2,#0xa", "mov x1,#0x0", "blr x3", "b 0x0010cf20", "ldr w0,[sp, #0x28]"]], 
        [1101668, ["ldr w0,[sp, #0x28]", "b 0x0010cf20", "ldr w0,[sp, #0x28]"]], 
        [1101676, ["ldr w0,[sp, #0x28]", "ldrb w1,[sp, #0x2c]", "ubfiz w0,w0,#0x8,#0x8", "orr w0,w0,w1", "orr w0,w0,#0xff000000", "b 0x0010cf20", "ldp w0,w2,[sp, #0x28]"]], 
        [1101700, ["ldp w0,w2,[sp, #0x28]", "ldrb w1,[sp, #0x30]", "ubfiz w0,w0,#0x10,#0x8", "ubfiz w2,w2,#0x8,#0x8", "orr w1,w1,#0xff000000", "orr w0,w0,w2", "orr w0,w0,w1", "b 0x0010cf20", "ldp w1,w0,[sp, #0x28]"]], 
        [1101732, ["ldp w1,w0,[sp, #0x28]", "ldr w2,[sp, #0x30]", "ldrb w3,[sp, #0x34]", "ubfiz w0,w0,#0x10,#0x8", "ubfiz w2,w2,#0x8,#0x8", "orr w0,w0,w2", "orr w1,w3,w1, LSL #0x18", "orr w0,w0,w1", "b 0x0010cf20", "adrp x3,0x131000"]], 
        [1101768, ["adrp x3,0x131000", "ldr x3,[x3, #0xe30]", "mov x0,x19", "add x2,sp,#0x28", "adrp x1,0x117000", "add x1,x1,#0x698", "blr x3", "b 0x0010cefc", "adrp x0,0x131000"]], 
        [1101800, ["adrp x0,0x131000", "ldr x0,[x0, #0xc68]", "blr x0", "stp x19,x20,[sp, #-0x30]!"]]]'''
        )


@pytest.fixture(scope="session")
def model():
    path = 'NetherlandsForensicInstitute/ARM64BERT-embedding'
    return ASMSentenceTransformer.from_pretrained(path)


@pytest.mark.skipif(os.environ.get("CI") == 'true',
                    reason="don't run this test in GitLabCI")
def test_single_embedding(anchor, model):
    """We took one sample and calculated the sum, min and max of the embedding. This should be used as reference
    material to notice if anything changed over time"""
    embedding = model.encode(anchor)
    assert np.isclose(embedding.sum(), -0.05649805)
    assert np.isclose(embedding.min(), -0.11195006)
    assert np.isclose(embedding.max(), 0.115316495)


@pytest.mark.skipif(os.environ.get("CI") == 'true',
                    reason="don't run this test in GitLabCI")
def test_compare_identical(anchor, model):
    """This test ensures that batch size in inference does not affect the embeddings that come out
    Some variation is allowed for float rounding errors (this tolerance is based on 16 bit floating points)
    This test should still run regardless of the model we use"""
    embedding = model.encode(anchor)
    embeddings = model.encode([anchor, anchor])
    assert np.allclose(embedding, embeddings[0], rtol=1e-3)
    assert np.allclose(embedding, embeddings[1], rtol=1e-3)
