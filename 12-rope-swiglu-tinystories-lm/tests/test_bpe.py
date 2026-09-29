import random

import pytest

from tfs.bpe import EOT, BPETokenizer, pretokenize

CORPUS = [
    "Once upon a time, there was a little dog named Max. Max liked to run.",
    "The cat sat on the mat. The cat was happy! Wasn't it? 123 4567",
    "Lily and Tom played in the park. They're best friends, and they'll play again.",
] * 20

TRICKY = [
    "",
    " ",
    "\n\n\t  ",
    "hello_world snake_case __init__",
    "tabs\tand\r\nwindows newlines",
    "naïve café, 東京, emoji 🐶🐱 and zwj 👩‍👩‍👧",
    "numbers 3.14159 and 1,000,000",
    "   leading and trailing   ",
    "odd punctuation ... !!! ??? --- ~~~ @@@ ### $$$",
    f"two stories{EOT}joined by the separator{EOT}",
]


@pytest.fixture(scope="module")
def tok() -> BPETokenizer:
    return BPETokenizer.train(CORPUS, vocab_size=256 + 90 + 1, special_tokens=[EOT])


def test_pretokenize_covers_every_character():
    # Any character the regex fails to match would vanish from the output.
    for s in TRICKY:
        assert "".join(pretokenize(s)) == s


@pytest.mark.parametrize("text", TRICKY + CORPUS[:3])
def test_roundtrip_tricky(tok, text):
    assert tok.decode(tok.encode(text)) == text


def test_roundtrip_random_unicode(tok):
    rng = random.Random(0)
    alphabet = [chr(c) for c in range(32, 127)] + list("éß漢字🙂\n\t _") + ["́"]
    for _ in range(300):
        s = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
        assert tok.decode(tok.encode(s)) == s


def test_merges_compress_training_text(tok):
    text = CORPUS[0]
    assert len(tok.encode(text)) < len(text.encode("utf-8")) / 2


def test_vocab_size_and_special_token(tok):
    assert tok.vocab_size == 256 + 90 + 1
    assert tok.eot_id == tok.vocab_size - 1
    ids = tok.encode(f"a{EOT}b")
    assert ids.count(tok.eot_id) == 1
    # with allow_special=False the marker is just text, encoded as bytes/merges
    assert tok.eot_id not in tok.encode(f"a{EOT}b", allow_special=False)


def test_classic_example_first_merge():
    # "aaabdaaabac": the most frequent pair is (a, a); it must be merge 0.
    t = BPETokenizer.train(["aaabdaaabac"], vocab_size=257)
    assert t.merges[0] == (ord("a"), ord("a"))


def test_training_is_deterministic():
    a = BPETokenizer.train(CORPUS, vocab_size=300)
    b = BPETokenizer.train(CORPUS, vocab_size=300)
    assert a.merges == b.merges


def test_encode_matches_training_segmentation(tok):
    # Each merge must be usable: the merged token for every learned pair should
    # appear when we encode the exact bytes that pair decodes to.
    for i in range(len(tok.merges)):
        token_bytes = tok.vocab[256 + i]
        try:
            s = token_bytes.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if len(pretokenize(s)) == 1:
            assert tok.encode(s) == [256 + i], (s, tok.encode(s))


def test_save_load(tok, tmp_path):
    p = tmp_path / "tok.json"
    tok.save(p)
    t2 = BPETokenizer.load(p)
    assert t2.merges == tok.merges
    assert t2.special_tokens == tok.special_tokens
    for s in TRICKY:
        assert t2.encode(s) == tok.encode(s)


def test_training_stops_when_pairs_run_out():
    # "ab" has exactly one pair; asking for many merges must not loop or crash.
    t = BPETokenizer.train(["ab ab ab"], vocab_size=1000)
    assert len(t.merges) < 10
    assert t.decode(t.encode("ab ab")) == "ab ab"
