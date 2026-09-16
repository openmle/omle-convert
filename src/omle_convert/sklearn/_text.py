"""sklearn text transformers → OMLE omle.text operators.

Supported:
  CountVectorizer    → [RegexTokenize] → [StopWordsRemove] → [NGram] → CountVectorize
  TfidfTransformer   → TfIdfTransform
  TfidfVectorizer    → (CountVectorizer chain) → TfIdfTransform
  HashingVectorizer  → [RegexTokenize] → [NGram] → HashingVectorize
"""
from __future__ import annotations

import re

import numpy as np

import omle

from ._builder import Builder


def _op_snake(op: str) -> str:
    """Convert an op name like 'RegexTokenizer' to 'regex_tokenizer'."""
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', op)
    return re.sub(r'([a-z\d])([A-Z])', r'\1_\2', s).lower()

_TEXT_TRANSFORMER_CLASSES: frozenset[str] = frozenset({
    "CountVectorizer",
    "TfidfTransformer",
    "TfidfVectorizer",
    "HashingVectorizer",
})


def _text_node(op: str, input_name: str, output_name: str,
               attributes: list[omle.Attribute], builder: Builder,
               prefix: str = "") -> str:
    node_name = builder.unique_node_name(f"{prefix}_{_op_snake(op)}" if prefix else output_name)
    builder.add_node(omle.Node(
        name=node_name,
        domain="omle.text",
        op=op,
        inputs=[omle.NodeInput(name=input_name)],
        outputs=[omle.NodeOutput(name=output_name)],
        attributes=attributes,
    ))
    return output_name


def _tokenize(t, input_name: str, prefix: str, builder: Builder) -> str:
    """Emit a RegexTokenize node. Output tensor: {input_name}_tokens."""
    analyzer = getattr(t, "analyzer", "word")
    if analyzer != "word":
        return input_name  # char/char_wb: no tokenizer emitted

    pattern = getattr(t, "token_pattern", None) or r"(?u)\b\w\w+\b"
    tok_out = builder.unique_name(f"{input_name}_tokens")
    return _text_node("RegexTokenizer", input_name, tok_out, [
        omle.Attribute(name="pattern", s=pattern),
    ], builder, prefix=prefix)


def _stop_words(t, tok_name: str, prefix: str, builder: Builder) -> str:
    """Emit StopWordsRemove if stop words are configured. Output tensor: {tok_name}_sw."""
    stop_words = getattr(t, "stop_words_", None)
    if not stop_words:
        return tok_name
    sw_out = builder.unique_name(f"{tok_name}_sw")
    return _text_node("StopWordsRemover", tok_name, sw_out, [
        builder.string_tensor_attr("stop_words", f"{prefix}_stopwords", sorted(stop_words)),
    ], builder, prefix=prefix)


def _ngram(t, tok_name: str, prefix: str, builder: Builder) -> str:
    """Emit an NGram node if ngram_range extends beyond unigrams. Output tensor: {tok_name}_ngrams."""
    ngram_min, ngram_max = getattr(t, "ngram_range", (1, 1))
    if ngram_min == 1 and ngram_max == 1:
        return tok_name
    ng_out = builder.unique_name(f"{tok_name}_ngrams")
    attrs = [omle.Attribute(name="n_max", i=ngram_max)]
    if ngram_min != 1:
        attrs.append(omle.Attribute(name="n_min", i=ngram_min))
    return _text_node("NGram", tok_name, ng_out, attrs, builder, prefix=prefix)


def _count_vectorize(t, seq_name: str, prefix: str, builder: Builder) -> str:
    """Emit a CountVectorize node. Output tensor: {seq_name}_count_vec."""
    vocab = sorted(t.vocabulary_, key=t.vocabulary_.get)
    binary = bool(getattr(t, "binary", False))
    out = builder.unique_name(f"{seq_name}_count_vec")
    return _text_node("CountVectorizer", seq_name, out, [
        builder.string_tensor_attr("vocabulary", f"{prefix}_vocab", vocab),
        omle.Attribute(name="binary", b=binary),
    ], builder, prefix=prefix)


def _tfidf_weight(t, cv_name: str, prefix: str, builder: Builder) -> str:
    """Emit a TfIdfTransform node. Output tensor: {cv_name}_tfidf."""
    out = builder.unique_name(f"{cv_name}_tfidf")
    return _text_node("TfIdfTransformer", cv_name, out, [
        builder.tensor_attr("idf", f"{prefix}_idf", np.asarray(t.idf_)),
    ], builder, prefix=prefix)


def _count_vectorizer(t, input_name: str, prefix: str, builder: Builder) -> str:
    cur = _tokenize(t, input_name, prefix, builder)
    cur = _stop_words(t, cur, prefix, builder)
    cur = _ngram(t, cur, prefix, builder)
    return _count_vectorize(t, cur, prefix, builder)


def _tfidf_transformer(t, input_name: str, prefix: str, builder: Builder) -> str:
    return _tfidf_weight(t, input_name, prefix, builder)


def _tfidf_vectorizer(t, input_name: str, prefix: str, builder: Builder) -> str:
    cur = _tokenize(t, input_name, prefix, builder)
    cur = _stop_words(t, cur, prefix, builder)
    cur = _ngram(t, cur, prefix, builder)
    cur = _count_vectorize(t, cur, prefix, builder)
    return _tfidf_weight(t, cur, prefix, builder)


def _hashing_vectorizer(t, input_name: str, prefix: str, builder: Builder) -> str:
    cur = _tokenize(t, input_name, prefix, builder)
    cur = _ngram(t, cur, prefix, builder)
    num_features = int(getattr(t, "n_features", 2 ** 20))
    binary = bool(getattr(t, "binary", False))
    alternate_sign = bool(getattr(t, "alternate_sign", False))
    out = builder.unique_name(f"{cur}_hv")
    return _text_node("HashingVectorizer", cur, out, [
        omle.Attribute(name="num_features", i=num_features),
        omle.Attribute(name="binary", b=binary),
        omle.Attribute(name="alternate_sign", b=alternate_sign),
    ], builder, prefix=prefix)


_TEXT_DISPATCH = {
    "CountVectorizer":   _count_vectorizer,
    "TfidfTransformer":  _tfidf_transformer,
    "TfidfVectorizer":   _tfidf_vectorizer,
    "HashingVectorizer": _hashing_vectorizer,
}
