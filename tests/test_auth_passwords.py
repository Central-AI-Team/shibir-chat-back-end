"""app/auth/passwords.py: argon2id hashing, verify, rehash and the policy."""

from __future__ import annotations

import pytest
from argon2 import PasswordHasher

from app.auth import passwords


def test_hash_is_argon2id_and_verifies():
    h = passwords.hash_password("correct horse battery")
    assert h.startswith("$argon2id$")
    assert passwords.verify_password(h, "correct horse battery")
    assert not passwords.verify_password(h, "correct horse batterY")


def test_verify_never_raises_on_garbage_hash():
    assert passwords.verify_password("not-a-hash", "whatever123") is False


def test_production_hasher_uses_library_defaults():
    # conftest swaps in a cheap hasher; the module's own default must not be weakened.
    import importlib

    fresh = importlib.reload(passwords)
    try:
        default = PasswordHasher()
        assert fresh._hasher.time_cost == default.time_cost
        assert fresh._hasher.memory_cost == default.memory_cost
        assert fresh._hasher.parallelism == default.parallelism
    finally:
        importlib.reload(passwords)


def test_needs_rehash_when_parameters_change(monkeypatch):
    weak = PasswordHasher(time_cost=1, memory_cost=8, parallelism=1).hash("correct horse battery")
    monkeypatch.setattr(passwords, "_hasher", PasswordHasher(time_cost=2, memory_cost=16, parallelism=1))
    assert passwords.needs_rehash(weak)
    assert not passwords.needs_rehash(passwords.hash_password("correct horse battery"))
    assert passwords.needs_rehash("garbage")


def test_dummy_verify_runs_a_real_verify(monkeypatch):
    calls = []
    real = passwords.verify_password
    monkeypatch.setattr(passwords, "verify_password", lambda h, p: calls.append(h) or real(h, p))
    passwords.dummy_verify("anything at all")
    assert len(calls) == 1 and calls[0].startswith("$argon2id$")


@pytest.mark.parametrize(
    "password, ok",
    [
        ("a" * 9, False),
        ("a" * 10, True),
        ("a" * 128, True),
        ("a" * 129, False),
        (" " * 12, False),
        ("\t\n" * 6, False),
        ("alice.smith", False),  # equals the email local part
        ("ALICE.SMITH", False),
        ("alice.smith2", True),
        ("all lowercase no digits", True),  # no composition rules
    ],
)
def test_policy(password, ok):
    assert (passwords.policy_error(password, "alice.smith@example.com") is None) is ok
