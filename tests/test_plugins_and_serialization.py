"""Tests for mock plugins and serialization utilities."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import pytest

from agentic_os.core.planning.goal import create_goal
from agentic_os.exceptions import ValidationError
from agentic_os.plugins.mock import (
    MockEvaluator,
    MockExecutor,
    MockLLM,
    MockMemoryStore,
)
from agentic_os.utils.serialization import (
    dataclass_to_dict,
    load_json,
    save_json,
    to_json,
)

# --- Mock Plugins ---


class TestMockLLM:
    def test_default_response(self):
        llm = MockLLM()
        result = llm.generate("hello world")
        assert "[Mock]" in result
        assert len(llm.call_log) == 1

    def test_matched_response(self):
        llm = MockLLM(responses={"weather": "Sunny, 25C"})
        result = llm.generate("What is the weather today?")
        assert result == "Sunny, 25C"

    def test_case_insensitive_match(self):
        llm = MockLLM(responses={"python": "Python is great"})
        result = llm.generate("I love PYTHON")
        assert result == "Python is great"

    def test_embed_deterministic(self):
        llm = MockLLM()
        v1 = llm.embed("test")
        v2 = llm.embed("test")
        assert len(v1) == 8
        assert v1 == v2

    def test_embed_normalized(self):
        llm = MockLLM()
        vec = llm.embed("normalize")
        norm = sum(v * v for v in vec) ** 0.5
        assert abs(norm - 1.0) < 0.01


class TestMockEvaluator:
    def test_deterministic(self):
        ev = MockEvaluator(bias=0.5)
        s1 = ev.evaluate({}, "test thought")
        s2 = ev.evaluate({}, "test thought")
        assert s1 == s2
        assert 0.0 <= s1 <= 1.0

    def test_bias_affects_score(self):
        ev_low = MockEvaluator(bias=0.1)
        ev_high = MockEvaluator(bias=0.9)
        score_low = ev_low.evaluate({}, "same input")
        score_high = ev_high.evaluate({}, "same input")
        assert score_high > score_low

    def test_call_log(self):
        ev = MockEvaluator()
        ev.evaluate({"k": "v"}, "thought")
        assert len(ev.call_log) == 1
        assert ev.call_log[0][1] == "thought"


class TestMockExecutor:
    def test_default_success(self):
        g = create_goal("test goal")
        ex = MockExecutor()
        success, msg = ex.execute(g)
        assert success
        assert "test goal" in msg

    def test_match_by_id(self):
        g = create_goal("test")
        ex = MockExecutor(results={g.id: (False, "failed")})
        success, msg = ex.execute(g)
        assert not success
        assert msg == "failed"

    def test_match_by_description(self):
        g = create_goal("deploy")
        ex = MockExecutor(results={"deploy": (True, "done")})
        success, _msg = ex.execute(g)
        assert success

    def test_call_log(self):
        g = create_goal("log test")
        ex = MockExecutor()
        ex.execute(g)
        assert len(ex.call_log) == 1


class TestMockMemoryStore:
    def test_save_and_load(self):
        store = MockMemoryStore()
        store.save("k1", {"data": 42})
        assert store.load("k1") == {"data": 42}

    def test_load_missing(self):
        store = MockMemoryStore()
        assert store.load("missing") is None

    def test_delete_existing(self):
        store = MockMemoryStore()
        store.save("k1", {"data": 1})
        assert store.delete("k1") is True
        assert store.load("k1") is None

    def test_delete_missing(self):
        store = MockMemoryStore()
        assert store.delete("ghost") is False

    def test_query(self):
        store = MockMemoryStore()
        store.save("a", {"v": 1})
        store.save("b", {"v": 2})
        results = store.query(lambda k, v: k == "a")
        assert len(results) == 1
        assert results[0][0] == "a"


# --- Serialization ---


class TestToJson:
    def test_enum_encoding(self):
        class Color(Enum):
            RED = "red"

        result = to_json(Color.RED)
        assert "__enum__" in result

    def test_datetime_encoding(self):
        dt = datetime(2026, 1, 15, 12, 0)
        result = to_json(dt)
        assert "__datetime__" in result

    def test_dataclass_encoding(self):
        @dataclass
        class Item:
            name: str

        result = to_json(Item(name="test"))
        assert "__dataclass__" in result

    def test_plain_types(self):
        result = to_json({"key": "value", "num": 42})
        assert '"key"' in result


class TestSaveLoadJson:
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/test.json"
            data = {"hello": "world", "nums": [1, 2, 3]}
            save_json(data, path)
            loaded = load_json(path)
            assert loaded == data

    def test_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/a/b/c/test.json"
            save_json({"x": 1}, path)
            assert load_json(path) == {"x": 1}


class TestDataclassToDict:
    def test_basic(self):
        @dataclass
        class Simple:
            name: str
            value: int

        result = dataclass_to_dict(Simple(name="a", value=1))
        assert result == {"name": "a", "value": 1}

    def test_enum_field(self):
        class Status(Enum):
            ACTIVE = "active"

        @dataclass
        class Item:
            status: Status

        result = dataclass_to_dict(Item(status=Status.ACTIVE))
        assert result["status"] == "active"

    def test_datetime_field(self):
        dt = datetime(2026, 5, 21)

        @dataclass
        class Event:
            when: datetime

        result = dataclass_to_dict(Event(when=dt))
        assert "2026-05-21" in result["when"]

    def test_nested_dataclass(self):
        @dataclass
        class Inner:
            x: int

        @dataclass
        class Outer:
            inner: Inner

        result = dataclass_to_dict(Outer(inner=Inner(x=42)))
        assert result == {"inner": {"x": 42}}

    def test_non_dataclass_raises(self):
        with pytest.raises(ValidationError):
            dataclass_to_dict("not a dataclass")

    def test_dataclass_type_raises(self):
        @dataclass
        class Foo:
            x: int

        with pytest.raises(ValidationError):
            dataclass_to_dict(Foo)
