"""Tests for SharedMemoryGraph and AgentMemoryHandle."""

from __future__ import annotations

import threading

from agentic_os.core.graph.edge import EdgeType
from agentic_os.core.graph.node import MemoryNode, NodeType, create_fact
from agentic_os.core.memory.shared import AgentMemoryHandle, SharedMemoryGraph


class TestRegisterAgent:
    def test_register_and_get_handle(self) -> None:
        shared = SharedMemoryGraph()
        handle = shared.register_agent("alice")
        assert isinstance(handle, AgentMemoryHandle)
        assert handle.agent_id == "alice"
        assert "alice" in shared.list_agents()

    def test_register_duplicate_raises(self) -> None:
        shared = SharedMemoryGraph()
        shared.register_agent("alice")
        try:
            shared.register_agent("alice")
            raise AssertionError("Expected ValueError")
        except ValueError:
            pass

    def test_register_multiple_agents(self) -> None:
        shared = SharedMemoryGraph()
        shared.register_agent("alice")
        shared.register_agent("bob")
        agents = shared.list_agents()
        assert "alice" in agents
        assert "bob" in agents


class TestUnregisterAgent:
    def test_unregister_removes_from_list(self) -> None:
        shared = SharedMemoryGraph()
        shared.register_agent("alice")
        shared.unregister_agent("alice")
        assert "alice" not in shared.list_agents()

    def test_unregister_nonexistent_is_noop(self) -> None:
        shared = SharedMemoryGraph()
        shared.unregister_agent("ghost")  # should not raise


class TestStoreAndRetrieve:
    def test_store_and_retrieve_own(self) -> None:
        shared = SharedMemoryGraph()
        alice = shared.register_agent("alice")
        alice.store_episode("Discovered new pattern in data")
        results = alice.retrieve("pattern")
        assert len(results) == 1
        assert "pattern" in results[0].content

    def test_retrieve_include_shared(self) -> None:
        shared = SharedMemoryGraph()
        alice = shared.register_agent("alice")
        bob = shared.register_agent("bob")
        alice.store_episode("Alice discovered a pattern")
        results = bob.retrieve("pattern", include_shared=True)
        assert len(results) == 1

    def test_retrieve_exclude_shared(self) -> None:
        shared = SharedMemoryGraph()
        alice = shared.register_agent("alice")
        bob = shared.register_agent("bob")
        alice.store_episode("Alice discovered a pattern")
        results = bob.retrieve("pattern", include_shared=False)
        assert len(results) == 0

    def test_retrieve_top_k(self) -> None:
        shared = SharedMemoryGraph()
        alice = shared.register_agent("alice")
        for i in range(5):
            alice.store_episode(f"pattern number {i}")
        results = alice.retrieve("pattern", top_k=3)
        assert len(results) == 3

    def test_store_custom_node(self) -> None:
        shared = SharedMemoryGraph()
        alice = shared.register_agent("alice")
        node = MemoryNode(id="custom1", type=NodeType.FACT, content="custom fact")
        node_id = alice.store(node)
        assert node_id == "custom1"
        results = alice.retrieve("custom")
        assert len(results) == 1


class TestShareWith:
    def test_share_with_another_agent(self) -> None:
        shared = SharedMemoryGraph()
        alice = shared.register_agent("alice")
        bob = shared.register_agent("bob")
        nid = alice.store_episode("Important discovery about patterns")
        alice.share_with(nid, "bob")
        # Bob should now see it in own namespace
        results = bob.retrieve("pattern", include_shared=False)
        assert len(results) == 1

    def test_share_with_unregistered_raises(self) -> None:
        shared = SharedMemoryGraph()
        alice = shared.register_agent("alice")
        nid = alice.store_episode("Some content")
        try:
            alice.share_with(nid, "unknown_agent")
            raise AssertionError("Expected ValueError")
        except ValueError:
            pass


class TestGetSharedFrom:
    def test_get_shared_from(self) -> None:
        shared = SharedMemoryGraph()
        alice = shared.register_agent("alice")
        bob = shared.register_agent("bob")
        nid = alice.store_episode("Shared pattern discovery")
        alice.share_with(nid, "bob")
        shared_nodes = bob.get_shared_from("alice")
        assert len(shared_nodes) == 1
        assert "pattern" in shared_nodes[0].content

    def test_get_shared_from_no_overlap(self) -> None:
        shared = SharedMemoryGraph()
        alice = shared.register_agent("alice")
        bob = shared.register_agent("bob")
        alice.store_episode("Alice private memory")
        shared_nodes = bob.get_shared_from("alice")
        assert len(shared_nodes) == 0


class TestQueryShared:
    def test_query_shared_all(self) -> None:
        shared = SharedMemoryGraph()
        alice = shared.register_agent("alice")
        bob = shared.register_agent("bob")
        alice.store_episode("Pattern alpha discovery")
        bob.store_episode("Pattern beta analysis")
        results = shared.query_shared("pattern")
        assert len(results) == 2

    def test_query_shared_exclude_agent(self) -> None:
        shared = SharedMemoryGraph()
        alice = shared.register_agent("alice")
        bob = shared.register_agent("bob")
        alice.store_episode("Pattern alpha discovery")
        bob.store_episode("Pattern beta analysis")
        results = shared.query_shared("pattern", exclude_agent="alice")
        assert len(results) == 1
        assert "beta" in results[0].content


class TestListAgents:
    def test_list_agents(self) -> None:
        shared = SharedMemoryGraph()
        shared.register_agent("alice")
        shared.register_agent("bob")
        shared.register_agent("charlie")
        agents = shared.list_agents()
        assert set(agents) == {"alice", "bob", "charlie"}

    def test_list_agents_empty(self) -> None:
        shared = SharedMemoryGraph()
        assert shared.list_agents() == []


class TestStats:
    def test_stats(self) -> None:
        shared = SharedMemoryGraph()
        alice = shared.register_agent("alice")
        bob = shared.register_agent("bob")
        alice.store_episode("Alice memory one")
        alice.store_episode("Alice memory two")
        bob.store_episode("Bob memory one")
        s = shared.stats()
        assert s["nodes"] == 3
        assert s["agents"] == 2
        assert s["namespaces"]["alice"] == 2
        assert s["namespaces"]["bob"] == 1

    def test_stats_empty(self) -> None:
        shared = SharedMemoryGraph()
        s = shared.stats()
        assert s["nodes"] == 0
        assert s["agents"] == 0
        assert s["namespaces"] == {}


class TestLink:
    def test_link_creates_edge(self) -> None:
        shared = SharedMemoryGraph()
        alice = shared.register_agent("alice")
        nid1 = alice.store(create_fact("Python is great"))
        nid2 = alice.store(create_fact("Python 3.12 released"))
        alice.link(nid1, nid2, EdgeType.ASSOCIATIVE, weight=0.8)
        s = shared.stats()
        assert s["edges"] == 1

    def test_link_between_agents(self) -> None:
        shared = SharedMemoryGraph()
        alice = shared.register_agent("alice")
        bob = shared.register_agent("bob")
        nid1 = alice.store(create_fact("Source fact"))
        nid2 = bob.store(create_fact("Target fact"))
        alice.link(nid1, nid2, EdgeType.DERIVED_FROM, weight=0.9)
        s = shared.stats()
        assert s["edges"] == 1


class TestThreadSafety:
    def test_concurrent_store(self) -> None:
        shared = SharedMemoryGraph()
        agents = [shared.register_agent(f"agent_{i}") for i in range(5)]
        errors: list[Exception] = []

        def store_many(agent: AgentMemoryHandle, count: int) -> None:
            try:
                for j in range(count):
                    agent.store_episode(f"Memory from {agent.agent_id} item {j}")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=store_many, args=(a, 20)) for a in agents]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        s = shared.stats()
        assert s["nodes"] == 100  # 5 agents * 20 items
        assert s["agents"] == 5
