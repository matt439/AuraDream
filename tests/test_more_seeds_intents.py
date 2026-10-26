"""More seeds receipts distinguish a retry from a deliberate new extension."""
from __future__ import annotations

import asyncio

import pytest

from backend.more_seeds_intents import IntentLedger, IntentProblem


CONTEXT = ("server-a", 123, "checkpoint", "fp8")
PAYLOAD = {"ids": [1], "count": 4, "batch": 2, "expected": [1]}


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


async def wait_for(event: asyncio.Event) -> None:
    await asyncio.wait_for(event.wait(), timeout=2)


def test_concurrent_retries_share_one_operation_and_return_independent_receipts():
    async def scenario():
        ledger = IntentLedger()
        token = ledger.issue(PAYLOAD, CONTEXT)
        started, release = asyncio.Event(), asyncio.Event()
        calls = 0

        async def operation():
            nonlocal calls
            calls += 1
            started.set()
            await release.wait()
            return {"job_id": "one", "more": [{"job_id": "two"}], "queued": 4}

        first = asyncio.create_task(ledger.execute(token, PAYLOAD, CONTEXT, operation))
        second = None
        try:
            await wait_for(started)
            second = asyncio.create_task(ledger.execute(token, PAYLOAD, CONTEXT, operation))
            await asyncio.sleep(0)
            assert calls == 1 and ledger.pending
            assert not first.done() and not second.done()
            release.set()
            a, b = await asyncio.wait_for(asyncio.gather(first, second), timeout=2)
            assert a == b == {"job_id": "one", "more": [{"job_id": "two"}], "queued": 4}
            a["more"][0]["job_id"] = "changed by the first caller"
            assert b["more"][0]["job_id"] == "two"
            replay = await ledger.execute(token, dict(reversed(list(PAYLOAD.items()))), CONTEXT, operation)
            assert replay == b and replay is not b
            replay["more"].clear()
            assert (await ledger.execute(token, PAYLOAD, CONTEXT, operation))["more"] == [{"job_id": "two"}]
            assert calls == 1 and not ledger.pending
        finally:
            release.set()
            await asyncio.gather(*[task for task in (first, second) if task is not None], return_exceptions=True)
            await ledger.close()

    asyncio.run(scenario())


def test_distinct_tokens_are_distinct_intents_even_when_the_payloads_match():
    async def scenario():
        ledger = IntentLedger()
        first = ledger.issue(PAYLOAD, CONTEXT)
        second = ledger.issue(PAYLOAD, CONTEXT)
        entered, release = asyncio.Event(), asyncio.Event()
        calls = 0

        async def operation():
            nonlocal calls
            calls += 1
            number = calls
            if calls == 2:
                entered.set()
            await release.wait()
            return {"job_id": f"job-{number}"}

        tasks = [
            asyncio.create_task(ledger.execute(token, PAYLOAD, CONTEXT, operation))
            for token in (first, second)
        ]
        try:
            assert first != second
            await wait_for(entered)
            assert ledger.pending
            release.set()
            answers = await asyncio.wait_for(asyncio.gather(*tasks), timeout=2)
            assert {answer["job_id"] for answer in answers} == {"job-1", "job-2"}
            assert calls == 2 and not ledger.pending
        finally:
            release.set()
            await asyncio.gather(*tasks, return_exceptions=True)
            await ledger.close()

    asyncio.run(scenario())


def test_token_binds_a_payload_snapshot_and_context_before_and_after_acceptance():
    async def scenario():
        ledger = IntentLedger()
        payload = {"ids": [1], "count": 4, "expected": [1]}
        token = ledger.issue(payload, CONTEXT)
        payload["ids"].append(2)
        original = {"expected": [1], "count": 4, "ids": [1]}
        calls = 0

        async def operation():
            nonlocal calls
            calls += 1
            return {"job_id": "original"}

        try:
            for body, context in ((payload, CONTEXT), (original, ("server-b", *CONTEXT[1:]))):
                with pytest.raises(IntentProblem) as refused:
                    await ledger.execute(token, body, context, operation)
                assert refused.value.status_code == 409 and refused.value.detail
            assert calls == 0 and not ledger.pending
            assert await ledger.execute(token, original, CONTEXT, operation) == {"job_id": "original"}
            for body, context in ((payload, CONTEXT), (original, (CONTEXT[0], 124, *CONTEXT[2:]))):
                with pytest.raises(IntentProblem) as refused:
                    await ledger.execute(token, body, context, operation)
                assert refused.value.status_code == 409
            assert calls == 1
        finally:
            await ledger.close()

    asyncio.run(scenario())


def test_unknown_expired_and_another_managers_tokens_never_start_an_operation():
    async def scenario():
        clock = Clock()
        ledger = IntentLedger(clock=clock, issued_seconds=10, receipt_seconds=20)
        reset = IntentLedger(clock=clock)
        token = ledger.issue(PAYLOAD, CONTEXT)
        calls = 0

        async def operation():
            nonlocal calls
            calls += 1
            return {"job_id": "unexpected"}

        try:
            for owner, candidate in ((ledger, "unknown"), (reset, token)):
                with pytest.raises(IntentProblem) as refused:
                    await owner.execute(candidate, PAYLOAD, CONTEXT, operation)
                assert refused.value.status_code == 409
            clock.advance(11)
            with pytest.raises(IntentProblem) as refused:
                await ledger.execute(token, PAYLOAD, CONTEXT, operation)
            assert refused.value.status_code == 409
            assert calls == 0 and not ledger.pending and not reset.pending
        finally:
            await ledger.close()
            await reset.close()

    asyncio.run(scenario())


def test_receipt_lifetime_starts_at_settlement_and_running_entries_do_not_expire():
    async def scenario():
        clock = Clock()
        ledger = IntentLedger(clock=clock, issued_seconds=10, receipt_seconds=20)
        token = ledger.issue(PAYLOAD, CONTEXT)
        started, release = asyncio.Event(), asyncio.Event()
        calls = 0

        async def operation():
            nonlocal calls
            calls += 1
            started.set()
            await release.wait()
            return {"job_id": "long-operation"}

        task = asyncio.create_task(ledger.execute(token, PAYLOAD, CONTEXT, operation))
        replay = None
        try:
            await wait_for(started)
            clock.advance(100)
            replay = asyncio.create_task(ledger.execute(token, PAYLOAD, CONTEXT, operation))
            await asyncio.sleep(0)
            assert calls == 1 and ledger.pending
            release.set()
            await asyncio.wait_for(asyncio.gather(task, replay), timeout=2)
            clock.advance(19)
            assert await ledger.execute(token, PAYLOAD, CONTEXT, operation) == {"job_id": "long-operation"}
            clock.advance(2)
            with pytest.raises(IntentProblem) as refused:
                await ledger.execute(token, PAYLOAD, CONTEXT, operation)
            assert refused.value.status_code == 409
            assert calls == 1 and not ledger.pending
        finally:
            release.set()
            await asyncio.gather(*[item for item in (task, replay) if item is not None], return_exceptions=True)
            await ledger.close()

    asyncio.run(scenario())


def test_settlement_keeps_a_receipt_before_task_done_callbacks_run():
    async def scenario():
        clock = Clock()
        ledger = IntentLedger(clock=clock, capacity=1, issued_seconds=10, receipt_seconds=20)
        token = ledger.issue(PAYLOAD, CONTEXT)
        started, release = asyncio.Event(), asyncio.Event()
        loop = asyncio.get_running_loop()
        observed = loop.create_future()
        calls = 0

        def issue_before_done_callbacks():
            try:
                replacement = ledger.issue(PAYLOAD, CONTEXT)
            except IntentProblem as exc:
                observed.set_result(("refused", exc.status_code))
            except Exception as exc:
                observed.set_exception(exc)
            else:
                observed.set_result(("issued", replacement))

        async def operation():
            nonlocal calls
            calls += 1
            started.set()
            await release.wait()
            # This callback is queued before completing the task queues its
            # done callbacks. The receipt must already have its new lifetime.
            loop.call_soon(issue_before_done_callbacks)
            return {"job_id": "kept-after-a-slow-admission"}

        task = asyncio.create_task(ledger.execute(token, PAYLOAD, CONTEXT, operation))
        try:
            await wait_for(started)
            clock.advance(100)
            release.set()
            assert await asyncio.wait_for(observed, timeout=2) == ("refused", 503)
            answer = await asyncio.wait_for(task, timeout=2)
            assert await ledger.execute(token, PAYLOAD, CONTEXT, operation) == answer
            assert calls == 1 and not ledger.pending
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            await ledger.close()

    asyncio.run(scenario())


def test_cancelling_one_caller_waits_for_settlement_without_cancelling_the_intent():
    async def scenario():
        ledger = IntentLedger()
        token = ledger.issue(PAYLOAD, CONTEXT)
        started, release = asyncio.Event(), asyncio.Event()
        calls = 0

        async def operation():
            nonlocal calls
            calls += 1
            started.set()
            await release.wait()
            return {"job_id": "accepted-once"}

        caller = asyncio.create_task(ledger.execute(token, PAYLOAD, CONTEXT, operation))
        retry = None
        try:
            await wait_for(started)
            caller.cancel()
            await asyncio.sleep(0)
            assert not caller.done() and ledger.pending
            retry = asyncio.create_task(ledger.execute(token, PAYLOAD, CONTEXT, operation))
            await asyncio.sleep(0)
            assert calls == 1
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(caller, timeout=2)
            assert await asyncio.wait_for(retry, timeout=2) == {"job_id": "accepted-once"}
            assert await ledger.execute(token, PAYLOAD, CONTEXT, operation) == {"job_id": "accepted-once"}
            assert calls == 1 and not ledger.pending
        finally:
            release.set()
            await asyncio.gather(*[task for task in (caller, retry) if task is not None], return_exceptions=True)
            await ledger.close()

    asyncio.run(scenario())


def test_a_failed_intent_replays_its_failure_without_running_again():
    async def scenario():
        ledger = IntentLedger()
        token = ledger.issue(PAYLOAD, CONTEXT)
        calls = 0

        async def operation():
            nonlocal calls
            calls += 1
            raise RuntimeError("the second group could not be admitted")

        try:
            for _ in range(2):
                with pytest.raises(RuntimeError, match="the second group could not be admitted"):
                    await ledger.execute(token, PAYLOAD, CONTEXT, operation)
            assert calls == 1 and not ledger.pending
        finally:
            await ledger.close()

    asyncio.run(scenario())


def test_capacity_evicts_an_unsubmitted_token_instead_of_turning_it_into_a_new_intent():
    async def scenario():
        ledger = IntentLedger(capacity=2)
        oldest = ledger.issue(PAYLOAD, CONTEXT)
        second = ledger.issue(PAYLOAD, CONTEXT)
        newest = ledger.issue(PAYLOAD, CONTEXT)
        calls = 0

        async def operation():
            nonlocal calls
            calls += 1
            return {"job_id": f"job-{calls}"}

        try:
            with pytest.raises(IntentProblem) as refused:
                await ledger.execute(oldest, PAYLOAD, CONTEXT, operation)
            assert refused.value.status_code == 409 and calls == 0
            await ledger.execute(second, PAYLOAD, CONTEXT, operation)
            await ledger.execute(newest, PAYLOAD, CONTEXT, operation)
            assert calls == 2
        finally:
            await ledger.close()

    asyncio.run(scenario())


def test_capacity_preserves_running_work_and_recent_receipts_until_expiry():
    async def scenario():
        clock = Clock()
        ledger = IntentLedger(clock=clock, capacity=1, issued_seconds=10, receipt_seconds=20)
        token = ledger.issue(PAYLOAD, CONTEXT)
        started, release = asyncio.Event(), asyncio.Event()
        calls = 0

        async def operation():
            nonlocal calls
            calls += 1
            started.set()
            await release.wait()
            return {"job_id": "held"}

        task = asyncio.create_task(ledger.execute(token, PAYLOAD, CONTEXT, operation))
        try:
            await wait_for(started)
            clock.advance(100)
            with pytest.raises(IntentProblem) as refused:
                ledger.issue(PAYLOAD, CONTEXT)
            assert refused.value.status_code == 503
            assert ledger.pending and calls == 1
            release.set()
            await asyncio.wait_for(task, timeout=2)
            with pytest.raises(IntentProblem) as refused:
                ledger.issue(PAYLOAD, CONTEXT)
            assert refused.value.status_code == 503
            assert await ledger.execute(token, PAYLOAD, CONTEXT, operation) == {"job_id": "held"}
            clock.advance(21)
            replacement = ledger.issue(PAYLOAD, CONTEXT)
            assert replacement != token
            with pytest.raises(IntentProblem) as refused:
                await ledger.execute(token, PAYLOAD, CONTEXT, operation)
            assert refused.value.status_code == 409
            await ledger.execute(replacement, PAYLOAD, CONTEXT, operation)
            assert calls == 2
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            await ledger.close()

    asyncio.run(scenario())


def test_close_drains_work_refuses_fresh_operations_and_keeps_accepted_replay():
    async def scenario():
        ledger = IntentLedger()
        running = ledger.issue(PAYLOAD, CONTEXT)
        fresh = ledger.issue(PAYLOAD, CONTEXT)
        started, release = asyncio.Event(), asyncio.Event()
        calls = 0

        async def operation():
            nonlocal calls
            calls += 1
            started.set()
            await release.wait()
            return {"job_id": "settled-before-close"}

        task = asyncio.create_task(ledger.execute(running, PAYLOAD, CONTEXT, operation))
        closing = None
        try:
            await wait_for(started)
            closing = asyncio.create_task(ledger.close())
            await asyncio.sleep(0)
            assert not closing.done() and ledger.pending
            with pytest.raises(IntentProblem) as refused:
                ledger.issue(PAYLOAD, CONTEXT)
            assert refused.value.status_code == 503
            with pytest.raises(IntentProblem) as refused:
                await ledger.execute(fresh, PAYLOAD, CONTEXT, operation)
            assert refused.value.status_code == 503 and calls == 1
            release.set()
            answer, _ = await asyncio.wait_for(asyncio.gather(task, closing), timeout=2)
            assert not ledger.pending
            assert await ledger.execute(running, PAYLOAD, CONTEXT, operation) == answer
            with pytest.raises(IntentProblem) as refused:
                ledger.issue(PAYLOAD, CONTEXT)
            assert refused.value.status_code == 503
            await ledger.close()
            assert calls == 1
        finally:
            release.set()
            await asyncio.gather(*[item for item in (task, closing) if item is not None], return_exceptions=True)
            await ledger.close()

    asyncio.run(scenario())
