"""Explicit --db single-line JSON CLI for LSMMemory administration and ops."""

from __future__ import annotations

import argparse
import json
import sys

from .policy import assert_content_safe
from .retrieval import AdaptiveRetriever
from .store import KINDS, RELATION_TYPES, MemoryStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LSMMemory CLI")
    parser.add_argument("--db", required=True, help="SQLite memory database path")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status")

    p_add = sub.add_parser("add")
    p_add.add_argument("--content", required=True)
    p_add.add_argument("--kind", choices=KINDS, default="episode")
    p_add.add_argument("--scope", default="personal")
    p_add.add_argument("--terms", default="", help="Comma-separated aliases/keywords")
    p_add.add_argument("--topic", default="")
    p_add.add_argument("--source-client", default="cli")
    p_add.add_argument("--machine", default="local")
    p_add.add_argument("--session-ref", default="")

    p_search = sub.add_parser("search")
    p_search.add_argument("query")
    p_search.add_argument("--scope", default="personal")
    p_search.add_argument("--limit", type=int, default=5)

    p_sup = sub.add_parser("suppress")
    p_sup.add_argument("id", type=int)

    p_res = sub.add_parser("restore")
    p_res.add_argument("id", type=int)

    p_fgt = sub.add_parser("forget")
    p_fgt.add_argument("id", type=int)
    p_fgt.add_argument("--reason", default="")

    p_rel = sub.add_parser("relate")
    p_rel.add_argument("source_id", type=int)
    p_rel.add_argument("target_id", type=int)
    p_rel.add_argument("--type", choices=RELATION_TYPES, default="related")

    p_cap = sub.add_parser("capture")
    p_cap.add_argument("--content", required=True)
    p_cap.add_argument("--scope", default="personal")
    p_cap.add_argument("--source-client", default="cli")
    p_cap.add_argument("--machine", default="local")

    args = parser.parse_args(argv)
    store = MemoryStore(args.db)

    if args.cmd == "status":
        out = store.meta()
    elif args.cmd == "add":
        terms = [t.strip() for t in args.terms.split(",") if t.strip()]
        assert_content_safe(args.content, terms)
        out = store.add_memory(
            args.content,
            kind=args.kind,
            scope=args.scope,
            search_terms=terms,
            topic=args.topic,
            source_client=args.source_client,
            machine=args.machine,
            session_ref=args.session_ref,
        )
    elif args.cmd == "search":
        retriever = AdaptiveRetriever(store)
        try:
            out = retriever.search(args.query, scope=args.scope, limit=args.limit)
        finally:
            retriever.close()
    elif args.cmd == "suppress":
        out = store.set_suppressed(args.id, True, actor="cli")
    elif args.cmd == "restore":
        out = store.set_suppressed(args.id, False, actor="cli")
    elif args.cmd == "forget":
        out = store.forget_memory(args.id, reason=args.reason, actor="cli")
    elif args.cmd == "relate":
        out = store.add_relation(args.source_id, args.target_id, args.type)
    elif args.cmd == "capture":
        assert_content_safe(args.content)
        out = store.enqueue_inbox(
            args.content,
            scope=args.scope,
            source_client=args.source_client,
            machine=args.machine,
        )
    else:
        raise SystemExit(2)

    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
