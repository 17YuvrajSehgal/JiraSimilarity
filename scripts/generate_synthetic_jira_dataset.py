from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from dataclasses import dataclass
from pathlib import Path
import random


@dataclass(frozen=True, slots=True)
class ScenarioTemplate:
    name: str
    project_key: str
    issue_type: str
    component: str
    base_title: str
    base_description: str
    duplicate_title_variants: tuple[str, ...]
    duplicate_description_variants: tuple[str, ...]
    related_title_variants: tuple[str, ...]
    related_description_variants: tuple[str, ...]
    hard_negative_title_variants: tuple[str, ...]
    hard_negative_description_variants: tuple[str, ...]


def _scenario_templates() -> list[ScenarioTemplate]:
    return [
        ScenarioTemplate(
            name="checkout-null-pointer",
            project_key="PAY",
            issue_type="Bug",
            component="payments",
            base_title="Checkout throws null pointer when billing address is missing",
            base_description=(
                "Order placement crashes in payment service when customer profile has no billing address. "
                "Observed in web checkout after selecting saved card."
            ),
            duplicate_title_variants=(
                "Payment checkout crashes with null pointer for empty billing address",
                "Null pointer in payment service when address is blank at checkout",
            ),
            duplicate_description_variants=(
                "The checkout flow fails with a null pointer if billing address is absent in profile data.",
                "Checkout fails at payment confirmation when customer has no stored billing address.",
            ),
            related_title_variants=(
                "Checkout fails tax calculation when billing address region is unknown",
                "Payment service rejects orders when address country code is invalid",
            ),
            related_description_variants=(
                "Not always a crash, but address parsing fails and checkout returns error before payment capture.",
                "Address validation in payment pre-check fails for incomplete region metadata.",
            ),
            hard_negative_title_variants=(
                "Add country-specific invoice formatting for enterprise checkout",
                "Support address autocomplete in checkout payment form",
            ),
            hard_negative_description_variants=(
                "Feature request to improve invoice line item formatting by locale. No crash involved.",
                "Enhancement ticket for UX improvements in payment address field suggestions.",
            ),
        ),
        ScenarioTemplate(
            name="auth-token-refresh",
            project_key="AUTH",
            issue_type="Bug",
            component="authentication",
            base_title="Session expires repeatedly after token refresh",
            base_description=(
                "Users are redirected to login every few minutes even though refresh token exchange succeeds. "
                "Seen in both browser and API clients."
            ),
            duplicate_title_variants=(
                "Users forced to re-login despite successful refresh token flow",
                "Access token renews but session still drops to login screen",
            ),
            duplicate_description_variants=(
                "Refresh endpoint returns success but client receives unauthorized shortly after.",
                "Session persistence breaks after refresh cycle and user authentication state is lost.",
            ),
            related_title_variants=(
                "Refresh token endpoint latency spikes under heavy traffic",
                "OAuth callback intermittently misses state parameter",
            ),
            related_description_variants=(
                "Could be connected to refresh handling because failures increase during timeout windows.",
                "Authentication flow has related reliability issue around callback validation.",
            ),
            hard_negative_title_variants=(
                "Update login page branding assets for new release",
                "Document SSO onboarding guide for partner tenants",
            ),
            hard_negative_description_variants=(
                "UI design refresh task for login page typography and icon usage.",
                "Documentation-only change for SSO integration setup steps.",
            ),
        ),
        ScenarioTemplate(
            name="search-stale-results",
            project_key="SRCH",
            issue_type="Bug",
            component="search",
            base_title="Search index returns stale issues after status transition",
            base_description=(
                "Closed issues still appear as open for several minutes in global search results. "
                "Reindex job appears delayed."
            ),
            duplicate_title_variants=(
                "Global search shows outdated issue status after workflow update",
                "Closed tickets remain visible as open in search index",
            ),
            duplicate_description_variants=(
                "Indexing lag causes stale workflow fields in search results after transitions.",
                "Search cache retains old status values after issue state changes.",
            ),
            related_title_variants=(
                "Incremental indexer misses issue labels during bulk edits",
                "Search relevance drops after analyzer dictionary update",
            ),
            related_description_variants=(
                "Indexer quality issue likely in same ingestion pipeline, but symptom is missing labels.",
                "Search pipeline behaves differently after analyzer release and ranking shifted unexpectedly.",
            ),
            hard_negative_title_variants=(
                "Add quick filters to search UI sidebar",
                "Improve dark theme contrast in search result cards",
            ),
            hard_negative_description_variants=(
                "Feature request for UX convenience filters in query panel.",
                "Visual accessibility ticket unrelated to index freshness.",
            ),
        ),
        ScenarioTemplate(
            name="upload-timeout",
            project_key="DOC",
            issue_type="Bug",
            component="attachments",
            base_title="Large file upload times out before virus scan completes",
            base_description=(
                "Uploads over 200MB fail with timeout while antivirus scan queue is backlogged. "
                "User receives generic network failure."
            ),
            duplicate_title_variants=(
                "Attachment upload fails for big files due to scan queue timeout",
                "Timeout while uploading large documents during antivirus processing",
            ),
            duplicate_description_variants=(
                "Scan service latency causes client upload operation to exceed timeout threshold.",
                "Large upload requests expire before security scan callback returns.",
            ),
            related_title_variants=(
                "Attachment service memory spikes during concurrent uploads",
                "Virus scanning worker retries increase for encrypted archives",
            ),
            related_description_variants=(
                "Likely same subsystem; high memory pressure reduces attachment throughput.",
                "Security scanner has reliability issue that may contribute to delayed processing.",
            ),
            hard_negative_title_variants=(
                "Support drag-and-drop reordering for attachments",
                "Add attachment preview zoom controls",
            ),
            hard_negative_description_variants=(
                "Product enhancement for attachment list ordering in UI.",
                "Usability enhancement for preview pane controls.",
            ),
        ),
        ScenarioTemplate(
            name="mobile-offline-sync",
            project_key="MOB",
            issue_type="Bug",
            component="mobile-sync",
            base_title="Mobile app creates duplicate comments after offline sync",
            base_description=(
                "When device reconnects, queued comments are posted twice in some threads. "
                "Conflict resolver appears to replay the same operation."
            ),
            duplicate_title_variants=(
                "Offline sync posts same comment twice after reconnect",
                "Duplicate mobile comments created when network returns",
            ),
            duplicate_description_variants=(
                "Sync queue de-duplication fails and comment events are replayed after reconnect.",
                "Reconnect workflow re-submits comment payloads causing duplicates.",
            ),
            related_title_variants=(
                "Offline sync drops image attachments on reconnect",
                "Conflict resolver marks unchanged fields as modified",
            ),
            related_description_variants=(
                "Related sync reliability issue in the same reconciliation path.",
                "State merge issue in mobile sync leads to unnecessary update patches.",
            ),
            hard_negative_title_variants=(
                "Add haptic feedback for comment submission",
                "Update mobile typography scale for accessibility",
            ),
            hard_negative_description_variants=(
                "UI/UX enhancement request unrelated to sync correctness.",
                "Accessibility improvements for font scaling in mobile views.",
            ),
        ),
        ScenarioTemplate(
            name="deployment-rollback",
            project_key="OPS",
            issue_type="Incident",
            component="deployments",
            base_title="Automated rollback fails after partial canary deployment",
            base_description=(
                "Rollback pipeline marks deployment successful but traffic still routes to unhealthy pods. "
                "Manual intervention required."
            ),
            duplicate_title_variants=(
                "Canary rollback reports success while unhealthy version stays live",
                "Rollback automation does not fully revert failed canary release",
            ),
            duplicate_description_variants=(
                "Deployment controller exits early, leaving a subset of traffic on failed revision.",
                "Rollback status is incorrect and old replicas are not restored correctly.",
            ),
            related_title_variants=(
                "Deployment monitor alerts delayed during canary failures",
                "Traffic router keeps stale route weights after rollback",
            ),
            related_description_variants=(
                "Monitoring lag hides failed rollbacks and likely relates to deployment controller behavior.",
                "Route table update issue in the same release management flow.",
            ),
            hard_negative_title_variants=(
                "Add deployment timeline export for compliance reporting",
                "Refactor release notes template for ops documentation",
            ),
            hard_negative_description_variants=(
                "Reporting enhancement for release audit trail.",
                "Documentation task for release summary templates.",
            ),
        ),
    ]


def _pick(rng: random.Random, values: tuple[str, ...]) -> str:
    return values[rng.randrange(len(values))]


def _format_ts(value: datetime) -> str:
    return value.strftime("%Y-%m-%dT%H:%M:%S.000%z")


def _comments_blob(comments: tuple[str, ...]) -> str:
    return "\n".join(comments)


def _build_history(
    *,
    issue_key: str,
    reporter: str,
    assignee: str,
    base_time: datetime,
    status: str,
    resolution: str | None,
) -> list[dict[str, object]]:
    history: list[dict[str, object]] = [
        {
            "id": f"{issue_key}-h1",
            "author": reporter,
            "created": _format_ts(base_time + timedelta(minutes=5)),
            "items": [
                {
                    "field": "status",
                    "fieldtype": "jira",
                    "from": "Needs Triage",
                    "to": "Open",
                    "from_id": None,
                    "to_id": "1",
                }
            ],
        }
    ]
    if status in {"In Progress", "Resolved"}:
        history.append(
            {
                "id": f"{issue_key}-h2",
                "author": assignee,
                "created": _format_ts(base_time + timedelta(hours=6)),
                "items": [
                    {
                        "field": "status",
                        "fieldtype": "jira",
                        "from": "Open",
                        "to": "In Progress",
                        "from_id": "1",
                        "to_id": "3",
                    }
                ],
            }
        )
    if status == "Resolved":
        history.append(
            {
                "id": f"{issue_key}-h3",
                "author": assignee,
                "created": _format_ts(base_time + timedelta(days=2)),
                "items": [
                    {
                        "field": "resolution",
                        "fieldtype": "jira",
                        "from": None,
                        "to": resolution or "Fixed",
                        "from_id": None,
                        "to_id": "1",
                    },
                    {
                        "field": "status",
                        "fieldtype": "jira",
                        "from": "In Progress",
                        "to": "Resolved",
                        "from_id": "3",
                        "to_id": "5",
                    },
                ],
            }
        )
    return history


def _build_activity(
    *,
    issue_key: str,
    history: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for entry in history:
        created = str(entry["created"])
        date_key = created[:10]
        events = grouped.setdefault(date_key, [])
        for item in entry["items"]:
            events.append(
                {
                    "type": "history",
                    "id": entry["id"],
                    "author": entry["author"],
                    "created": created,
                    "timestamp": created,
                    "field": item["field"],
                    "from": item["from"],
                    "to": item["to"],
                    "description": (
                        f"{entry['author']} changed {item['field']} "
                        f"from {item['from']} to {item['to']} on {issue_key}"
                    ),
                }
            )
    return [{"date": day, "events": events} for day, events in sorted(grouped.items())]


def _build_issue(
    *,
    issue_id: int,
    issue_key: str,
    project_key: str,
    issue_type: str,
    priority: str,
    status: str,
    resolution: str | None,
    component: str,
    title: str,
    description: str,
    affected_versions: tuple[str, ...],
    fix_versions: tuple[str, ...],
    comments: tuple[str, ...],
    semantic_cluster_id: str,
    role: str,
    reporter: str,
    assignee: str,
    created_at: datetime,
    rng: random.Random,
) -> dict[str, object]:
    updated_at = created_at + timedelta(days=2)
    resolved_at = updated_at if status == "Resolved" else None
    history = _build_history(
        issue_key=issue_key,
        reporter=reporter,
        assignee=assignee,
        base_time=created_at,
        status=status,
        resolution=resolution,
    )
    activity = _build_activity(issue_key=issue_key, history=history)
    comment_ids = [f"{issue_id}{index:02d}" for index, _ in enumerate(comments, start=1)]
    return {
        "jira_id": issue_key,
        # Backward-compatible flat fields for older JSON adapters.
        "issue_id": issue_id,
        "issue_key": issue_key,
        "project_key": project_key,
        "title": title,
        "description_text": description,
        "issue_type": issue_type,
        "priority": priority,
        "status": status,
        "resolution": resolution,
        "components": [component],
        "affected_versions": list(affected_versions),
        "fix_versions": list(fix_versions),
        "comments": list(comments),
        "linked_issue_ids": [],
        "duplicate_issue_ids": [],
        "metadata": {
            "issue_id": issue_id,
            "summary": title,
            "project_id": str(18000 + (issue_id % 1000)),
            "project_key": project_key,
            "project_name": f"{project_key} Synthetic Engineering",
            "issue_type": issue_type,
            "status": status,
            "priority": priority,
            "affects_versions": list(affected_versions),
            "components": [component],
            "labels": [f"cluster-{semantic_cluster_id.split('-')[-1]}", role],
            "design_category": None,
            "found_in_load": None,
            "epic_link": None,
            "backlog_status": None,
            "resolution": resolution,
            "fix_versions": list(fix_versions),
            "description": description,
            "attachments": [],
            "assignee": assignee,
            "reporter": reporter,
            "watcher_count": rng.randint(1, 6),
            "created_at": _format_ts(created_at),
            "updated_at": _format_ts(updated_at),
            "resolved_at": _format_ts(resolved_at) if resolved_at else None,
            "development": {
                "commit_count": rng.randint(0, 8),
                "pr_count": rng.randint(0, 3),
                "branch_count": rng.randint(0, 2),
                "repository_count": rng.randint(1, 4),
                "last_updated": _format_ts(updated_at),
            },
            "sprints": [],
            "related_issues": [],
            "duplicate_issues": [],
            "comments_id": comment_ids,
            "comments_body": _comments_blob(comments),
            "worklog": [],
            "history": history,
            "activity": activity,
            "submissions": [],
            "synthetic_profile": {
                "semantic_cluster_id": semantic_cluster_id,
                "role": role,
            },
        },
    }


def build_synthetic_dataset(*, cluster_count: int, random_seed: int) -> dict[str, object]:
    if cluster_count < 1:
        raise ValueError("cluster_count must be >= 1")

    rng = random.Random(random_seed)
    templates = _scenario_templates()
    issues: list[dict[str, object]] = []
    pair_labels: list[dict[str, object]] = []
    related_issue_ids: list[int] = []
    semantic_cluster_by_issue: dict[int, str] = {}
    issue_key_by_id: dict[int, str] = {}
    people = (
        ("Alex Wong", "Priya Iyer"),
        ("Jordan Lee", "Marta Silva"),
        ("Nina Patel", "Chris Bennett"),
        ("Samir Khan", "Dana Ortiz"),
    )

    issue_id = 10000
    release_cycle = ("2026.1", "2026.2", "2026.3", "2026.4")
    priorities = ("Highest", "High", "Medium", "Low")

    def next_issue_key(project_key: str, iid: int) -> str:
        return f"{project_key}-{iid}"

    for cluster_idx in range(cluster_count):
        template = templates[cluster_idx % len(templates)]
        semantic_cluster_id = f"{template.name}-{cluster_idx + 1:03d}"
        signature_token = f"sig_{template.name.replace('-', '_')}_{cluster_idx + 1:03d}"
        affected_version = release_cycle[cluster_idx % len(release_cycle)]
        next_version = release_cycle[(cluster_idx + 1) % len(release_cycle)]
        reporter, assignee = people[cluster_idx % len(people)]
        created_base = datetime(2025, 1, 1, 9, 0, tzinfo=timezone.utc) + timedelta(days=cluster_idx)

        anchor_id = issue_id
        anchor_key = next_issue_key(template.project_key, anchor_id)
        anchor = _build_issue(
            issue_id=anchor_id,
            issue_key=anchor_key,
            project_key=template.project_key,
            issue_type=template.issue_type,
            priority=priorities[cluster_idx % 2],
            status="Open",
            resolution=None,
            component=template.component,
            title=template.base_title,
            description=f"{template.base_description} Error signature: {signature_token}.",
            affected_versions=(affected_version,),
            fix_versions=(next_version,),
            comments=(
                "Initial report from QA with reproducible steps.",
                "Crash observed in staging and production-like environment.",
            ),
            semantic_cluster_id=semantic_cluster_id,
            role="anchor",
            reporter=reporter,
            assignee=assignee,
            created_at=created_base,
            rng=rng,
        )
        issue_id += 1

        duplicate_id = issue_id
        duplicate_key = next_issue_key(template.project_key, duplicate_id)
        duplicate = _build_issue(
            issue_id=duplicate_id,
            issue_key=duplicate_key,
            project_key=template.project_key,
            issue_type=template.issue_type,
            priority=priorities[(cluster_idx + 1) % 3],
            status="In Progress",
            resolution=None,
            component=template.component,
            title=_pick(rng, template.duplicate_title_variants),
            description=f"{_pick(rng, template.duplicate_description_variants)} Error signature: {signature_token}.",
            affected_versions=(affected_version,),
            fix_versions=(next_version,),
            comments=(
                "Likely duplicate reported by support channel.",
                "Engineer confirms matching logs with existing incident.",
            ),
            semantic_cluster_id=semantic_cluster_id,
            role="duplicate-paraphrase",
            reporter=reporter,
            assignee=assignee,
            created_at=created_base + timedelta(hours=2),
            rng=rng,
        )
        issue_id += 1

        related_id = issue_id
        related_key = next_issue_key(template.project_key, related_id)
        related = _build_issue(
            issue_id=related_id,
            issue_key=related_key,
            project_key=template.project_key,
            issue_type=template.issue_type,
            priority="Medium",
            status="Open",
            resolution=None,
            component=template.component,
            title=_pick(rng, template.related_title_variants),
            description=f"{_pick(rng, template.related_description_variants)} Diagnostic tag: {signature_token}.",
            affected_versions=(affected_version,),
            fix_versions=(next_version,),
            comments=("May share root cause but not confirmed duplicate.",),
            semantic_cluster_id=semantic_cluster_id,
            role="related-medium-similarity",
            reporter=reporter,
            assignee=assignee,
            created_at=created_base + timedelta(hours=4),
            rng=rng,
        )
        issue_id += 1

        hard_negative_id = issue_id
        hard_negative_key = next_issue_key(template.project_key, hard_negative_id)
        hard_negative = _build_issue(
            issue_id=hard_negative_id,
            issue_key=hard_negative_key,
            project_key=template.project_key,
            issue_type=("Task" if template.issue_type != "Task" else "Story"),
            priority="Low",
            status="To Do",
            resolution=None,
            component=template.component,
            title=_pick(rng, template.hard_negative_title_variants),
            description=_pick(rng, template.hard_negative_description_variants),
            affected_versions=(affected_version,),
            fix_versions=(),
            comments=("Backlog item with lexical overlap but different intent.",),
            semantic_cluster_id=semantic_cluster_id,
            role="hard-negative-low-similarity",
            reporter=reporter,
            assignee=assignee,
            created_at=created_base + timedelta(days=1),
            rng=rng,
        )
        issue_id += 1

        cross_project = "PLAT" if template.project_key != "PLAT" else "CORE"
        cross_project_id = issue_id
        cross_project_key = next_issue_key(cross_project, cross_project_id)
        cross_project_issue = _build_issue(
            issue_id=cross_project_id,
            issue_key=cross_project_key,
            project_key=cross_project,
            issue_type="Bug",
            priority="Medium",
            status="Open",
            resolution=None,
            component="platform",
            title=f"Investigate {template.component} telemetry inconsistency in shared services",
            description=(
                "Cross-project observability issue with similar terms but different subsystem ownership. "
                "Should typically score lower than true duplicates."
            ),
            affected_versions=(affected_version,),
            fix_versions=(next_version,),
            comments=("Potentially related by wording only; not same defect.",),
            semantic_cluster_id=f"cross-{cluster_idx + 1:03d}",
            role="cross-project-low-similarity",
            reporter=reporter,
            assignee=assignee,
            created_at=created_base + timedelta(days=1, hours=6),
            rng=rng,
        )
        issue_id += 1

        anchor["metadata"]["duplicate_issues"] = [duplicate_key]
        duplicate["metadata"]["duplicate_issues"] = [anchor_key]
        anchor["metadata"]["related_issues"] = [duplicate_key, related_key]
        duplicate["metadata"]["related_issues"] = [anchor_key, related_key]
        related["metadata"]["related_issues"] = [anchor_key, duplicate_key]
        anchor["duplicate_issue_ids"] = [duplicate_id]
        duplicate["duplicate_issue_ids"] = [anchor_id]
        anchor["linked_issue_ids"] = [duplicate_id, related_id]
        duplicate["linked_issue_ids"] = [anchor_id, related_id]
        related["linked_issue_ids"] = [anchor_id, duplicate_id]

        if related_issue_ids and rng.random() < 0.35:
            bridge_target = related_issue_ids[rng.randrange(len(related_issue_ids))]
            related_links = related["metadata"]["related_issues"]
            if isinstance(related_links, list):
                related_links.append(issue_key_by_id.get(bridge_target, f"SYN-{bridge_target}"))
            related_link_ids = related["linked_issue_ids"]
            if isinstance(related_link_ids, list):
                related_link_ids.append(bridge_target)

        issues.extend([anchor, duplicate, related, hard_negative, cross_project_issue])

        for issue in (anchor, duplicate, related, hard_negative, cross_project_issue):
            issue_metadata = issue["metadata"]
            issue_key_by_id[int(issue_metadata["issue_id"])] = str(issue["jira_id"])
            semantic_cluster_by_issue[int(issue_metadata["issue_id"])] = str(
                issue_metadata["synthetic_profile"]["semantic_cluster_id"]
            )

        related_issue_ids.append(related_id)

        pair_labels.append(
            {
                "left_issue_id": anchor_id,
                "right_issue_id": duplicate_id,
                "left_jira_id": anchor_key,
                "right_jira_id": duplicate_key,
                "label": 1,
                "relationship_type": "duplicate",
                "similarity_band": "high",
            }
        )
        pair_labels.append(
            {
                "left_issue_id": anchor_id,
                "right_issue_id": related_id,
                "left_jira_id": anchor_key,
                "right_jira_id": related_key,
                "label": 1,
                "relationship_type": "linked",
                "similarity_band": "medium",
            }
        )
        pair_labels.append(
            {
                "left_issue_id": anchor_id,
                "right_issue_id": hard_negative_id,
                "left_jira_id": anchor_key,
                "right_jira_id": hard_negative_key,
                "label": 0,
                "relationship_type": "hard_negative",
                "similarity_band": "low",
            }
        )

    all_issue_ids = [int(issue["metadata"]["issue_id"]) for issue in issues]
    negative_target_count = min(len(pair_labels), max(10, cluster_count * 2))
    negative_count = 0
    seen_pairs: set[tuple[int, int]] = set()

    while negative_count < negative_target_count:
        left = all_issue_ids[rng.randrange(len(all_issue_ids))]
        right = all_issue_ids[rng.randrange(len(all_issue_ids))]
        if left == right:
            continue
        pair = (min(left, right), max(left, right))
        if pair in seen_pairs:
            continue
        if semantic_cluster_by_issue[left] == semantic_cluster_by_issue[right]:
            continue
        seen_pairs.add(pair)
        pair_labels.append(
            {
                "left_issue_id": pair[0],
                "right_issue_id": pair[1],
                "left_jira_id": issue_key_by_id.get(pair[0], f"SYN-{pair[0]}"),
                "right_jira_id": issue_key_by_id.get(pair[1], f"SYN-{pair[1]}"),
                "label": 0,
                "relationship_type": "random_negative",
                "similarity_band": "low",
            }
        )
        negative_count += 1

    duplicate_edges = sum(len(issue["metadata"]["duplicate_issues"]) for issue in issues)
    linked_edges = sum(len(issue["metadata"]["related_issues"]) for issue in issues)

    return {
        "meta": {
            "name": "synthetic-jira-research-dataset",
            "description": (
                "Synthetic Jira-like dataset for similarity and duplicate benchmarking. "
                "Contains high/medium/low similarity cases, paraphrases, hard negatives, and graph links."
            ),
            "generator_version": "1.0",
            "random_seed": random_seed,
            "cluster_count": cluster_count,
            "issue_count": len(issues),
            "duplicate_edge_count_directed": duplicate_edges,
            "linked_edge_count_directed": linked_edges,
            "research_extensions": [
                "synthetic_profile",
                "pair_labels",
                "metadata.issue_id",
                "metadata.duplicate_issues",
            ],
        },
        "issues": issues,
        "pair_labels": pair_labels,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic Jira-style dataset for similarity and duplicate benchmarking."
    )
    parser.add_argument(
        "--output",
        default="datasets/synthetic/synthetic_jira_issues.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--cluster-count",
        type=int,
        default=30,
        help="Number of semantic clusters to generate. Each cluster creates 5 issues.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260421,
        help="Random seed for deterministic generation.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = build_synthetic_dataset(cluster_count=args.cluster_count, random_seed=args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"Generated synthetic dataset at {output_path} "
        f"(issues={payload['meta']['issue_count']}, clusters={payload['meta']['cluster_count']})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
