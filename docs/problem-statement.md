# Problem Statement

Organizations that use Jira accumulate large volumes of issue reports over time, including bugs, incidents, change requests, feature requests, operational tasks, and support tickets. As this issue history grows, it becomes increasingly difficult for teams to quickly determine whether a newly reported issue is:

- related to a previously reported issue
- functionally similar to an older problem
- an exact or near-exact duplicate of an existing ticket
- part of a recurring pattern across projects, components, versions, or teams

This creates a practical industry problem in issue triage and knowledge reuse.

When a new Jira issue is created, teams often lack an effective way to automatically and reliably search historical issues for the most relevant prior cases. As a result:

- duplicate issues remain open and create unnecessary noise
- engineers spend time investigating problems that may already be known
- important historical context is missed during triage
- similar failures are handled inconsistently across teams
- reporting and prioritization become less accurate because the issue inventory is inflated by repetition

The central problem is therefore:

How can we build a robust system that, given a new Jira issue, can identify the most relevant historical issues and determine whether the new issue is likely to be a duplicate of an existing one?

This problem has two closely related but distinct parts.

## 1. Similarity Analysis of Jira Issues

Given a new Jira issue as input, the system should find historical Jira issues that are strongly related in meaning, context, symptoms, affected area, or underlying problem.

This is not limited to exact text overlap. Two issues may be highly relevant to one another even if they are written differently, use different terminology, or come from different reporting styles.

The goal of similarity analysis is to support:

- faster triage
- reuse of prior issue knowledge
- easier investigation of known problem patterns
- discovery of related incidents, bugs, or tasks
- better decision-making during issue assignment and prioritization

## 2. Duplicate Analysis of Jira Issues

Given a new Jira issue as input, the system should determine whether it is likely to refer to the same underlying problem as an already existing Jira issue.

This is a stricter problem than general similarity. Two issues may be related without being duplicates. A duplicate relationship implies that opening a new issue may be redundant because the problem is already represented elsewhere in the issue history.

The goal of duplicate analysis is to support:

- reduction of redundant tickets
- cleaner issue backlogs
- lower manual triage effort
- more consistent issue tracking
- improved productivity for engineering and support teams

## Broader Research Problem

The broader research problem is not only to build one working approach, but to investigate which kinds of methods are most effective for this task under realistic conditions.

This includes understanding how well different approaches perform when applied to:

- historical public Jira datasets
- future unseen datasets
- real industrial issue repositories
- changing project structures and reporting styles

The problem is therefore not just to optimize performance on one dataset, but to study how to build issue-similarity and duplicate-detection systems that generalize well to new and varied environments.

## Practical Requirement

The system must address real industrial needs rather than only producing good results on a public benchmark dataset.

That means the problem we are solving is:

How can we create a reliable, generalizable, and practically useful Jira issue intelligence system that helps organizations find similar issues and detect duplicates accurately on real-world data?
