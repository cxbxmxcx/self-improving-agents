// Proposals panel (chapter 7): the human gate. A queue of candidates the
// search produced, each with its diff, behavior diffs, and test result,
// and the decide buttons. Outline until the gate writes a proposals table.

import { renderOutline } from "../components/outline.js";

export async function render(container) {
  renderOutline(container, {
    title: "Proposals",
    caption:
      "Search proposes, a human decides. Each card shows what changed, what " +
      "the tests said, and how three replayed trajectories behave under the " +
      "candidate. Rollback promotes the parent.",
    chapter: 7,
    route: "#/proposals?archive=<key>&status=open|approved|rejected&sel=<proposal>",
    endpoints: [
      "GET /api/archives/{key}/proposals?status=",
      "GET /api/archives/{key}/proposals/{id}",
      "POST /api/archives/{key}/proposals/{id}/decide  {decision: approve|reject|rollback, approver}",
    ],
    events: ["ProposalCreated", "ProposalDecided", "Promoted"],
    figures: [
      "7.4: a proposal card with three behavior diffs",
      "7.5: the harness composite's multi-lane lineage (reuses the scaffold lanes)",
    ],
    regions: [
      { name: "Queue", note: "open first; candidate ref, parent ref, artifact kind, test result, age" },
      { name: "Proposal card", note: "unified diff (text or code), test gate result, approver, decision buttons" },
      { name: "Behavior diffs", note: "three recent trajectories replayed against parent and candidate; before/after traces, divergence per trajectory", grow: true },
      { name: "Decision log", note: "who decided what and when; rollback promotes the parent version", grow: true },
    ],
  });
}
