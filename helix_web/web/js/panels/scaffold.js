// Scaffold panel (chapter 6): the metacognition composite, one lane per
// constituent, with the hook-fire timeline and the behavior-diff pane.
// Outline until composites are resolved at runtime and HookFired is published.

import { renderOutline } from "../components/outline.js";

export async function render(container) {
  renderOutline(container, {
    title: "Scaffold",
    caption:
      "A composite evolves as one artifact but is read as four: planner, " +
      "monitor, reflector, and task state each get a lane. Below it, which " +
      "hooks fired on a run and what they returned, and two trajectories " +
      "diffed to show whether a reflection changed behavior or only output.",
    chapter: 6,
    route: "#/scaffold?archive=<key>&composite=<id>&a=<traj>&b=<traj>",
    endpoints: [
      "GET /api/archives/{key}/composites",
      "GET /api/archives/{key}/lineage?lanes=constituents&composite=",
      "GET /api/archives/{key}/trajectories/{id}/hooks",
      "GET /api/archives/{key}/behavior-diff?a=&b=",
    ],
    events: ["HookFired", "ArtifactCreated (composite)", "TrajectoryStepRecorded"],
    figures: [
      "6.3: reflection theater versus behavior change as two diffs",
      "6.4: the dual-process router's cost curve",
      "6.5: composite lineage with constituent lanes",
    ],
    regions: [
      { name: "Constituent lanes", note: "planner · monitor · reflector · task state as columns; each a mini lineage; composite versions as rows", grow: true },
      { name: "Hook timeline", note: "per run: PRE_MODEL, PRE_TOOL, POST_TOOL, END_OF_TURN, SESSION_END; which wrapper answered and what it returned (pass, patch, refuse)" },
      { name: "Behavior diff", note: "two trajectories side by side: tool calls, steps, outcome; divergence score from the behavior-diff signal" },
      { name: "Router cost curve", note: "System 1 versus System 2 share, cost per task, pass rate on hidden tests", grow: true },
    ],
  });
}
