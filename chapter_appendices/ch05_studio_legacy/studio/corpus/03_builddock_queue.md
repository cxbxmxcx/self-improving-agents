# BuildDock: finding a delayed playtest build

Document ID: BUILD-03. Revision: 1. Owner: Build Engineering.

## Separating queued and failed builds

BuildDock displays a job's ID, project, branch, commit, and state. QUEUED means the job is waiting for a worker, RUNNING means a worker has started it, and FAILED means execution ended with an error. A long queue is not evidence that the source code is broken.

## Checking the right job

Record the job ID and inspect its current queue state before submitting another build. In the teaching simulator, `check_build_queue` returns that state and creates no new job. Match the project and commit to the playtest request before handing over an artifact.

## Handling a blocked release

If a job remains queued, report its age and worker status to Build Engineering. If it failed, preserve the first failing step and its log reference. The helpdesk cannot approve queue priority, promise an arrival time, or invent a successful build.
