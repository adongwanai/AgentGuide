# InterviewGuide Agent Workflow

## Required Loop Per Session

1. Open `task.json`.
2. Pick exactly one task with `status: "pending"` (or continue one `in_progress` task).
3. Execute with TDD:
   - Write failing test.
   - Verify failure.
   - Write minimal implementation.
   - Verify pass.
4. Run verification commands required by the task.
5. Update `progress.txt` with what changed and verification evidence.
6. Update `task.json` status and `commit` field.
7. Commit code changes + `progress.txt` + `task.json` in one commit.

## Status Values

- `pending`: not started.
- `in_progress`: currently being implemented.
- `done`: completed and verified.
- `blocked`: waiting for external dependency or manual input.

## Blocking Rule

If blocked, do not mark task as done. Update `task.json` to `blocked` and write:

- blocking reason
- exact unblock actions
- next command to resume
