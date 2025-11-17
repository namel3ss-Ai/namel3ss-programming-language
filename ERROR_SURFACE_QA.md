# ErrorSurface QA Checklist

Use this checklist to verify that the ErrorSurface plumbing behaves as expected across server-rendered pages, interactive widgets, and realtime refresh paths.

## Prerequisites
- Build a fresh static bundle (`namel3ss build frontend`) and start the demo backend.
- Ensure browser console is open with network tab recording.
- Reset application database or run against seeded test data to avoid residue from previous runs.

## Page Load / Hydration
- Load a page that includes forms and actions; confirm that any server-side errors appear in the page-level banner (`data-n3-page-errors`).
- Confirm that dismissing/refreshing the page clears the banner when errors are absent.
- Verify that component-level errors (e.g., dataset fetch failures) render under the respective widget sections with correct severity class.

## Form Submission
- Submit a form with all required fields blank; expect field-level inline errors (`scope: field:<name>`), a page status of `error`, and no backend side-effects.
- Submit with partially completed data to ensure only the missing fields show errors and the rest remain clear.
- Submit valid data and confirm form errors clear, success toast appears (if configured), and any page-level banner updates or clears.
- Cause a backend validation error (e.g., simulated 422/400) and verify the payload surfaces in both the field slot and the widget error list with severity preserved.

## Action Trigger
- Trigger an action that intentionally fails (e.g., backend raises `HTTPException`); confirm the action error slot displays the message and page banner reflects the error list.
- Run an action that succeeds but publishes warnings; verify warnings appear without blocking follow-up actions and page banner shows `warning` severity.
- Ensure repeated action invocations clear previous errors when the backend responds successfully.

## Realtime / Polling Updates
- Enable polling or websocket updates for a page that streams component changes.
- Cause a backend-side dataset error during a push; confirm the updated payload hydrates component errors and the page banner without requiring a manual refresh.
- After the issue resolves, verify a subsequent snapshot clears the page-level banner and widget error slot.

## Observability
- Validate that network responses for forms and actions carry the structured `errors`, `pageErrors`, and `status` fields.
- Inspect browser devtools to confirm ErrorSurface DOM nodes toggle `n3-widget-errors--hidden` and severity modifier classes correctly.
- Confirm console logs remain free of uncaught exceptions during error rendering/clearing cycles.
