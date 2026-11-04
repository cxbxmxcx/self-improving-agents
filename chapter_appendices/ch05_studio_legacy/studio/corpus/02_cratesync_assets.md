# CrateSync: getting Moonbase Mayhem assets

Document ID: ASSET-02. Revision: 1. Owner: Tools Team.

## Distinguishing the failure

CrateSync transfers project assets into a developer's existing workspace. NET01 indicates an unreachable asset service and starts with the OrbitLink connection check in VPN-01. AUTH17 indicates an expired asset-client session, while PERM04 indicates missing project membership.

## Refreshing an expired session

For AUTH17 with a connected VPN, refresh the CrateSync sign-in session and retry the same sync. Preserve the workspace, project selection, and local changes. The teaching simulator names this action `refresh_asset_token`; it never handles real credentials or tokens.

## Respecting project access

For PERM04, request Moonbase Mayhem membership through AccessDesk as described in ACCESS-05. Reconnecting the VPN or refreshing a session does not grant project access. Do not borrow another developer's account to get a playtest moving.

## Preserving work during diagnosis

Do not delete the workspace, clear unsaved assets, or reinstall the client as the opening troubleshooting step. Record the exact error and the result of each attempted fix. A ticket is resolved only when the user or simulator confirms the sync succeeds.
