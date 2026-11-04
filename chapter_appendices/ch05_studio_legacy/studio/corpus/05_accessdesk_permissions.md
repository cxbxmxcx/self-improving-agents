# AccessDesk: requesting project membership

Document ID: ACCESS-05. Revision: 1. Owner: Studio Operations.

## Recognizing a permission problem

CrateSync PERM04 means the signed-in account does not have membership for the selected project. Check the requested project and account identity before submitting a request. An operational VPN and a valid session do not establish project authorization.

## Requesting access

Submit the project name, role, and business reason through AccessDesk for the project owner's approval. The simulator action `request_asset_access` records that request and does not grant access. Only an approved membership change followed by a successful sync closes the permission issue.

## Keeping support history scoped

Keep a user's device details and ticket history at user scope unless there is an explicit reason and authorization to share them. A generally useful troubleshooting guide can be shared without copying a user's entire ticket. Another support agent still needs an authorized store connection and the correct identity to retrieve personal case history.
