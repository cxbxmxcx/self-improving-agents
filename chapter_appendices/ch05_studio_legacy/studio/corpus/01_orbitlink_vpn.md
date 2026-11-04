# OrbitLink VPN: reconnecting after sleep

Document ID: VPN-01. Revision: 1. Owner: Studio IT.

## Recognizing a disconnected tunnel

Second Breakfast Games uses OrbitLink to reach internal services from remote laptops. CrateSync error NET01 means the workstation cannot reach the studio asset service; it does not establish which network component failed. Check OrbitLink's connection status before changing the asset client.

## Restoring the connection

If OrbitLink reports disconnected after the laptop wakes, reconnect the existing Studio profile and retry the asset sync once. Record whether the tunnel connected and whether the retry succeeded. In the teaching simulator, the corresponding support action is `reconnect_vpn`.

## Knowing when to move on

If the tunnel is already connected and CrateSync reports AUTH17, follow ASSET-02 to refresh the asset session. Repeated VPN reconnects do not refresh that session. An earlier successful reconnect is useful case history, but it does not establish the cause of today's failure.

## Recording an unresolved case

If reconnecting fails or NET01 persists, collect the current error, connection state, and time of failure for Studio IT. Keep local work intact and avoid changing managed network profiles. The playtest can wait longer than an accidentally deleted workspace.
