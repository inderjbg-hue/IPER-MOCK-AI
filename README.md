# IPER Video Engine — Phase 1

This is the first standalone video-conferencing engine for the IPER Placement Portal.

## Phase 1 target

- IPER-owned WebRTC conferencing
- mediasoup SFU
- Maximum 7 participants per GD room
- Authenticated participant identity
- Participant name + Scholar ID displayed on each video tile
- Mobile-friendly responsive interface
- Camera + microphone controls
- 10-minute client timer (server-authoritative timer will be added in the integration phase)
- No Jitsi / Google Meet / Zoom UI

mediasoup is an SFU designed for group video chat and relays media streams instead of mixing/transcoding them. See the official docs:
https://mediasoup.org/documentation/overview/

## Important: HTTPS

Camera/microphone access normally requires a secure context (HTTPS), except localhost during development. Production must use HTTPS/WSS.

## Local development

Requirements:
- Node.js 20+ recommended
- A machine/network capable of receiving WebRTC UDP/TCP traffic

Install:

    npm install

Run:

    DEV_MODE=true PUBLIC_IP=127.0.0.1 JWT_SECRET=replace-me npm start

Open:

    http://localhost:8080/gd_room.html?room=DEMO01&token=dev

For local testing, enter any demo profile using an @iper.ac.in email.

## Production integration

Do NOT leave DEV_MODE=true.

The main Streamlit app should create a short-lived JWT containing:

- studentId
- firstName
- lastName
- scholarId
- email
- gdSessionId / roomId
- exp

The video engine validates the JWT with the same JWT_SECRET.

The portal should open:

    https://gd.iper.ac.in/gd_room.html?room=ROOMCODE&token=SIGNED_TOKEN

## WebRTC network requirements

The server needs:
- HTTPS/WSS
- UDP/TCP WebRTC ports
- a public/announced IP
- firewall rules for the configured RTC port range
- TURN for difficult NAT/firewall environments in the production phase

## Mobile design

The client is responsive and uses:
- two-column video tiles on phones
- touch-friendly controls
- playsInline video
- front-facing camera preference
- adaptive video bitrate layers

The browser/device still determines actual camera capability.

## Next phases

Phase 2:
- Integrate with app.py authentication and GD code generation
- server-authoritative 10-minute room lifecycle
- room expiry
- host controls

Phase 3:
- server-side composite recording at 1280x720
- FFmpeg recording pipeline
- cloud storage + downloadable recording

Phase 4:
- speech transcription
- speaker attribution
- qualitative GD feedback for voice, perspective, participation and listening

Phase 5:
- separate mentor dashboard
