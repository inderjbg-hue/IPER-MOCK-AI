import http from "http";
import express from "express";
import { WebSocketServer } from "ws";
import jwt from "jsonwebtoken";
import mediasoup from "mediasoup";
import crypto from "crypto";

const PORT = Number(process.env.PORT || 8080);
const PUBLIC_IP = process.env.PUBLIC_IP || process.env.HOST_IP || "127.0.0.1";
const JWT_SECRET = process.env.JWT_SECRET || "CHANGE_ME_IN_PRODUCTION";
const DEV_MODE = String(process.env.DEV_MODE || "true").toLowerCase() === "true";
const MAX_PARTICIPANTS = 7;

const app = express();
app.use(express.json());
app.use(express.static(new URL("../client", import.meta.url).pathname));

app.get("/health", (_req, res) => {
  res.json({ ok: true, service: "IPER Video Engine", maxParticipants: MAX_PARTICIPANTS });
});

const server = http.createServer(app);
const wss = new WebSocketServer({ server });

let worker;
let router;

const rooms = new Map();
const peers = new Map();

function id() {
  return crypto.randomUUID();
}

function send(ws, message) {
  if (ws.readyState === ws.OPEN) ws.send(JSON.stringify(message));
}

function broadcast(room, message, exceptWs = null) {
  for (const peer of room.peers.values()) {
    if (peer.ws !== exceptWs) send(peer.ws, message);
  }
}

function verifyIdentity(token, profile = {}) {
  if (DEV_MODE && token === "dev") {
    return {
      studentId: profile.studentId || "DEV-" + Math.floor(Math.random() * 10000),
      firstName: profile.firstName || "Demo",
      lastName: profile.lastName || "Student",
      scholarId: profile.scholarId || "DEV001",
      email: profile.email || "demo@iper.ac.in"
    };
  }

  if (!token) throw new Error("Missing access token.");
  const payload = jwt.verify(token, JWT_SECRET);
  if (!payload.studentId || !payload.scholarId || !payload.email) {
    throw new Error("Invalid IPER identity token.");
  }
  if (!String(payload.email).toLowerCase().endsWith("@iper.ac.in")) {
    throw new Error("Only @iper.ac.in accounts may enter.");
  }

  return {
    studentId: payload.studentId,
    firstName: payload.firstName,
    lastName: payload.lastName,
    scholarId: payload.scholarId,
    email: payload.email
  };
}

async function createRouter() {
  worker = await mediasoup.createWorker({
    rtcMinPort: Number(process.env.RTC_MIN_PORT || 40000),
    rtcMaxPort: Number(process.env.RTC_MAX_PORT || 40100),
    logLevel: "warn"
  });

  worker.on("died", () => {
    console.error("mediasoup worker died; exiting.");
    setTimeout(() => process.exit(1), 2000);
  });

  router = await worker.createRouter({
    mediaCodecs: [
      {
        kind: "audio",
        mimeType: "audio/opus",
        clockRate: 48000,
        channels: 2
      },
      {
        kind: "video",
        mimeType: "video/VP8",
        clockRate: 90000,
        parameters: {
          "x-google-start-bitrate": 1000,
          "x-google-max-bitrate": 2500
        }
      }
    ]
  });
}

async function createWebRtcTransport() {
  return router.createWebRtcTransport({
    listenInfos: [
      {
        protocol: "udp",
        ip: "0.0.0.0",
        announcedAddress: PUBLIC_IP
      },
      {
        protocol: "tcp",
        ip: "0.0.0.0",
        announcedAddress: PUBLIC_IP
      }
    ],
    enableUdp: true,
    enableTcp: true,
    preferUdp: true,
    initialAvailableOutgoingBitrate: 1200000,
    enableSctp: true,
    numSctpStreams: { OS: 1024, MIS: 1024 }
  });
}

function getOrCreateRoom(roomId) {
  let room = rooms.get(roomId);
  if (!room) {
    room = {
      id: roomId,
      createdAt: Date.now(),
      peers: new Map(),
      producers: new Map()
    };
    rooms.set(roomId, room);
  }
  return room;
}

async function handleRequest(peer, request) {
  const { action, data = {}, requestId } = request;
  const reply = (payload = {}) => send(peer.ws, { type: "response", requestId, ok: true, data: payload });
  const fail = (message) => send(peer.ws, { type: "response", requestId, ok: false, error: message });

  try {
    switch (action) {
      case "join": {
        const identity = verifyIdentity(data.token, data.profile);
        const roomId = String(data.roomId || "").trim().toUpperCase();
        if (!roomId) throw new Error("GD room code is required.");

        const room = getOrCreateRoom(roomId);
        if (room.peers.size >= MAX_PARTICIPANTS) {
          throw new Error("This GD room is full. Maximum 7 students are allowed.");
        }

        peer.roomId = roomId;
        peer.identity = identity;
        room.peers.set(peer.id, peer);

        reply({
          peerId: peer.id,
          roomId,
          identity,
          maxParticipants: MAX_PARTICIPANTS,
          participants: [...room.peers.values()].map(p => ({
            peerId: p.id,
            ...p.identity
          }))
        });

        broadcast(room, {
          type: "peer-joined",
          peer: { peerId: peer.id, ...identity }
        }, peer.ws);
        break;
      }

      case "getRouterRtpCapabilities":
        reply({ routerRtpCapabilities: router.rtpCapabilities });
        break;

      case "createTransport": {
        if (!peer.roomId) throw new Error("Join a room first.");
        const transport = await createWebRtcTransport();
        transport.appData = { peerId: peer.id, direction: data.direction };
        peer.transports.set(transport.id, transport);

        transport.on("dtlsstatechange", state => {
          if (state === "closed") transport.close();
        });
        transport.on("close", () => peer.transports.delete(transport.id));

        reply({
          id: transport.id,
          iceParameters: transport.iceParameters,
          iceCandidates: transport.iceCandidates,
          dtlsParameters: transport.dtlsParameters,
          sctpParameters: transport.sctpParameters
        });
        break;
      }

      case "connectTransport": {
        const transport = peer.transports.get(data.transportId);
        if (!transport) throw new Error("Transport not found.");
        await transport.connect({ dtlsParameters: data.dtlsParameters });
        reply({});
        break;
      }

      case "produce": {
        const transport = peer.transports.get(data.transportId);
        if (!transport) throw new Error("Send transport not found.");
        const producer = await transport.produce({
          kind: data.kind,
          rtpParameters: data.rtpParameters,
          appData: { ...(data.appData || {}), peerId: peer.id }
        });
        peer.producers.set(producer.id, producer);
        const room = rooms.get(peer.roomId);
        room.producers.set(producer.id, { producer, peerId: peer.id });

        producer.on("transportclose", () => {
          peer.producers.delete(producer.id);
          room.producers.delete(producer.id);
        });

        reply({ id: producer.id });
        broadcast(room, {
          type: "new-producer",
          producerId: producer.id,
          peerId: peer.id,
          kind: producer.kind,
          identity: peer.identity
        }, peer.ws);
        break;
      }

      case "getProducers": {
        const room = rooms.get(peer.roomId);
        const list = [];
        for (const [producerId, item] of room.producers.entries()) {
          if (item.peerId !== peer.id) {
            list.push({
              producerId,
              peerId: item.peerId,
              kind: item.producer.kind,
              identity: room.peers.get(item.peerId)?.identity
            });
          }
        }
        reply({ producers: list });
        break;
      }

      case "consume": {
        const room = rooms.get(peer.roomId);
        const item = room.producers.get(data.producerId);
        if (!item) throw new Error("Producer not found.");

        if (!router.canConsume({
          producerId: data.producerId,
          rtpCapabilities: data.rtpCapabilities
        })) {
          throw new Error("This device cannot consume the requested stream.");
        }

        const recvTransport = peer.transports.get(data.transportId);
        if (!recvTransport) throw new Error("Receive transport not found.");

        const consumer = await recvTransport.consume({
          producerId: data.producerId,
          rtpCapabilities: data.rtpCapabilities,
          paused: true,
          appData: { peerId: peer.id }
        });

        peer.consumers.set(consumer.id, consumer);
        consumer.on("transportclose", () => peer.consumers.delete(consumer.id));
        consumer.on("producerclose", () => {
          peer.consumers.delete(consumer.id);
          send(peer.ws, { type: "consumer-closed", consumerId: consumer.id });
        });

        reply({
          id: consumer.id,
          producerId: data.producerId,
          kind: consumer.kind,
          rtpParameters: consumer.rtpParameters,
          identity: item.peerId === peer.id ? peer.identity : room.peers.get(item.peerId)?.identity,
          peerId: item.peerId
        });
        break;
      }

      case "resumeConsumer": {
        const consumer = peer.consumers.get(data.consumerId);
        if (!consumer) throw new Error("Consumer not found.");
        await consumer.resume();
        reply({});
        break;
      }

      case "leave":
        await closePeer(peer);
        reply({});
        break;

      default:
        throw new Error(`Unknown action: ${action}`);
    }
  } catch (error) {
    console.error("Request error:", action, error);
    fail(error.message || "Request failed.");
  }
}

async function closePeer(peer) {
  if (!peer) return;
  const room = peer.roomId ? rooms.get(peer.roomId) : null;

  for (const consumer of peer.consumers.values()) consumer.close();
  for (const producer of peer.producers.values()) producer.close();
  for (const transport of peer.transports.values()) transport.close();

  peer.consumers.clear();
  peer.producers.clear();
  peer.transports.clear();

  if (room) {
    room.peers.delete(peer.id);
    for (const [producerId, item] of room.producers.entries()) {
      if (item.peerId === peer.id) room.producers.delete(producerId);
    }

    broadcast(room, { type: "peer-left", peerId: peer.id });

    if (room.peers.size === 0) rooms.delete(room.id);
  }

  peers.delete(peer.id);
}

wss.on("connection", ws => {
  const peer = {
    id: id(),
    ws,
    roomId: null,
    identity: null,
    transports: new Map(),
    producers: new Map(),
    consumers: new Map()
  };

  peers.set(peer.id, peer);
  send(ws, { type: "connected", peerId: peer.id });

  ws.on("message", async raw => {
    try {
      const request = JSON.parse(raw.toString());
      await handleRequest(peer, request);
    } catch {
      send(ws, { type: "error", error: "Invalid message." });
    }
  });

  ws.on("close", () => closePeer(peer));
  ws.on("error", () => closePeer(peer));
});

await createRouter();

server.listen(PORT, "0.0.0.0", () => {
  console.log(`IPER Video Engine listening on port ${PORT}`);
  console.log(`Public IP/host for WebRTC: ${PUBLIC_IP}`);
  console.log(`Maximum participants per room: ${MAX_PARTICIPANTS}`);
});
