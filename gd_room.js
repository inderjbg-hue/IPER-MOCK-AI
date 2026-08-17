import * as mediasoupClient from "https://cdn.jsdelivr.net/npm/mediasoup-client@3.21.0/+esm";

const state = {
  ws: null,
  device: null,
  sendTransport: null,
  recvTransport: null,
  localStream: null,
  roomId: null,
  peerId: null,
  identity: null,
  consumers: new Map(),
  requestCounter: 0,
  pending: new Map(),
  producerIds: new Set(),
  secondsLeft: 600,
  timerHandle: null
};

const $ = id => document.getElementById(id);

function setStatus(online, text) {
  $("connectionDot").className = "dot " + (online ? "online" : "offline");
  $("connectionText").textContent = text;
}

function showError(target, message) {
  $(target).textContent = message || "";
}

function request(action, data = {}) {
  return new Promise((resolve, reject) => {
    const requestId = String(++state.requestCounter);
    state.pending.set(requestId, { resolve, reject });
    state.ws.send(JSON.stringify({ action, data, requestId }));
  });
}

function getToken() {
  // Production integration:
  // app.py should redirect/open this page with a short-lived signed token:
  // ?room=ABC123&token=<JWT>
  const params = new URLSearchParams(location.search);
  return params.get("token") || "dev";
}

function getProfileFromUrl() {
  const params = new URLSearchParams(location.search);
  return {
    firstName: params.get("firstName") || $("firstName").value.trim(),
    lastName: params.get("lastName") || $("lastName").value.trim(),
    scholarId: params.get("scholarId") || $("scholarId").value.trim(),
    email: params.get("email") || $("email").value.trim()
  };
}

async function connectSocket() {
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  state.ws = new WebSocket(`${protocol}//${location.host}`);

  await new Promise((resolve, reject) => {
    state.ws.addEventListener("open", resolve, { once: true });
    state.ws.addEventListener("error", reject, { once: true });
  });

  state.ws.addEventListener("message", onMessage);
  state.ws.addEventListener("close", () => setStatus(false, "Disconnected"));
  setStatus(true, "Connected");
}

function onMessage(event) {
  const message = JSON.parse(event.data);

  if (message.type === "response") {
    const pending = state.pending.get(message.requestId);
    if (!pending) return;
    state.pending.delete(message.requestId);
    message.ok ? pending.resolve(message.data) : pending.reject(new Error(message.error));
    return;
  }

  if (message.type === "new-producer") {
    consumeProducer(message.producerId, message.peerId, message.identity).catch(err =>
      showError("roomError", err.message)
    );
    return;
  }

  if (message.type === "peer-joined") {
    // The server's identity is authoritative; no user-entered name is trusted.
    return;
  }

  if (message.type === "peer-left") {
    for (const [consumerId, item] of state.consumers.entries()) {
      if (item.peerId === message.peerId) {
        item.consumer.close();
        item.card?.remove();
        state.consumers.delete(consumerId);
      }
    }
  }

  if (message.type === "consumer-closed") {
    const item = state.consumers.get(message.consumerId);
    if (item) {
      item.consumer.close();
      item.card?.remove();
      state.consumers.delete(message.consumerId);
    }
  }
}

async function setupDevice() {
  const capabilities = await request("getRouterRtpCapabilities");
  state.device = new mediasoupClient.Device();
  await state.device.load({ routerRtpCapabilities: capabilities.routerRtpCapabilities });
}

async function createSendTransport() {
  const params = await request("createTransport", { direction: "send" });
  state.sendTransport = state.device.createSendTransport(params);

  state.sendTransport.on("connect", async ({ dtlsParameters }, callback, errback) => {
    try {
      await request("connectTransport", {
        transportId: state.sendTransport.id,
        dtlsParameters
      });
      callback();
    } catch (e) { errback(e); }
  });

  state.sendTransport.on("produce", async ({ kind, rtpParameters, appData }, callback, errback) => {
    try {
      const { id } = await request("produce", {
        transportId: state.sendTransport.id,
        kind,
        rtpParameters,
        appData
      });
      callback({ id });
      state.producerIds.add(id);
    } catch (e) { errback(e); }
  });
}

async function createRecvTransport() {
  const params = await request("createTransport", { direction: "recv" });
  state.recvTransport = state.device.createRecvTransport(params);

  state.recvTransport.on("connect", async ({ dtlsParameters }, callback, errback) => {
    try {
      await request("connectTransport", {
        transportId: state.recvTransport.id,
        dtlsParameters
      });
      callback();
    } catch (e) { errback(e); }
  });
}

async function startLocalMedia() {
  state.localStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true
    },
    video: {
      width: { ideal: 1280, max: 1920 },
      height: { ideal: 720, max: 1080 },
      frameRate: { ideal: 30, max: 30 },
      facingMode: "user"
    }
  });

  addVideoTile("local", state.localStream, state.identity, true);

  const audioTrack = state.localStream.getAudioTracks()[0];
  const videoTrack = state.localStream.getVideoTracks()[0];

  await state.sendTransport.produce({
    track: audioTrack,
    appData: { source: "microphone" }
  });

  await state.sendTransport.produce({
    track: videoTrack,
    appData: { source: "camera" },
    encodings: [
      { maxBitrate: 250000, scaleResolutionDownBy: 2 },
      { maxBitrate: 700000, scaleResolutionDownBy: 1.5 },
      { maxBitrate: 1500000, scaleResolutionDownBy: 1 }
    ],
    codecOptions: { videoGoogleStartBitrate: 1000 }
  });
}

async function consumeProducer(producerId, peerId, identity) {
  if (!state.recvTransport || state.consumers.has(producerId)) return;

  const params = await request("consume", {
    producerId,
    transportId: state.recvTransport.id,
    rtpCapabilities: state.device.recvRtpCapabilities
  });

  const consumer = await state.recvTransport.consume({
    id: params.id,
    producerId: params.producerId,
    kind: params.kind,
    rtpParameters: params.rtpParameters
  });

  const stream = new MediaStream([consumer.track]);
  const card = getOrCreateRemoteCard(peerId, identity);
  const video = card.querySelector("video");

  if (params.kind === "video") {
    video.srcObject = stream;
  } else {
    const audio = document.createElement("audio");
    audio.autoplay = true;
    audio.srcObject = stream;
    card.appendChild(audio);
  }

  state.consumers.set(consumer.id, { consumer, peerId, identity, card });
  await request("resumeConsumer", { consumerId: consumer.id });
}

function addVideoTile(key, stream, identity, local = false) {
  const card = document.createElement("div");
  card.className = "video-card";
  card.dataset.key = key;

  const video = document.createElement("video");
  video.autoplay = true;
  video.playsInline = true;
  video.muted = local;
  video.srcObject = stream;

  const label = document.createElement("div");
  label.className = "video-label";
  label.innerHTML = `<strong>${escapeHtml(identity.firstName)} ${escapeHtml(identity.lastName)}</strong>${escapeHtml(identity.scholarId)}`;

  card.append(video, label);
  $("videoGrid").appendChild(card);
  return card;
}

function getOrCreateRemoteCard(peerId, identity) {
  const existing = [...$("videoGrid").children].find(c => c.dataset.key === peerId);
  if (existing) return existing;

  return addVideoTile(peerId, new MediaStream(), identity, false);
}

function escapeHtml(value) {
  return String(value || "").replace(/[&<>"']/g, c => ({
    "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#039;"
  }[c]));
}

async function joinRoom() {
  showError("setupError", "");

  const roomId = $("roomCode").value.trim().toUpperCase();
  const profile = getProfileFromUrl();

  if (!roomId) throw new Error("Enter the GD room code.");
  if (!profile.firstName || !profile.lastName || !profile.scholarId || !profile.email) {
    throw new Error("Complete the student profile fields.");
  }
  if (!profile.email.toLowerCase().endsWith("@iper.ac.in")) {
    throw new Error("Only @iper.ac.in email IDs can join.");
  }

  await connectSocket();

  const joined = await request("join", {
    roomId,
    token: getToken(),
    profile
  });

  state.roomId = joined.roomId;
  state.peerId = joined.peerId;
  state.identity = joined.identity;

  $("roomLabel").textContent = `Room ${state.roomId} • ${state.identity.firstName} ${state.identity.lastName}`;
  $("setup").classList.add("hidden");
  $("room").classList.remove("hidden");

  await setupDevice();
  await createSendTransport();
  await createRecvTransport();
  await startLocalMedia();

  const existing = await request("getProducers");
  for (const p of existing.producers) {
    await consumeProducer(p.producerId, p.peerId, p.identity);
  }

  startTimer();
}

function startTimer() {
  clearInterval(state.timerHandle);
  renderTimer();

  state.timerHandle = setInterval(() => {
    state.secondsLeft -= 1;
    renderTimer();

    if (state.secondsLeft <= 0) {
      clearInterval(state.timerHandle);
      alert("The 10-minute GD time limit has been reached.");
      leaveRoom();
    }
  }, 1000);
}

function renderTimer() {
  const m = Math.floor(state.secondsLeft / 60);
  const s = state.secondsLeft % 60;
  $("timer").textContent = `${String(m).padStart(2,"0")}:${String(s).padStart(2,"0")}`;
}

function setMic(enabled) {
  state.localStream?.getAudioTracks().forEach(t => t.enabled = enabled);
  $("micBtn").textContent = enabled ? "🎙️ Mic" : "🔇 Mic";
}

function setCamera(enabled) {
  state.localStream?.getVideoTracks().forEach(t => t.enabled = enabled);
  $("cameraBtn").textContent = enabled ? "📷 Camera" : "🚫 Camera";
}

async function leaveRoom() {
  clearInterval(state.timerHandle);
  try { await request("leave"); } catch {}
  state.localStream?.getTracks().forEach(t => t.stop());
  state.ws?.close();
  location.reload();
}

$("joinBtn").addEventListener("click", async () => {
  $("joinBtn").disabled = true;
  try {
    await joinRoom();
  } catch (e) {
    showError("setupError", e.message || "Unable to join the GD room.");
    $("joinBtn").disabled = false;
  }
});

$("micBtn").addEventListener("click", () => {
  const enabled = state.localStream?.getAudioTracks()[0]?.enabled !== false;
  setMic(!enabled);
});

$("cameraBtn").addEventListener("click", () => {
  const enabled = state.localStream?.getVideoTracks()[0]?.enabled !== false;
  setCamera(!enabled);
});

$("leaveBtn").addEventListener("click", leaveRoom);

$("roomCode").addEventListener("input", e => {
  e.target.value = e.target.value.toUpperCase().replace(/[^A-Z0-9]/g, "");
});
