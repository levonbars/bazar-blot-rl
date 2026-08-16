"use strict";

const RANKS = ["7", "8", "9", "10", "J", "Q", "K", "A"];
const SUITS = ["C", "D", "H", "S"];
const SUIT_SYMBOL = { C: "♣", D: "♦", H: "♥", S: "♠" };
const SUIT_COLOR = { C: "black", D: "red", H: "red", S: "black" };
const SEAT_NAMES = ["South", "West", "North", "East"];

function cardInfo(cid) {
  const rank = RANKS[cid % 8];
  const suit = SUITS[Math.floor(cid / 8)];
  return { rank, suit, symbol: SUIT_SYMBOL[suit], color: SUIT_COLOR[suit] };
}

// ---------------------------------------------------------------- app state

let mode = null; // "watch" | "play" | "replay"
let sessionId = null;
let humanSeat = null;
let latest = null; // last fetched state payload

// ---------------------------------------------------------------- fetch helpers

async function api(method, path, body) {
  const opts = { method, headers: {} };
  if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(path, opts);
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.detail || `request failed: ${res.status}`);
  }
  return data;
}

// ---------------------------------------------------------------- lobby

document.getElementById("startWatch").addEventListener("click", async () => {
  const seedInput = document.getElementById("watchSeed").value;
  const body = { mode: "watch" };
  if (seedInput) body.match_seed = parseInt(seedInput, 10);
  try {
    const data = await api("POST", "/api/game", body);
    enterGame("watch", data);
  } catch (e) {
    alert(e.message);
  }
});

document.getElementById("startPlay").addEventListener("click", async () => {
  const seat = parseInt(document.getElementById("playSeat").value, 10);
  const seedInput = document.getElementById("playSeed").value;
  const body = { mode: "play", human_seat: seat };
  if (seedInput) body.match_seed = parseInt(seedInput, 10);
  try {
    const data = await api("POST", "/api/game", body);
    enterGame("play", data);
  } catch (e) {
    alert(e.message);
  }
});

document.getElementById("startReplay").addEventListener("click", async () => {
  const raw = document.getElementById("replayInput").value;
  let log;
  try {
    log = JSON.parse(raw);
  } catch (e) {
    alert("that doesn't look like valid JSON");
    return;
  }
  try {
    const data = await api("POST", "/api/replay", { log });
    enterReplay(data);
  } catch (e) {
    alert(e.message);
  }
});

function enterGame(m, data) {
  mode = m;
  sessionId = data.session_id;
  humanSeat = data.human_seat;
  document.getElementById("lobby").classList.add("hidden");
  document.getElementById("game").classList.remove("hidden");
  document.getElementById("matchInfo").classList.remove("hidden");
  document.getElementById("watchControls").classList.toggle("hidden", m !== "watch");
  document.getElementById("replayControls").classList.add("hidden");
  document.getElementById("copyReplayBtn").classList.remove("hidden");
  renderGame(data);
}

function enterReplay(data) {
  mode = "replay";
  sessionId = data.session_id;
  humanSeat = null;
  document.getElementById("lobby").classList.add("hidden");
  document.getElementById("game").classList.remove("hidden");
  document.getElementById("matchInfo").classList.add("hidden");
  document.getElementById("watchControls").classList.add("hidden");
  document.getElementById("auctionPanel").classList.add("hidden");
  document.getElementById("nextDealBtn").classList.add("hidden");
  document.getElementById("copyReplayBtn").classList.add("hidden");
  document.getElementById("replayControls").classList.remove("hidden");
  renderReplay(data);
}

// ---------------------------------------------------------------- watch/play controls

document.getElementById("stepBtn").addEventListener("click", async () => {
  const data = await api("POST", `/api/game/${sessionId}/bot_step`);
  renderGame(data);
});

document.getElementById("autoplayBtn").addEventListener("click", async () => {
  const data = await api("POST", `/api/game/${sessionId}/autoplay`);
  renderGame(data);
});

document.getElementById("nextDealBtn").addEventListener("click", async () => {
  const data = await api("POST", `/api/game/${sessionId}/next_deal`);
  renderGame(data);
});

document.getElementById("copyReplayBtn").addEventListener("click", async () => {
  try {
    const log = await api("GET", `/api/game/${sessionId}/replay_log`);
    await navigator.clipboard.writeText(JSON.stringify(log));
    alert("Replay log copied to clipboard.");
  } catch (e) {
    alert(e.message);
  }
});

document.getElementById("passBtn").addEventListener("click", () => submitAction({ type: "pass" }));
document.getElementById("contraBtn").addEventListener("click", () => submitAction({ type: "contra" }));
document.getElementById("recontraBtn").addEventListener("click", () => submitAction({ type: "recontra" }));
document.getElementById("bidBtn").addEventListener("click", () => {
  submitAction({
    type: "bid",
    level: parseInt(document.getElementById("bidLevel").value, 10),
    contract_type: document.getElementById("bidType").value,
    capot: document.getElementById("bidCapot").checked,
  });
});

async function submitAction(action) {
  const seat = latest.state.to_act;
  try {
    const data = await api("POST", `/api/game/${sessionId}/action`, { seat, action });
    renderGame(data);
  } catch (e) {
    alert(e.message);
  }
}

async function playCard(card) {
  await submitAction({ type: "play", card });
}

// ---------------------------------------------------------------- replay controls

document.getElementById("replayNext").addEventListener("click", async () => {
  const data = await api("POST", `/api/replay/${sessionId}/step`);
  renderReplay(data);
});
document.getElementById("replayPrev").addEventListener("click", async () => {
  const idx = Math.max(0, latest.index - 1);
  const data = await api("POST", `/api/replay/${sessionId}/goto`, { index: idx });
  renderReplay(data);
});

// ---------------------------------------------------------------- rendering

function seatEl(seat) {
  return document.getElementById(`seat${seat}`);
}
function handEl(seat) {
  return document.getElementById(`hand${seat}`);
}

function makeCardDiv(cid, opts) {
  opts = opts || {};
  const info = cardInfo(cid);
  const div = document.createElement("div");
  div.className = `card ${info.color}`;
  div.innerHTML = `<div>${info.rank}</div><div>${info.symbol}</div>`;
  if (opts.legal) {
    div.classList.add("legal");
    div.addEventListener("click", () => playCard(cid));
  } else if (opts.dim) {
    div.classList.add("illegal");
  }
  return div;
}

function makeBackDiv() {
  const div = document.createElement("div");
  div.className = "card back";
  return div;
}

function makeCountDiv(n) {
  const div = document.createElement("div");
  div.className = "card count";
  div.textContent = n;
  return div;
}

function clearHighlights() {
  for (let s = 0; s < 4; s++) seatEl(s).classList.remove("to-act");
}

function highlightToAct(seat) {
  clearHighlights();
  if (seat !== null && seat !== undefined) seatEl(seat).classList.add("to-act");
}

function renderTrick(currentTrick) {
  const el = document.getElementById("trick");
  el.innerHTML = "";
  currentTrick.forEach(([seat, cid]) => {
    const slot = document.createElement("div");
    slot.className = "card-slot";
    const tag = document.createElement("div");
    tag.className = "seat-tag";
    tag.textContent = SEAT_NAMES[seat];
    slot.appendChild(tag);
    slot.appendChild(makeCardDiv(cid));
    el.appendChild(slot);
  });
}

function renderContractBanner(state) {
  const el = document.getElementById("contractBanner");
  if (!state.contract) {
    el.textContent = state.phase === "AUCTION" ? "Auction in progress…" : "";
    return;
  }
  const c = state.contract;
  const doubling = c.doubling !== "none" ? ` (${c.doubling})` : "";
  el.textContent = `${c.contract_type}-${c.level}${c.capot ? " Capot" : ""}${doubling} — declarer seat ${c.declarer_seat}`;
}

function renderDealBanner(state) {
  document.getElementById("dealBanner").textContent = `Deal ${state.deal_number} · ${state.phase}`;
}

function renderBidLog(entries) {
  const el = document.getElementById("bidLog");
  el.innerHTML = "";
  entries
    .slice()
    .reverse()
    .forEach((e) => {
      const li = document.createElement("li");
      let text = `Seat ${e.seat}: `;
      if (e.kind === "pass") text += "pass";
      else if (e.kind === "contra") text += "CONTRA";
      else if (e.kind === "recontra") text += "RECONTRA";
      else text += `bid ${e.contract_type}-${e.level}${e.capot ? " capot" : ""}`;
      li.textContent = text;
      el.appendChild(li);
    });
}

function renderEvents(events) {
  const el = document.getElementById("eventLog");
  el.innerHTML = "";
  events
    .slice()
    .reverse()
    .forEach((e) => {
      const li = document.createElement("li");
      li.textContent = e;
      el.appendChild(li);
    });
}

function renderResult(state) {
  const el = document.getElementById("resultPanel");
  if (!state.result) {
    el.innerHTML = "";
    return;
  }
  const r = state.result;
  const cls = r.made ? "made" : "failed";
  el.innerHTML = `<div class="${cls}">${r.made ? "MADE" : "FAILED"}</div>
    <div>Attackers (team ${r.attackers_team}): raw ${r.raw_attackers}, score +${r.score_attackers}</div>
    <div>Defenders: raw ${r.raw_defenders}, score +${r.score_defenders}</div>`;
}

function bidTypeOptions(types) {
  const sel = document.getElementById("bidType");
  sel.innerHTML = "";
  types.forEach((t) => {
    const opt = document.createElement("option");
    opt.value = t;
    opt.textContent = t;
    sel.appendChild(opt);
  });
}

function renderAuctionPanel(legalActions, canAct) {
  const panel = document.getElementById("auctionPanel");
  if (!canAct || !legalActions || legalActions.length === 0) {
    panel.classList.add("hidden");
    return;
  }
  panel.classList.remove("hidden");
  const has = (t) => legalActions.find((a) => a.type === t);
  document.getElementById("passBtn").classList.toggle("hidden", !has("pass"));
  document.getElementById("contraBtn").classList.toggle("hidden", !has("contra"));
  document.getElementById("recontraBtn").classList.toggle("hidden", !has("recontra"));
  const bid = has("bid");
  document.getElementById("bidForm").classList.toggle("hidden", !bid);
  if (bid) {
    bidTypeOptions(bid.contract_types);
    const levelInput = document.getElementById("bidLevel");
    levelInput.min = bid.min_level;
    levelInput.max = bid.max_level;
    levelInput.value = bid.min_level;
    const capot = document.getElementById("bidCapot");
    capot.checked = bid.forced_capot;
    capot.disabled = bid.forced_capot;
  }
}

function renderNextDealButton(state) {
  const show = state.phase === "TERMINAL" || state.phase === "ABORTED";
  document.getElementById("nextDealBtn").classList.toggle("hidden", !show);
}

// ---- watch/play dispatch

function renderGame(data) {
  latest = data;
  document.getElementById("matchScore").textContent = `Match: ${data.match_scores[0]} – ${data.match_scores[1]}`;
  document.getElementById("matchTarget").textContent = `(target ${data.match_target})`;
  renderBidLog(data.bid_log);
  renderEvents(data.events);

  const state = data.state;
  renderContractBanner(state);
  renderDealBanner(state);
  renderTrick(state.current_trick);
  renderResult(state);
  renderNextDealButton(state);
  highlightToAct(state.to_act);

  if (data.mode === "watch") {
    renderFullHands(state, null);
    renderAuctionPanel(state.legal_actions, state.to_act !== null);
  } else {
    renderPlayerHands(state);
    const canAct = state.to_act === humanSeat;
    renderAuctionPanel(state.legal_actions, canAct);
  }

  document.getElementById("watchControls").classList.toggle(
    "hidden",
    mode !== "watch" || state.phase === "TERMINAL" || state.phase === "ABORTED"
  );
}

function renderFullHands(state, highlightSeat) {
  for (let s = 0; s < 4; s++) {
    const el = handEl(s);
    el.innerHTML = "";
    const legalCards = s === state.to_act ? new Set((state.legal_actions || []).filter((a) => a.type === "play").map((a) => a.card)) : new Set();
    state.hands[s].forEach((cid) => {
      el.appendChild(makeCardDiv(cid, { legal: legalCards.has(cid), dim: s === state.to_act && legalCards.size > 0 && !legalCards.has(cid) }));
    });
  }
}

function renderPlayerHands(state) {
  for (let s = 0; s < 4; s++) {
    const el = handEl(s);
    el.innerHTML = "";
    if (s === humanSeat) {
      const legalCards = new Set((state.legal_actions || []).filter((a) => a.type === "play").map((a) => a.card));
      const isMyTurn = state.to_act === humanSeat;
      state.my_hand.forEach((cid) => {
        el.appendChild(
          makeCardDiv(cid, {
            legal: isMyTurn && legalCards.has(cid),
            dim: isMyTurn && legalCards.size > 0 && !legalCards.has(cid),
          })
        );
      });
    } else {
      const n = state.hand_sizes[s];
      for (let i = 0; i < n; i++) el.appendChild(makeBackDiv());
    }
  }
}

// ---- replay dispatch

function renderReplay(data) {
  latest = data;
  document.getElementById("replayPos").textContent = `${data.index} / ${data.total_actions}`;
  const state = data.state;
  renderContractBanner(state);
  renderDealBanner(state);
  renderTrick(state.current_trick);
  renderResult(state);
  highlightToAct(state.to_act);
  renderFullHands(state, null);
  document.getElementById("bidLog").innerHTML = "";
  document.getElementById("eventLog").innerHTML = "";
}
