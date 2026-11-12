"use strict";

// ================================================================ constants

const RANKS = ["7", "8", "9", "10", "J", "Q", "K", "A"];
const SUITS = ["C", "D", "H", "S"];
const SUIT_SYMBOL = { C: "♣", D: "♦", H: "♥", S: "♠" };
const RED_SUITS = new Set(["D", "H"]);
const SEAT_NAMES = ["South", "West", "North", "East"];
// display order of suits in a hand (alternating colours), and rank order inside a suit
const SUIT_DISPLAY = ["S", "H", "C", "D"];
const PLAIN_ORDER = ["A", "10", "K", "Q", "J", "9", "8", "7"];
const TRUMP_ORDER = ["J", "9", "A", "10", "K", "Q", "8", "7"];

const SPEEDS = [2.2, 1.5, 1.0, 0.6, 0.3]; // delay multipliers for the speed slider
const DELAY = { bid: 750, card: 600, trickShow: 1000, trickSweep: 450, aborted: 1600 };

function cardInfo(cid) {
  const rank = RANKS[cid % 8];
  const suit = SUITS[Math.floor(cid / 8)];
  return { rank, suit, symbol: SUIT_SYMBOL[suit], red: RED_SUITS.has(suit) };
}

// ================================================================ app state

const app = {
  mode: null, // "play" | "watch" | "replay"
  sessionId: null,
  humanSeat: null,
  latest: null,
  busy: false, // an action / animation is in flight
  autoplay: false, // watch mode: running
  speed: 2,
  generation: 0, // bumped when leaving a game, to stop stale loops
  shownTrick: [], // cards currently rendered in the trick zone, for fly-in detection
  level: 8,
};

const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const scaled = (ms) => ms * (app.mode === "watch" ? SPEEDS[app.speed] : 1);

// ================================================================ fetch helpers

async function api(method, path, body) {
  const opts = { method, headers: {} };
  if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(path, opts);
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || `request failed: ${res.status}`);
  return data;
}

let toastTimer = null;
function toast(msg, kind) {
  const el = $("toast");
  el.textContent = msg;
  el.className = "toast" + (kind === "info" ? " info" : "");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.add("hidden"), 3200);
}

// ================================================================ seats, teams, names

function pos(seat) {
  // screen position: 0 bottom, 1 left, 2 top, 3 right (clockwise = play order)
  if (app.mode === "play") return (seat - app.humanSeat + 4) % 4;
  return seat;
}
function seatAt(p) {
  for (let s = 0; s < 4; s++) if (pos(s) === p) return s;
  return p;
}
const teamOf = (seat) => seat % 2;
function usTeam() {
  return app.mode === "play" ? teamOf(app.humanSeat) : 0;
}
function teamName(team) {
  if (app.mode === "play") return team === usTeam() ? "Us" : "Them";
  return team === 0 ? "N–S" : "E–W";
}
function teamClass(team) {
  return team === usTeam() ? "us" : "them";
}
function seatName(seat) {
  if (app.mode === "play") {
    if (seat === app.humanSeat) return "You";
    if (seat === (app.humanSeat + 2) % 4) return "Partner";
  }
  return SEAT_NAMES[seat];
}
function seatRole(seat) {
  if (app.mode === "play") {
    if (seat === app.humanSeat) return SEAT_NAMES[seat];
    if (seat === (app.humanSeat + 2) % 4) return `${SEAT_NAMES[seat]} · AI`;
    return "Opponent · AI";
  }
  return teamName(teamOf(seat));
}

// ================================================================ formatting

function suitHtml(type) {
  if (type === "NT") return "NT";
  const red = RED_SUITS.has(type);
  return `<span class="${red ? "red" : ""}">${SUIT_SYMBOL[type]}</span>`;
}
function bidHtml(level, type, capot) {
  return `${level}${suitHtml(type)}${capot ? ' <span class="tag tag-gold">capot</span>' : ""}`;
}
function contractText(c) {
  const t = c.contract_type === "NT" ? "No trump" : { C: "Clubs", D: "Diamonds", H: "Hearts", S: "Spades" }[c.contract_type];
  return `${c.level} ${t}`;
}

// ================================================================ cards

function cardEl(cid, opts = {}) {
  const info = cardInfo(cid);
  const div = document.createElement("div");
  div.className = `card face${info.red ? " red" : ""}${opts.small ? " small" : ""}`;
  const court = ["J", "Q", "K"].includes(info.rank);
  const mid = court
    ? `<div class="court"><span class="court-l">${info.rank}</span><span class="court-s">${info.symbol}</span></div>`
    : `<span class="big-pip">${info.symbol}</span>`;
  div.innerHTML =
    `<div class="corner tl"><span class="r">${info.rank}</span><span class="s">${info.symbol}</span></div>` +
    `<div class="mid">${mid}</div>` +
    `<div class="corner br"><span class="r">${info.rank}</span><span class="s">${info.symbol}</span></div>`;
  return div;
}
function backEl(small) {
  const div = document.createElement("div");
  div.className = "card back" + (small ? " small" : "");
  return div;
}

function sortHand(cards, contract) {
  const trump = contract && contract.contract_type !== "NT" ? contract.contract_type : null;
  const suitOrder = trump ? [trump, ...SUIT_DISPLAY.filter((s) => s !== trump)] : SUIT_DISPLAY;
  return cards.slice().sort((a, b) => {
    const A = cardInfo(a), B = cardInfo(b);
    if (A.suit !== B.suit) return suitOrder.indexOf(A.suit) - suitOrder.indexOf(B.suit);
    const order = A.suit === trump ? TRUMP_ORDER : PLAIN_ORDER;
    return order.indexOf(A.rank) - order.indexOf(B.rank);
  });
}

// ================================================================ title screen

for (const tab of document.querySelectorAll(".mode-tab")) {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".mode-tab").forEach((t) => t.classList.toggle("active", t === tab));
    for (const m of ["play", "watch", "replay"]) $(`panel-${m}`).classList.toggle("hidden", m !== tab.dataset.mode);
  });
}
let pickedSeat = 0;
for (const b of document.querySelectorAll("#seatPicker button")) {
  b.addEventListener("click", () => {
    pickedSeat = parseInt(b.dataset.seat, 10);
    document.querySelectorAll("#seatPicker button").forEach((x) => x.classList.toggle("active", x === b));
  });
}

(async function loadBots() {
  try {
    const data = await api("GET", "/api/bots");
    for (const sel of document.querySelectorAll(".bot-select")) {
      for (const b of data.bots) {
        const opt = document.createElement("option");
        opt.value = b.id;
        opt.textContent = b.label;
        sel.appendChild(opt);
      }
    }
  } catch (e) {
    console.error("could not load bot list", e);
  }
})();

$("startPlay").addEventListener("click", async () => {
  const body = { mode: "play", human_seat: pickedSeat, bot: $("playBot").value || null, autoplay: false };
  const seed = $("playSeed").value;
  if (seed) body.match_seed = parseInt(seed, 10);
  try {
    const data = await api("POST", "/api/game", body);
    enterGame("play", data);
  } catch (e) {
    toast(e.message);
  }
});

$("startWatch").addEventListener("click", async () => {
  const body = { mode: "watch", bot: $("watchBot").value || null };
  const seed = $("watchSeed").value;
  if (seed) body.match_seed = parseInt(seed, 10);
  try {
    const data = await api("POST", "/api/game", body);
    enterGame("watch", data);
  } catch (e) {
    toast(e.message);
  }
});

$("startReplay").addEventListener("click", async () => {
  let log;
  try {
    log = JSON.parse($("replayInput").value);
  } catch (e) {
    toast("That doesn't look like a replay log (invalid JSON).");
    return;
  }
  try {
    const data = await api("POST", "/api/replay", { log });
    enterReplay(data);
  } catch (e) {
    toast(e.message);
  }
});

function showScreen(game) {
  $("lobby").classList.toggle("hidden", game);
  $("game").classList.toggle("hidden", !game);
}

function resetTable() {
  app.generation++;
  app.autoplay = false;
  app.busy = false;
  app.shownTrick = [];
  $("trickZone").innerHTML = "";
  $("trickZone").className = "trick-zone";
  $("resultOverlay").classList.add("hidden");
  $("menuOverlay").classList.add("hidden");
  $("playPauseBtn").textContent = "▶ Play";
}

function enterGame(mode, data) {
  resetTable();
  app.mode = mode;
  app.sessionId = data.session_id;
  app.humanSeat = data.human_seat;
  showScreen(true);
  $("scoreboard").classList.remove("hidden");
  $("watchControls").classList.toggle("hidden", mode !== "watch");
  $("replayControls").classList.add("hidden");
  $("menuMeta").textContent = `Players: ${botLabel(data.bot)}`;
  render(data);
  if (mode === "play") runBots();
}

function enterReplay(data) {
  resetTable();
  app.mode = "replay";
  app.sessionId = data.session_id;
  app.humanSeat = null;
  showScreen(true);
  $("watchControls").classList.add("hidden");
  $("replayControls").classList.remove("hidden");
  $("menuMeta").textContent = "Replay";
  renderReplay(data);
}

function botLabel(id) {
  const opt = [...$("playBot").options].find((o) => o.value === id);
  return opt ? opt.textContent : id;
}

// ================================================================ game flow

function isTerminal(state) {
  return state.phase === "TERMINAL" || state.phase === "ABORTED";
}

/** Show a server update, animating a completed trick first if one just finished. */
async function applyUpdate(data) {
  const prev = app.latest;
  const gen = app.generation;
  const prevTricks = prev ? prev.state.completed_tricks.length : 0;
  const tricks = data.state.completed_tricks;
  const sameDeal = prev && prev.state.deal_number === data.state.deal_number && prev.state.phase !== "ABORTED";
  if (sameDeal && tricks.length > prevTricks) {
    const t = tricks[tricks.length - 1];
    renderHands(data.state); // the card just played leaves the hand before the trick is shown
    renderPlates(data.state);
    renderCallBubbles(data.state);
    renderTrick(t.plays, t.winner);
    await sleep(scaled(DELAY.trickShow));
    if (gen !== app.generation) return;
    const zone = $("trickZone");
    zone.classList.add("sweep", `to${pos(t.winner)}`);
    await sleep(scaled(DELAY.trickSweep));
    if (gen !== app.generation) return;
    zone.className = "trick-zone";
    app.shownTrick = [];
  }
  render(data);
  if (data.state.phase === "TERMINAL") {
    await sleep(400);
    if (gen !== app.generation) return;
    showResult(data);
  } else if (data.state.phase === "ABORTED") {
    await sleep(scaled(DELAY.aborted));
    if (gen !== app.generation) return;
    await nextDeal(false); // whoever called us carries on stepping the bots
  }
}

/** Play mode: step the bots one at a time until it is the human's turn. */
async function runBots() {
  const gen = app.generation;
  if (app.busy) return;
  app.busy = true;
  try {
    while (gen === app.generation && app.latest && app.latest.is_bot_turn && !isTerminal(app.latest.state)) {
      const auction = app.latest.state.phase === "AUCTION";
      await sleep(scaled(auction ? DELAY.bid : DELAY.card));
      if (gen !== app.generation) return;
      if (app.mode === "watch" && !app.autoplay) break;
      const data = await api("POST", `/api/game/${app.sessionId}/bot_step`);
      if (gen !== app.generation) return;
      await applyUpdate(data);
    }
  } catch (e) {
    toast(e.message);
  } finally {
    if (gen === app.generation) {
      app.busy = false;
      if (app.latest) renderDock(app.latest);
    }
  }
}

async function humanAction(action) {
  if (app.busy) return;
  app.busy = true;
  renderDock(app.latest); // hide controls straight away
  let data;
  try {
    data = await api("POST", `/api/game/${app.sessionId}/action`, {
      seat: app.latest.state.to_act,
      action,
      autoplay: false,
    });
  } catch (e) {
    app.busy = false;
    renderDock(app.latest);
    toast(e.message);
    return;
  }
  await applyUpdate(data);
  app.busy = false;
  runBots();
}

async function nextDeal(resume = true) {
  $("resultOverlay").classList.add("hidden");
  const gen = app.generation;
  try {
    const q = app.mode === "play" ? "?autoplay=false" : "";
    const data = await api("POST", `/api/game/${app.sessionId}/next_deal${q}`);
    if (gen !== app.generation) return;
    app.shownTrick = [];
    render(data);
    if (resume && (app.mode === "play" || app.autoplay)) runBots();
  } catch (e) {
    toast(e.message);
  }
}

// ================================================================ watch controls

$("stepBtn").addEventListener("click", async () => {
  if (app.busy || !app.latest || isTerminal(app.latest.state)) return;
  app.busy = true;
  try {
    const data = await api("POST", `/api/game/${app.sessionId}/bot_step`);
    await applyUpdate(data);
  } catch (e) {
    toast(e.message);
  }
  app.busy = false;
});
$("playPauseBtn").addEventListener("click", () => {
  app.autoplay = !app.autoplay;
  $("playPauseBtn").textContent = app.autoplay ? "❚❚ Pause" : "▶ Play";
  if (app.autoplay) {
    if (isTerminal(app.latest.state)) nextDeal();
    else runBots();
  }
});
$("speedRange").addEventListener("input", (e) => {
  app.speed = parseInt(e.target.value, 10);
});

// ================================================================ menu

$("menuBtn").addEventListener("click", () => $("menuOverlay").classList.remove("hidden"));
$("resumeBtn").addEventListener("click", () => $("menuOverlay").classList.add("hidden"));
$("quitBtn").addEventListener("click", () => {
  resetTable();
  app.mode = null;
  app.latest = null;
  showScreen(false);
});
$("toggleSideBtn").addEventListener("click", () => {
  const g = $("game");
  const narrow = window.matchMedia("(max-width: 1100px)").matches;
  if (narrow) g.classList.toggle("show-side");
  else g.classList.toggle("no-side");
  const hidden = narrow ? !g.classList.contains("show-side") : g.classList.contains("no-side");
  $("toggleSideBtn").textContent = hidden ? "Show side panel" : "Hide side panel";
  $("menuOverlay").classList.add("hidden");
});
$("copyReplayBtn").addEventListener("click", async () => {
  if (app.mode === "replay") return toast("Already in a replay.", "info");
  try {
    const log = await api("GET", `/api/game/${app.sessionId}/replay_log`);
    await navigator.clipboard.writeText(JSON.stringify(log));
    toast("Replay log of this deal copied to the clipboard.", "info");
  } catch (e) {
    toast(e.message);
  }
});

// ================================================================ bidding controls

$("levelDown").addEventListener("click", () => setLevel(app.level - 1));
$("levelUp").addEventListener("click", () => setLevel(app.level + 1));
$("passBtn").addEventListener("click", () => humanAction({ type: "pass" }));
$("contraBtn").addEventListener("click", () => humanAction({ type: "contra" }));
$("recontraBtn").addEventListener("click", () => humanAction({ type: "recontra" }));

function bidLegal() {
  return app.latest && (app.latest.state.legal_actions || []).find((a) => a.type === "bid");
}
function setLevel(v) {
  const bid = bidLegal();
  if (!bid) return;
  app.level = Math.max(bid.min_level, Math.min(bid.max_level, v));
  $("levelValue").textContent = app.level;
  $("levelDown").disabled = app.level <= bid.min_level;
  $("levelUp").disabled = app.level >= bid.max_level;
  for (const b of document.querySelectorAll(".suit-btn .lvl")) b.textContent = app.level;
}

function buildBidPanel(legal) {
  const has = (t) => legal.find((a) => a.type === t);
  const bid = has("bid");
  const contraPending = !!has("recontra") && !bid;
  $("bidTitle").textContent = contraPending ? "You were doubled — accept or redouble?" : "Your bid";
  $("passBtn").textContent = contraPending ? "Accept" : "Pass";
  $("contraBtn").classList.toggle("hidden", !has("contra"));
  $("recontraBtn").classList.toggle("hidden", !has("recontra"));
  $("bidRow").classList.toggle("hidden", !bid);
  if (!bid) return;
  const box = $("suitButtons");
  box.innerHTML = "";
  for (const t of bid.contract_types) {
    const b = document.createElement("button");
    b.className = "suit-btn" + (RED_SUITS.has(t) ? " red" : "") + (t === "NT" ? " nt" : "");
    b.innerHTML = `<span class="sym">${t === "NT" ? "NT" : SUIT_SYMBOL[t]}</span><span class="lvl"></span>`;
    b.title = t === "NT" ? "No trump" : t;
    b.addEventListener("click", () =>
      humanAction({ type: "bid", level: app.level, contract_type: t, capot: $("bidCapot").checked })
    );
    box.appendChild(b);
  }
  $("bidCapot").checked = bid.forced_capot;
  $("bidCapot").disabled = bid.forced_capot;
  app.level = bid.min_level;
  setLevel(bid.min_level);
}

// ================================================================ rendering

function render(data) {
  app.latest = data;
  const state = data.state;
  renderHud(data);
  renderPlates(state);
  renderHands(state);
  renderBubbles(data);
  renderCenter(state);
  renderSide(data);
  renderDock(data);
}

function renderHud(data) {
  const state = data.state;
  const us = usTeam(), them = 1 - us;
  const scores = data.match_scores || [0, 0];
  const target = data.match_target || 101;
  $("teamNameUs").textContent = teamName(us);
  $("teamNameThem").textContent = teamName(them);
  $("scoreUs").textContent = scores[us];
  $("scoreThem").textContent = scores[them];
  $("targetLabel").textContent = `to ${target}`;
  $("barUs").style.setProperty("--pct", `${Math.min(100, (100 * scores[us]) / target)}%`);
  $("barThem").style.setProperty("--pct", `${Math.min(100, (100 * scores[them]) / target)}%`);
  $("dealLabel").textContent = `Deal ${state.deal_number + 1}`;

  const badge = $("contractBadge");
  const c = state.contract;
  if (!c) {
    badge.classList.add("hidden");
    badge.dataset.key = "";
    return;
  }
  const key = `${c.level}${c.contract_type}${c.doubling}${c.declarer_seat}`;
  badge.classList.remove("hidden");
  if (badge.dataset.key === key) return;
  badge.dataset.key = key;
  const red = RED_SUITS.has(c.contract_type);
  const sym = c.contract_type === "NT" ? "NT" : SUIT_SYMBOL[c.contract_type];
  const dbl = c.doubling === "contra" ? '<span class="tag tag-red">contra</span>' : c.doubling === "recontra" ? '<span class="tag tag-red">recontra</span>' : "";
  const capot = c.capot ? '<span class="tag tag-gold">capot</span>' : "";
  badge.innerHTML =
    `<div class="contract-suit${red ? " red" : ""}${c.contract_type === "NT" ? " nt" : ""}">${sym}</div>` +
    `<div class="contract-text"><div class="contract-main">${contractText(c)}${dbl}${capot}</div>` +
    `<div class="contract-sub">by ${seatName(c.declarer_seat)} · needs ${c.level * 10}+ pts</div></div>`;
}

function renderPlates(state) {
  for (let s = 0; s < 4; s++) {
    const p = pos(s);
    const plate = $(`plate${p}`);
    const toAct = state.to_act === s && !isTerminal(state);
    const isHuman = app.mode === "play" && s === app.humanSeat;
    const team = teamClass(teamOf(s));
    const initial = seatName(s)[0];
    plate.className = "nameplate" + (toAct ? " to-act" : "");
    plate.innerHTML =
      `<div class="avatar ${team}">${isHuman ? "★" : initial}</div>` +
      `<div class="plate-text"><span class="plate-name">${seatName(s)}${toAct && !isHuman ? '<span class="thinking"><i></i><i></i><i></i></span>' : ""}</span>` +
      `<span class="plate-role">${seatRole(s)}</span></div>` +
      (state.dealer === s ? '<div class="dealer-chip" title="Dealer">D</div>' : "");
  }
}

function renderHands(state) {
  const legalCards = new Set((state.legal_actions || []).filter((a) => a.type === "play").map((a) => a.card));
  for (let s = 0; s < 4; s++) {
    const p = pos(s);
    const el = $(`hand${p}`);
    el.innerHTML = "";
    const faceUp = app.mode !== "play" || s === app.humanSeat;
    el.classList.toggle("faces", faceUp && p !== 0);
    const small = p !== 0;
    if (!faceUp) {
      const n = state.hand_sizes ? state.hand_sizes[s] : 0;
      for (let i = 0; i < n; i++) el.appendChild(backEl(true));
      continue;
    }
    const cards = sortHand(app.mode === "play" ? state.my_hand : state.hands[s], state.contract);
    const myTurn = state.to_act === s && state.phase === "PLAY";
    const n = cards.length;
    cards.forEach((cid, i) => {
      const c = cardEl(cid, { small });
      if (myTurn && legalCards.size) {
        if (legalCards.has(cid)) {
          c.classList.add("playable");
          if (app.mode === "play" && !app.busy) c.addEventListener("click", () => humanAction({ type: "play", card: cid }));
        } else c.classList.add("dim");
      }
      if (p === 0) {
        const off = i - (n - 1) / 2;
        c.style.setProperty("--rot", `${off * 3.2}deg`);
        c.style.setProperty("--lift", `${off * off * 1.4}px`);
      }
      el.appendChild(c);
    });
  }
}

function renderBubbles(data) {
  const state = data.state;
  if (state.phase === "PLAY" || state.phase === "TERMINAL") {
    renderCallBubbles(state);
    return;
  }
  app.callsSeen = 0;
  for (const k of Object.keys(bubbleTimers)) {
    clearTimeout(bubbleTimers[k]); // a combination call from the last deal must not linger
    bubbleTimers[k] = null;
  }
  const last = {};
  if (state.phase === "AUCTION" || state.phase === "ABORTED") {
    for (const e of data.bid_log || []) last[e.seat] = e;
  }
  for (let s = 0; s < 4; s++) {
    const b = $(`bubble${pos(s)}`);
    const e = last[s];
    if (!e) {
      b.className = "bubble";
      b.dataset.key = "";
      continue;
    }
    const key = JSON.stringify(e) + (data.bid_log || []).length;
    let html, cls = "bubble show";
    if (e.kind === "pass") { html = "Pass"; cls += " pass"; }
    else if (e.kind === "contra") { html = "Contra!"; cls += " contra"; }
    else if (e.kind === "recontra") { html = "Recontra!"; cls += " contra"; }
    else html = bidHtml(e.level, e.contract_type, e.capot);
    if (b.dataset.html !== html) {
      b.className = "bubble";
      void b.offsetWidth; // restart the pop animation
    }
    b.dataset.key = key;
    b.dataset.html = html;
    b.innerHTML = html;
    b.className = cls;
  }
}

// ---- combination calls (core/announcements.py): class at hand 1, show at hand 2, blot/reblot

const CALL_TEXT = { tierce: "Tierce!", fifty: "Fifty!", hundred: "Hundred!", "4x": "4×!", blot: "Blot!", reblot: "Reblot!" };
const bubbleTimers = {};

function meldText(m) {
  return m.kind === "sequence" ? `${m.label} ${suitHtml(m.suit)}` : m.label;
}
function callHtml(a) {
  if (a.kind === "show") return "Shows " + a.melds.map(meldText).join(" + ");
  return CALL_TEXT[a.kind] || a.kind;
}

/** Pop a bubble for each call made since the last render. */
function renderCallBubbles(state) {
  const calls = state.announcements || [];
  const key = `${app.sessionId}:${state.deal_number}`;
  if (app.callKey !== key) {
    app.callKey = key;
    app.callsSeen = 0;
  }
  for (let s = 0; s < 4; s++) {
    const b = $(`bubble${pos(s)}`);
    if (!b.classList.contains("call")) b.className = "bubble"; // auction calls end with the auction
  }
  for (const a of calls.slice(app.callsSeen)) {
    const b = $(`bubble${pos(a.seat)}`);
    b.innerHTML = callHtml(a);
    b.className = "bubble";
    void b.offsetWidth; // restart the pop animation
    b.className = "bubble show call" + (a.kind === "blot" || a.kind === "reblot" ? " blot" : "");
    clearTimeout(bubbleTimers[a.seat]);
    bubbleTimers[a.seat] = setTimeout(() => {
      b.className = "bubble";
      bubbleTimers[a.seat] = null;
    }, 2600);
  }
  app.callsSeen = calls.length;
  renderShown(state);
}

/** Cards shown at hand 2 stay face up beside their owner while trick 2 is played. */
function renderShown(state) {
  const showing = state.phase === "PLAY" && (state.completed_tricks || []).length === 1;
  for (let p = 0; p < 4; p++) $(`shown${p}`).innerHTML = "";
  if (!showing) return;
  for (const a of state.announcements || []) {
    if (a.kind !== "show") continue;
    const box = $(`shown${pos(a.seat)}`);
    for (const m of a.melds) {
      const g = document.createElement("div");
      g.className = "shown-group";
      g.innerHTML = `<div class="shown-label">${meldText(m)} <b>${m.value}</b></div>`;
      const row = document.createElement("div");
      row.className = "shown-cards";
      m.cards.forEach((cid) => row.appendChild(cardEl(cid, { small: true })));
      g.appendChild(row);
      box.appendChild(g);
    }
  }
}

function renderCombos(state) {
  const calls = state.announcements || [];
  $("combosBlock").classList.toggle("hidden", !calls.length);
  const box = $("combos");
  box.innerHTML = "";
  for (const a of calls) {
    const row = document.createElement("div");
    row.className = "combo-row";
    const who = `<span class="${teamClass(teamOf(a.seat))}">${seatName(a.seat)}</span>`;
    let text;
    if (a.kind === "show") text = "shows " + a.melds.map((m) => `${meldText(m)} <b>${m.value}</b>`).join(", ");
    else if (a.kind === "blot" || a.kind === "reblot") {
      const c = cardInfo(a.card);
      text = `${a.kind === "blot" ? "Blot" : "Reblot"} <span class="${c.red ? "red" : ""}">${c.rank}${c.symbol}</span>${a.kind === "reblot" ? " <b>20</b>" : ""}`;
    } else text = `announces ${CALL_TEXT[a.kind].replace("!", "")}`;
    row.innerHTML = `${who} ${text}`;
    box.appendChild(row);
  }
}

function renderTrick(plays, winner) {
  const zone = $("trickZone");
  const before = new Set(app.shownTrick);
  zone.innerHTML = "";
  for (const [seat, cid] of plays) {
    const p = pos(seat);
    const wrap = document.createElement("div");
    wrap.className = `trick-card p${p}` + (winner === seat ? " winner" : "");
    if (!before.has(cid)) {
      wrap.classList.add("fly");
      wrap.style.setProperty("--fly", `fly${p}`);
    }
    wrap.appendChild(cardEl(cid));
    zone.appendChild(wrap);
  }
  app.shownTrick = plays.map(([, cid]) => cid);
}

function renderCenter(state) {
  const msg = $("centerMsg");
  if (state.phase === "AUCTION") {
    $("trickZone").innerHTML = "";
    app.shownTrick = [];
    const who = state.to_act;
    msg.innerHTML = `Bidding<small>${who === null ? "" : seatName(who) === "You" ? "your call" : seatName(who) + " to call"}</small>`;
    return;
  }
  if (state.phase === "ABORTED") {
    $("trickZone").innerHTML = "";
    msg.innerHTML = "All passed<small>redealing…</small>";
    return;
  }
  msg.innerHTML = "";
  if (!$("trickZone").classList.contains("sweep")) renderTrick(state.current_trick, null);
}

function renderSide(data) {
  renderAuctionGrid(data.bid_log || [], data.state);
  renderCombos(data.state);
  const state = data.state;
  const tricks = state.completed_tricks || [];
  const won = [0, 0], pts = [0, 0];
  for (const t of tricks) {
    won[teamOf(t.winner)] += 1;
    pts[teamOf(t.winner)] += t.points;
  }
  const us = usTeam(), them = 1 - us;
  $("trickStats").innerHTML =
    `<span class="h">Team</span><span class="h n">Tricks</span><span class="h n">Points</span>` +
    `<span class="us">${teamName(us)}</span><span class="n">${won[us]}</span><span class="n">${pts[us]}</span>` +
    `<span class="them">${teamName(them)}</span><span class="n">${won[them]}</span><span class="n">${pts[them]}</span>`;
  $("trickStatsBlock").classList.toggle("hidden", state.phase === "AUCTION");

  const last = tricks[tricks.length - 1];
  $("lastTrickBlock").classList.toggle("hidden", !last);
  if (last) {
    const box = $("lastTrick");
    box.innerHTML = "";
    for (const [seat, cid] of last.plays) {
      const d = document.createElement("div");
      d.className = "lt" + (seat === last.winner ? " win" : "");
      d.appendChild(cardEl(cid));
      const n = document.createElement("span");
      n.textContent = seatName(seat);
      d.appendChild(n);
      box.appendChild(d);
    }
  }
}

function renderAuctionGrid(log, state) {
  const grid = $("auctionGrid");
  const first = log.length ? log[0].seat : state.to_act !== null && state.to_act !== undefined ? state.to_act : (state.dealer + 1) % 4;
  const cols = [0, 1, 2, 3].map((i) => (first + i) % 4);
  let html = "<tr>" + cols.map((s) => `<th class="${teamClass(teamOf(s))}">${seatName(s)}</th>`).join("") + "</tr>";
  if (!log.length) {
    grid.innerHTML = html + `<tr><td colspan="4" class="auction-empty">${state.phase === "AUCTION" ? "No calls yet" : "—"}</td></tr>`;
    return;
  }
  const rows = [];
  let row = new Array(4).fill(null);
  let col = 0;
  for (const e of log) {
    const c = cols.indexOf(e.seat);
    if (c < col) {
      rows.push(row);
      row = new Array(4).fill(null);
    }
    row[c] = e;
    col = c + 1;
    if (col === 4) {
      rows.push(row);
      row = new Array(4).fill(null);
      col = 0;
    }
  }
  if (row.some((x) => x)) rows.push(row);
  const lastEntry = log[log.length - 1];
  rows.forEach((r) => {
    const latest = r.includes(lastEntry);
    html += `<tr class="${latest ? "latest" : ""}">`;
    for (const e of r) {
      if (!e) html += "<td></td>";
      else if (e.kind === "pass") html += `<td class="pass filled">pass</td>`;
      else if (e.kind === "contra") html += `<td class="dbl filled">contra</td>`;
      else if (e.kind === "recontra") html += `<td class="dbl filled">recontra</td>`;
      else html += `<td class="filled">${e.level}${suitHtml(e.contract_type)}${e.capot ? "c" : ""}</td>`;
    }
    html += "</tr>";
  });
  grid.innerHTML = html;
}

function renderDock(data) {
  const state = data.state;
  const myTurn = app.mode === "play" && state.to_act === app.humanSeat && !isTerminal(state) && !app.busy;
  const bidPanel = $("bidPanel");
  const hint = $("turnHint");
  const auction = state.phase === "AUCTION";
  if (myTurn && auction) {
    if (bidPanel.classList.contains("hidden") || bidPanel.dataset.key !== String(data.bid_log.length)) {
      buildBidPanel(state.legal_actions || []);
      bidPanel.dataset.key = String(data.bid_log.length);
    }
    bidPanel.classList.remove("hidden");
  } else {
    bidPanel.classList.add("hidden");
  }
  hint.classList.toggle("hidden", !(myTurn && state.phase === "PLAY"));
  hint.textContent = "Your turn — play a card";
  if (app.mode === "play") renderHands(state); // refresh card click handlers with the busy flag
}

// ================================================================ result overlay

function showResult(data) {
  const r = data.state.result;
  const c = data.state.contract;
  if (!r || !c) return;
  const us = usTeam(), them = 1 - us;
  const att = r.attackers_team;
  const attName = teamName(att);
  let title, cls;
  if (app.mode === "play") {
    const weDeclared = att === us;
    const good = weDeclared === r.made;
    title = r.made ? (weDeclared ? "Contract made!" : "They made it") : weDeclared ? "Contract failed" : "Set them!";
    cls = good ? "good" : "bad";
  } else {
    title = r.made ? "Contract made" : "Contract failed";
    cls = r.made ? "good" : "bad";
  }
  const row = (team) => {
    const isAtt = team === att;
    const cards = isAtt ? r.cards_attackers : r.cards_defenders;
    const combo = isAtt ? r.combo_attackers : r.combo_defenders;
    const score = isAtt ? r.score_attackers : r.score_defenders;
    return `<tr><td class="${teamClass(team)}">${teamName(team)}${isAtt ? ' <span class="tag tag-gold">declarer</span>' : ""}</td>` +
      `<td>${cards}</td><td>${combo}</td><td class="gain">+${score}</td></tr>`;
  };
  const shutout = r.attackers_took_all_tricks ? `${attName} took every trick` : r.defenders_took_all_tricks ? "Defenders took every trick" : "";
  const scores = data.match_scores;
  let matchHtml = `<div class="result-match"><span class="${teamClass(us)}">${teamName(us)}</span><b>${scores[us]}</b><span>–</span><b>${scores[them]}</b><span class="${teamClass(them)}">${teamName(them)}</span></div>`;
  let kicker = `Deal ${data.state.deal_number + 1}`;
  let button = `<button class="btn btn-gold btn-big" id="nextDealBtn">Next deal</button>`;
  if (data.match_finished) {
    const winner = data.match_winner;
    kicker = "Match over";
    if (app.mode === "play") {
      title = winner === us ? "Victory!" : "Defeat";
      cls = winner === us ? "gold" : "bad";
    } else {
      title = `${teamName(winner)} win`;
      cls = "gold";
    }
    button = `<button class="btn btn-gold btn-big" id="newMatchBtn">Back to title</button>`;
  }
  $("resultCard").innerHTML =
    `<div class="result-kicker">${kicker}</div>` +
    `<div class="result-title ${cls}">${title}</div>` +
    `<div class="result-contract">${attName} bid ${bidHtml(c.level, c.contract_type, c.capot)}` +
    `${c.doubling !== "none" ? ` <span class="tag tag-red">${c.doubling}</span>` : ""} · needed ${c.level * 10}+${shutout ? " · " + shutout : ""}</div>` +
    `<table class="result-table"><tr><th></th><th>Cards</th><th>Combos</th><th>Score</th></tr>${row(us)}${row(them)}</table>` +
    scoredCombosHtml(data.state) +
    matchHtml + button;
  $("resultOverlay").classList.remove("hidden");
  const next = $("nextDealBtn");
  if (next) next.addEventListener("click", () => nextDeal());
  const nm = $("newMatchBtn");
  if (nm) nm.addEventListener("click", () => $("quitBtn").click());
  if (app.mode === "watch" && app.autoplay && !data.match_finished) {
    const gen = app.generation;
    setTimeout(() => {
      if (gen === app.generation && app.autoplay && !$("resultOverlay").classList.contains("hidden")) nextDeal();
    }, scaled(3000));
  }
}

function scoredCombosHtml(state) {
  const items = [];
  for (const a of state.announcements || []) {
    if (a.kind === "show") for (const m of a.melds) items.push([a.seat, `${meldText(m)}`, m.value]);
    if (a.kind === "reblot") items.push([a.seat, "Blot-Reblot", 20]);
  }
  if (!items.length) return '<div class="result-combos none">No combinations</div>';
  return '<div class="result-combos">' + items
    .map(([seat, txt, v]) => `<div><span class="${teamClass(teamOf(seat))}">${seatName(seat)}</span> ${txt} <b>+${v}</b></div>`)
    .join("") + "</div>";
}

// ================================================================ replay

$("replayNext").addEventListener("click", async () => replayGo("step"));
$("replayPrev").addEventListener("click", async () => replayGo(Math.max(0, app.latest.index - 1)));
$("replayStart").addEventListener("click", async () => replayGo(0));
$("replayEnd").addEventListener("click", async () => replayGo(app.latest.total_actions));

async function replayGo(target) {
  try {
    const data =
      target === "step"
        ? await api("POST", `/api/replay/${app.sessionId}/step`)
        : await api("POST", `/api/replay/${app.sessionId}/goto`, { index: target });
    renderReplay(data);
  } catch (e) {
    toast(e.message);
  }
}

function renderReplay(data) {
  app.latest = data;
  const state = data.state;
  $("replayPos").textContent = `${data.index} / ${data.total_actions}`;
  $("scoreboard").classList.add("hidden");
  renderHud({ state, match_scores: [0, 0], match_target: 0 });
  renderPlates(state);
  renderHands(state);
  for (let p = 0; p < 4; p++) $(`bubble${p}`).className = "bubble";
  app.callsSeen = (state.announcements || []).length; // no bubbles while scrubbing
  renderShown(state);
  renderCenter(state);
  renderSide({ bid_log: [], state });
  $("bidPanel").classList.add("hidden");
  $("turnHint").classList.add("hidden");
}

// keyboard: space = step/next in watch & replay, Esc = menu
document.addEventListener("keydown", (e) => {
  if ($("game").classList.contains("hidden")) return;
  if (e.key === "Escape") $("menuOverlay").classList.toggle("hidden");
  if (e.key === " " && e.target === document.body) {
    e.preventDefault();
    if (app.mode === "replay") $("replayNext").click();
    else if (app.mode === "watch") $("stepBtn").click();
  }
});
