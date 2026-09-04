const PUSH_API = "https://wnba-live-dashboard.mvnathan.workers.dev";

function decodeApplicationKey(value) {
  const padding = "=".repeat((4 - value.length % 4) % 4);
  const raw = atob((value + padding).replaceAll("-", "+").replaceAll("_", "/"));
  return Uint8Array.from(raw, character => character.charCodeAt(0));
}

async function pushRegistration() {
  if (!("serviceWorker" in navigator) || !("PushManager" in window)) throw new Error("Push notifications are not supported in this browser.");
  return navigator.serviceWorker.register("sw.js", { scope: "./" });
}

async function currentSubscription() {
  const registration = await pushRegistration();
  return registration.pushManager.getSubscription();
}

async function enablePush() {
  const permission = await Notification.requestPermission();
  if (permission !== "granted") throw new Error("Notification permission was not granted.");
  const registration = await pushRegistration();
  const keyResponse = await fetch(`${PUSH_API}/push/public-key`, { cache: "no-store" });
  if (!keyResponse.ok) throw new Error("Could not load the notification key.");
  const { publicKey } = await keyResponse.json();
  let subscription = await registration.pushManager.getSubscription();
  if (!subscription) subscription = await registration.pushManager.subscribe({ userVisibleOnly: true, applicationServerKey: decodeApplicationKey(publicKey) });
  const response = await fetch(`${PUSH_API}/push/subscribe`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(subscription) });
  if (!response.ok) throw new Error("Could not save the notification subscription.");
  await registration.showNotification("Model alerts enabled", { body: "You’ll be notified when a model meaningfully disagrees with the market.", icon: "icon.svg", badge: "icon.svg", tag: "push-enabled" });
}

async function disablePush() {
  const subscription = await currentSubscription();
  if (!subscription) return;
  await fetch(`${PUSH_API}/push/unsubscribe`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ endpoint: subscription.endpoint }) });
  await subscription.unsubscribe();
}

async function mountPushControl() {
  const button = document.createElement("button");
  button.className = "push-toggle";
  button.type = "button";
  button.style.cssText = "position:fixed;right:12px;bottom:12px;z-index:60;border:1px solid rgba(101,224,163,.55);border-radius:999px;background:#101b17;color:#f4fbf7;padding:10px 14px;font:800 13px Inter,system-ui;box-shadow:0 8px 30px rgba(0,0,0,.35);cursor:pointer";
  document.body.appendChild(button);
  async function update() {
    try { button.textContent = await currentSubscription() ? "Alerts on" : "Enable alerts"; }
    catch { button.textContent = "Alerts unavailable"; button.disabled = true; }
  }
  button.addEventListener("click", async () => {
    button.disabled = true; button.textContent = "Updating…";
    try { if (await currentSubscription()) await disablePush(); else await enablePush(); }
    catch (error) { window.alert(error.message + (/iPhone|iPad/i.test(navigator.userAgent) ? " On iPhone or iPad, first add this site to your Home Screen." : "")); }
    button.disabled = false; await update();
  });
  await update();
}

mountPushControl();
