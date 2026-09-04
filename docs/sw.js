self.addEventListener("push", event => {
  let data = {};
  try { data = event.data?.json() || {}; } catch { data = { title: "New model opportunity", body: event.data?.text() || "Open the dashboard for details." }; }
  event.waitUntil(self.registration.showNotification(data.title || "New model opportunity", {
    body: data.body || "A prediction now differs meaningfully from the market.",
    icon: "icon.svg", badge: "icon.svg", tag: data.id || "model-opportunity", renotify: true,
    data: { url: data.url || "https://mvnathan.github.io/wnba-data/" },
  }));
});

self.addEventListener("notificationclick", event => {
  event.notification.close();
  const url = event.notification.data?.url || "https://mvnathan.github.io/wnba-data/";
  event.waitUntil(clients.matchAll({ type: "window", includeUncontrolled: true }).then(windows => {
    const existing = windows.find(client => client.url.startsWith(url));
    return existing ? existing.focus() : clients.openWindow(url);
  }));
});
