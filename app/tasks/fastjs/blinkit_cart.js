(async () => {
  // Logged-in Blinkit: build the server cart for the picked products at the saved address and read the bill. Never orders.
  // Headers are the ones Blinkit's own page just sent (app version, device, session, login tokens).
  const H = Object.assign(JSON.parse(%(page_headers)s), {'content-type': 'application/json'});
  if (!H.access_token) return {status: 401, logged_in: false};
  const j = (k) => { try { return JSON.parse(localStorage.getItem(k)); } catch { return null; } };
  const loc = (j('location') || {}).coords || {};
  const items = JSON.parse(%(items)s);  // [{product_id, quantity}]
  const addressId = Number(%(address_id)s) || loc.addressId;
  const r = await fetch('/v5/carts', {method: 'POST', headers: H, body: JSON.stringify({items, address_id: addressId, promo_codes: ['']})});
  const text = await r.text();
  if (!r.ok) return {status: r.status, logged_in: true, error: text.slice(0, 300)};
  return {status: r.status, logged_in: true, address_id: addressId, raw: text};
})()
