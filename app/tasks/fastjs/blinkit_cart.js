(async () => {
  // Logged-in Blinkit: build the server cart for the picked products at the saved address and read the bill. Never orders.
  // Headers are the ones Blinkit's own page just sent (app version, device, session, login tokens).
  const H = Object.assign(JSON.parse(%(page_headers)s), {'content-type': 'application/json'});
  if (!H.access_token) return {status: 401, logged_in: false};
  const j = (k) => { try { return JSON.parse(localStorage.getItem(k)); } catch { return null; } };
  const loc = (j('location') || {}).coords || {};
  const items = JSON.parse(%(items)s);  // [{product_id, quantity}]
  const local = JSON.parse(%(local)s || '[]');  // [{product_id, price, mrp, unit, group_id, image_url, quantity}] as the app stores them
  const addressId = Number(%(address_id)s) || loc.addressId;
  const r = await fetch('/v5/carts', {method: 'POST', headers: H, body: JSON.stringify({items, address_id: addressId, promo_codes: ['']})});
  const text = await r.text();
  if (!r.ok) return {status: r.status, logged_in: true, error: text.slice(0, 300)};
  // The web app shows and checks out its own copy of the cart (localStorage 'cart'), syncing it to the server only at checkout
  // (lab 2026-10-10: the page showed an empty cart next to the server cart built here). Write the same items there.
  let local_written = false;
  try {
    const cartId = String((JSON.parse(text) || {}).cart_id || '');
    if (local.length && cartId) {
      const cur = j('cart') || {};
      const its = {};
      for (const l of local) its[String(l.product_id)] = {product: {product_id: Number(l.product_id), price: l.price, image_url: l.image_url || '', unit: l.unit,
        mrp: l.mrp, group_id: l.group_id}, quantity: Number(l.quantity || 1)};
      const next = Object.assign({}, cur, {items: its, count: local.reduce((t, l) => t + Number(l.quantity || 1), 0),
        total: local.reduce((t, l) => t + Number(l.price || 0) * Number(l.quantity || 1), 0), uniqueSkuInCart: local.length, id: cartId, cart_state: 'valid'});
      localStorage.setItem('cart', JSON.stringify(next));
      local_written = true;
    }
  } catch (e) {}
  return {status: r.status, logged_in: true, address_id: addressId, raw: text, local_written};
})()
