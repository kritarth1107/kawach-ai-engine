(async () => {
  // Swiggy Instamart cart on the logged-in account, no clicks. Run inside a tab on https://www.swiggy.com/instamart.
  // 1) the family place among the account's saved addresses (pincode + flat/first line), 2) select it (that picks the
  // serving store), 3) sync the cart to exactly these items (recorded from the web app: actionType SYNC_CART replaces the
  // cart), 4) read the bill and whether cash on delivery is offered. Never places anything.
  const items = JSON.parse(%(items)s);          // [{product_id, spin, item_id, qty}]
  const place = JSON.parse(%(place)s);          // {pincode, line1, full}
  const out = { logged_in: false };
  for (let i = 0; i < 80 && !window.___INITIAL_STATE___; i++) await new Promise(r => setTimeout(r, 100));
  const A = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
  const rot = (t, e) => t.replace(/[a-zA-Z0-9]/g, c => A[(A.indexOf(c) + e + A.length) % A.length]);
  const r5 = () => { const f = (x, n = 0) => n > 10 ? 99999 : (x < 1e4 && (x *= 10), x > 1e4 ? x : f(x, n + 1)); return f(Math.round(Math.random() * 1e5)); };
  const matcher = () => rot(String(r5()) + String(Date.now()) + String(r5()), 7);
  const waf = window.AwsWafIntegration;
  const F = waf && typeof waf.fetch === 'function' ? waf.fetch.bind(waf) : fetch;
  const call = (url, body) => F(url, { method: body ? 'POST' : 'GET', credentials: 'same-origin', ...(body ? { body: JSON.stringify(body) } : {}),
    headers: { 'Content-Type': 'application/json', 'matcher': matcher() } });
  try {
    const st = window.___INITIAL_STATE___ || {};
    const user = st.user || {};
    const saved = Array.isArray(user.addresses) ? user.addresses : [];
    if (!Object.keys(user).length || (!saved.length && !user.customer_id && !user.mobile)) return out;
    out.logged_in = true;
    // 1) the family place among the saved addresses
    const norm = s => String(s || '').toLowerCase().replace(/[^a-z0-9]/g, '');
    const keys = String(place.line1 || '').split(',').map(s => norm(s)).filter(k => k.length >= 3 || /[a-z]/.test(k) && /\d/.test(k)).slice(0, 2);
    const textOf = a => [a.display_address, (a.actual_address || {}).address_line, (a.actual_address || {}).address_line_2, (a.actual_address || {}).formatted_address].join(' ');
    const pin = a => (textOf(a).match(/\b\d{6}\b/) || [])[0] || String((a.actual_address || {}).pincode || '');
    const hit = saved.find(a => pin(a) === String(place.pincode) && (!keys.length || keys.some(k => norm(textOf(a)).includes(k))));
    out.addresses_seen = saved.length;
    if (!hit) { out.problem = 'address_missing'; return out; }
    const loc = (hit.actual_address || {}).location || hit.location || {};
    const lat = Number(loc.lat || loc.latitude || hit.lat), lng = Number(loc.lng || loc.longitude || hit.lng);
    const addressId = String(hit.id || hit.address_id || hit.addressId);
    const label = String(hit.annotation || hit.tag || hit.address_tag || '').trim();
    out.address_used = (label ? label + ': ' : '') + String(hit.display_address || place.full).replace(/\s+/g, ' ').trim().slice(0, 160);
    // 2) select it: the serving store comes back
    const r1 = await call('/api/instamart/home/select-location/v2', { data: { lat, lng, address: hit.display_address || place.full, addressId,
      annotation: hit.annotation || hit.tag || '', name: (hit.actual_address || {}).contact_name || '', clientId: 'INSTAMART-APP' } });
    if (!r1.ok) { out.status = r1.status; out.error = 'select-location HTTP ' + r1.status; return out; }
    const j1 = await r1.json();
    const cfg = ((((j1.data || {}).configs || {}).IM_PAGE_CONFIGS || {}).configInfo || [])[0];
    const pods = ((cfg || {}).card || {}).podDetailsList || [];
    const prim = pods.find(p => p.priority === 'PRIORITY_PRIMARY') || pods[0];
    if (!prim || ((cfg || {}).card || {}).serviceabilityStatus !== 'SERVICEABLE') { out.problem = 'not_serviceable'; return out; }
    const storeId = Number(prim.podId);
    // 3) the cart: exactly these items
    const body = { data: { items: items.map(i => ({ quantity: Number(i.qty || 1), productId: i.product_id, spin: i.spin, itemId: i.item_id,
        meta: { type: 'structure', storeId }, serviceLine: 'INSTAMART' })),
      cartMetaData: { contactlessDelivery: false, deliveryType: 'INSTANT', owner: 'APP', preferredAddressId: addressId, ageConsentProvided: false,
        useGiftBagPackaging: false, useReusablePackaging: false, incognitoCart: false, includeConsents: ['PHARMA'], primaryStoreId: storeId, storeIds: [storeId] },
      cartType: 'INSTAMART' }, source: 'userInitiated', cartAnalyticsMetaInfo: { actionType: 'SYNC_CART' } };
    const r2 = await call('/api/instamart/checkout/v2/cart?pageType=INSTAMART_CART', body);
    out.status = r2.status;
    if (!r2.ok) { out.error = 'cart HTTP ' + r2.status; return out; }
    const j2 = await r2.json();
    const d = (j2.data || {}).data || {};
    const bill = d.bill || {};
    out.cart_address_id = String(d.addressId || '');
    out.address_matches = out.cart_address_id === addressId;
    out.lines = (d.items || []).map(i => ({ name: i.name, qty: Number(i.quantity), item_id: i.itemId,
      price: Number(((((i.metadata || {}).variations || [])[0] || {}).price || {}).offer_price || 0) * Number(i.quantity),
      available: !i.outOfStock && (((i.metadata || {}).in_stock) !== false) }));
    // each charge after its discount (delivery is free above a minimum: value 30, discount 30)
    const after = c => Number(c.value || 0) - (((c.ctx || {}).chargesBreakup || []).reduce((t, b) => t + Number(b.discValue || 0), 0));
    out.fees = (bill.charges || []).map(c => ({ label: (c.ctx || {}).displayName, value: after(c) })).filter(c => c.value >= 0.5);
    if (Number(bill.deliveryFeeAfterDiscount) === 0) out.fees = out.fees.filter(c => !/delivery/i.test(c.label || ''));
    out.item_total = Number(bill.itemTotal || 0);
    out.total = Number(bill.toPay || 0);
    out.unavailable = d.unavailableErrorMsg || '';
    out.eta = ((d.deliveryOptions || [])[0] || {}).title || '';
    out.cart_id = String(d.cartId || '');
    out.address_id = addressId;
    // The web app keeps its own copy of the cart (IndexedDB keyval-store/instacart) and pushes it back to the server when a page
    // loads (lab 2026-10-10: the cart page re-synced an old Maggi cart over the one built here). Drop that copy so pages read
    // the server cart.
    out.cache_cleared = await new Promise(ok => { try { const rq = indexedDB.open('keyval-store'); rq.onsuccess = () => { const db = rq.result;
      if (![...db.objectStoreNames].includes('keyval')) { db.close(); return ok(false); }
      const tx = db.transaction(['keyval'], 'readwrite'); tx.objectStore('keyval').delete('instacart'); tx.oncomplete = () => { db.close(); ok(true); };
      tx.onerror = () => { db.close(); ok(false); }; }; rq.onerror = () => ok(false); } catch (e) { ok(false); } });
    try { sessionStorage.setItem('IS_IM_CART_CACHE_EXPIRED', 'true'); } catch (e) {}
    // 4) cash on delivery is read on the cart page (instamart_cod.js): only there does the app set its session headers.
  } catch (e) { out.error = String(e && e.message || e); }
  return out;
})()
