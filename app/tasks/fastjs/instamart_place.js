(async () => {
  // Swiggy Instamart: place the cart the family confirmed, cash on delivery, no clicks (recorded from the web app,
  // order lab 2026-10-10: POST /api/v3/instamart/checkout/order, then POST /api/v3/checkout/confirm/order).
  // First the server cart is read again: the same items and quantities, the same address, and a total no higher than the
  // one confirmed (+₹1). Anything else → nothing is sent. out.sent says whether the order call left the page.
  const want = JSON.parse(%(want)s);   // {items: {item_id: qty}, total, address_id}
  const out = { placed: false, sent: false };
  // run on /instamart/cart: the app sets its session headers there (sessionStorage 'headers'); the order call needs them
  for (let i = 0; i < 120 && !sessionStorage.getItem('headers'); i++) await new Promise(r => setTimeout(r, 100));
  if (!sessionStorage.getItem('headers')) { out.problem = 'no session headers on the cart page'; return out; }
  const A = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
  const rot = (t, e) => t.replace(/[a-zA-Z0-9]/g, c => A[(A.indexOf(c) + e + A.length) % A.length]);
  const r5 = () => { const f = (x, n = 0) => n > 10 ? 99999 : (x < 1e4 && (x *= 10), x > 1e4 ? x : f(x, n + 1)); return f(Math.round(Math.random() * 1e5)); };
  const matcher = () => rot(String(r5()) + String(Date.now()) + String(r5()), 7);
  try {
    // 1) the cart as it is now
    const r1 = await fetch('/api/instamart/checkout/v2/cart?pageType=INSTAMART_CART', { credentials: 'same-origin', headers: { 'Content-Type': 'application/json', matcher: matcher() } });
    if (!r1.ok) { out.problem = 'cart read HTTP ' + r1.status; return out; }
    const d = (((await r1.json()).data || {}).data) || {};
    const bill = d.bill || {};
    const got = {};
    for (const i of d.items || []) { if (i.outOfStock) { out.problem = 'out_of_stock: ' + i.name; return out; } got[i.itemId] = Number(i.quantity); }
    const same = Object.keys(want.items).length === Object.keys(got).length && Object.entries(want.items).every(([k, q]) => got[k] === Number(q));
    const total = Number(bill.toPay || 0);
    out.total = total;
    if (!same) { out.problem = 'cart_changed'; out.cart = got; return out; }
    if (String(d.addressId) !== String(want.address_id)) { out.problem = 'address_changed'; return out; }
    if (!(total > 0) || total > Number(want.total) + 1) { out.problem = 'total_changed'; return out; }
    if (d.unavailableErrorMsg) { out.problem = 'unavailable: ' + d.unavailableErrorMsg; return out; }
    const loc = d.location || {};
    const lat = String(loc.lat || loc.latitude || ''), lng = String(loc.lng || loc.longitude || '');
    // the page's own payment headers (session ids stay inside the page)
    const parse = x => { try { return JSON.parse(x || '{}') || {}; } catch (e) { return {}; } };
    const ph = { ...parse(localStorage.getItem('auth_headers')), ...parse(sessionStorage.getItem('headers')), 'content-type': 'application/json',
      'x-client-id': 'web-payment', platform: 'dweb', 'client-id': 'dweb', m_id: 'DASH', marketplacecategory: 'instamart', marketplacebusinessline: 'dash',
      marketplaceid: '1', 'x-web-checkout-flow': 'payment', 'x-checkout-webview': 'dweb', 'x-origin-id': 'dweb', cartid: String(d.cartId || ''),
      cartaddressid: String(d.addressId), lat, lng, latitude: lat, longitude: lng };
    if (!ph.tid && sessionStorage.getItem('tid')) ph.tid = sessionStorage.getItem('tid');
    const meta = { address_id: String(d.addressId), payment_cod_method: 'Cash', order_comments: '', force_validate_coupon: false, transaction_amount: total,
      selected_split_pay_amounts: {}, twid_rewards_meta: {}, lat, lng, paymentAmount: String(total),
      order_meta: { ordered_from: 'instamart', transaction_type: 'PRE_PAYMENT' }, payment_method_meta: {} };
    const body = { force_validate_coupon: false, transaction_amount: total, selected_split_pay_amounts: {}, twid_rewards_meta: {}, lat, lng,
      addressId: String(d.addressId), paymentMethod: 'Cash', useJuspayNative: false, paymentReturnUrl: 'https://www.swiggy.com/instamart/payment/payment-received',
      orderComments: '', paymentAmount: String(total), address_id: String(d.addressId), payment_cod_method: 'Cash', order_comments: '', payment_type: 'PRE_PAYMENT',
      payment_info: { payment_method: 'Cash', order_context: 'ORDER_JOB', payment_type: 'PRE_PAYMENT', emi_info: {}, metadata: JSON.stringify(meta), transaction_amount: total },
      cartType: 'INSTAMART' };
    // 2) the order (sent once)
    out.sent = true;
    const r2 = await fetch('/api/v3/instamart/checkout/order?cartType=INSTAMART', { method: 'POST', credentials: 'same-origin', headers: ph, body: JSON.stringify(body) });
    const j2 = await r2.json().catch(() => ({}));
    const order = (((j2.data || {}).orders) || [])[0] || {};
    const job = ((order.order_jobs) || [])[0] || {};
    const pay = ((job.payment_info) || [])[0] || {};
    out.order_id = String(order.order_id || (j2.data || {}).order_group_id || '');
    out.status = job.status || '';
    if (!r2.ok || !out.order_id) { out.problem = 'order HTTP ' + r2.status + ' ' + String(j2.statusMessage || '').slice(0, 120); return out; }
    try { const m = JSON.parse(job.metadata || '{}'); if (m.slaMin) out.eta = Math.round(m.slaMin / 60) + '-' + Math.round((m.slaMax || m.slaMin) / 60) + ' min'; } catch (e) {}
    // 3) confirm it (cash: the transaction is just recorded)
    const r3 = await fetch('/api/v3/checkout/confirm/order', { method: 'POST', credentials: 'same-origin', headers: ph,
      body: JSON.stringify({ transactionId: String(pay.transaction_id || ''), orderId: out.order_id, payment_transaction_id: String(pay.transaction_id || ''), order_id: out.order_id }) });
    const j3 = await r3.json().catch(() => ({}));
    const job3 = (((((j3.data || {}).orders) || [])[0] || {}).order_jobs || [])[0] || {};
    out.status = job3.status || out.status;
    out.placed = /PLACED|CONFIRMED/i.test(out.status);
    // the app's cached cart is now stale
    await new Promise(ok => { try { const rq = indexedDB.open('keyval-store'); rq.onsuccess = () => { const db = rq.result; if (![...db.objectStoreNames].includes('keyval')) { db.close(); return ok(); }
      const tx = db.transaction(['keyval'], 'readwrite'); tx.objectStore('keyval').delete('instacart'); tx.oncomplete = () => { db.close(); ok(); }; tx.onerror = () => { db.close(); ok(); }; }; rq.onerror = () => ok(); } catch (e) { ok(); } });
  } catch (e) { out.error = String(e && e.message || e); }
  return out;
})()
