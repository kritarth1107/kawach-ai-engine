(async () => {
  // Run on https://www.swiggy.com/instamart/cart after instamart_cart.js: the app loads the server cart and sets its own session
  // headers (sessionStorage 'headers', fresh for ~15 min). Returns the cart as the store sees it and whether cash on delivery is
  // offered. The headers are used here only; just the yes/no and the cart lines leave the page.
  for (let i = 0; i < 120 && !sessionStorage.getItem('headers'); i++) await new Promise(r => setTimeout(r, 100));
  const out = { headers: !!sessionStorage.getItem('headers') };
  const A = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
  const rot = (t, e) => t.replace(/[a-zA-Z0-9]/g, c => A[(A.indexOf(c) + e + A.length) % A.length]);
  const r5 = () => { const f = (x, n = 0) => n > 10 ? 99999 : (x < 1e4 && (x *= 10), x > 1e4 ? x : f(x, n + 1)); return f(Math.round(Math.random() * 1e5)); };
  const matcher = () => rot(String(r5()) + String(Date.now()) + String(r5()), 7);
  try {
    const r1 = await fetch('/api/instamart/checkout/v2/cart?pageType=INSTAMART_CART', { credentials: 'same-origin', headers: { 'Content-Type': 'application/json', matcher: matcher() } });
    const d = (((await r1.json()).data || {}).data) || {};
    out.items = {}; for (const i of d.items || []) out.items[i.itemId] = Number(i.quantity);
    out.total = Number((d.bill || {}).toPay || 0); out.address_id = String(d.addressId || '');
    const parse = x => { try { return JSON.parse(x || '{}') || {}; } catch (e) { return {}; } };
    const key = (crypto.randomUUID && crypto.randomUUID()) || String(Date.now());
    const ph = { ...parse(localStorage.getItem('auth_headers')), ...parse(sessionStorage.getItem('headers')), 'content-type': 'application/json',
      'x-client-id': 'web-payment', platform: 'dweb', 'client-id': 'dweb', m_id: 'DASH', marketplacecategory: 'instamart', marketplacebusinessline: 'dash',
      marketplaceid: '1', 'x-web-checkout-flow': 'payment', 'x-checkout-webview': 'dweb', cartid: String(d.cartId || ''), cartkey: key, cartaddressid: out.address_id };
    const r3 = await fetch('/api/v3/payment/get-payment-options', { method: 'POST', credentials: 'same-origin', headers: ph, body: JSON.stringify({
      address_id: '', card_details: false, is_user_cred_eligible: false, is_user_eligible_for_single_click: false, without_split_pay: false,
      split_pay_methods: 'default', is_gifting_flow: false, is_inapp_upi_eligible: false, bill_to_company_toggle: false, paymentLinkId: key }) });
    const j3 = await r3.json().catch(() => ({}));
    out.cod_available = j3.statusCode === 0 ? !!((j3.data || {}).codEnabled) : null;
  } catch (e) { out.error = String(e && e.message || e); }
  return out;
})()
