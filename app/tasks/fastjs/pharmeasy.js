(async () => {
  const q = %(q)s, pin = %(pincode)s;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const cookie = (n) => { const m = document.cookie.match(new RegExp('(?:^|; )' + n + '=([^;]*)')); return m ? decodeURIComponent(m[1]) : null; };
  // Let the page finish its own start-up first: about 0.5 s after the HTML arrives it writes its default X-Pincode
  // (Mumbai) and X-Feature-Flags cookies and looks its pincode up. A pincode written before that gets overwritten,
  // and calls made without X-Feature-Flags return different (higher) prices.
  for (let i = 0; i < 50; i++) {
    if (cookie('X-Pincode') && cookie('X-Feature-Flags') && performance.getEntriesByType('resource').some((e) => e.name.includes('/api/app/fetchPincodeDetails'))) break;
    await sleep(150);
  }
  const H = {'x-phone-platform': 'web'};
  if (pin) H['X-Pincode'] = pin; // read by the server only when the cookie is missing; the cookie wins
  // Same as "Choose your Location" -> Check: look the pincode up; the site then keeps it in two cookies that every
  // later API call (search prices/stock, delivery date) reads.
  let serviceable = null, cityId = null;
  if (pin) {
    const pr = await fetch('/api/app/fetchPincodeDetails?pincode=' + encodeURIComponent(pin), {headers: H});
    let pj = null; try { pj = await pr.json(); } catch (e) {}
    const ca = pj && pj.data && pj.data.cityAttributes;
    if (ca) { serviceable = !!ca.isMedicine; cityId = ca.id; } else if (pr.ok) serviceable = false;
  }
  const setPin = () => {
    if (!pin || cityId == null) return;
    document.cookie = 'X-Pincode=' + encodeURIComponent(pin) + '; path=/; max-age=31536000';
    document.cookie = 'X-Default-City=' + cityId + '; path=/; max-age=31536000';
  };
  const num = (s) => (s == null || s === '' ? null : (isNaN(+s) ? s : +s));
  const run = async () => {
    setPin();
    const sr = await fetch('/api/search/postSearch/?highMarginOnly=false&intent_id&page=1&q=' + encodeURIComponent(q),
      {method: 'POST', headers: Object.assign({'content-type': 'application/json'}, H), body: '[]'});
    if (!sr.ok) return {status: sr.status, serviceable, eta: null, products: []};
    const sj = await sr.json();
    const list = ((sj.data && sj.data.products) || []).filter((p) => p && p.productId && p.name && (p.mrpDecimal || p.salePriceDecimal));
    const products = list.map((p) => ({
      name: p.name, pack: p.measurementUnit || null,
      price: num(p.salePriceDecimal || p.mrpDecimal), mrp: num(p.mrpDecimal),
      available: !!(p.productAvailabilityFlags && p.productAvailabilityFlags.isAvailable),
      rx_required: p.isRxRequired === 1 || p.isRxRequired === true, id: String(p.productId),
    }));
    // Search results carry no date; the product page asks for it per product (pincode from the cookie).
    let eta = null;
    const first = list.find((p) => p.productAvailabilityFlags && p.productAvailabilityFlags.isAvailable);
    if (first && serviceable !== false) {
      try {
        const er = await fetch('/api/otc/fetchOtcEdd/' + first.productId, {headers: H});
        const e = er.ok ? (await er.json()).edd : null;
        if (e && e.time) eta = ((e.text || '').trim() + ' ' + e.time).trim();
      } catch (e) {}
    }
    return {status: sr.status, serviceable, eta, products};
  };
  let out = await run();
  if (pin && cityId != null && cookie('X-Pincode') !== pin) out = await run(); // the page reset it mid-way: redo once
  return out;
})()
