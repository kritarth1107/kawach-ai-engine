(async () => {
  const q = %(q)s, pin = %(pincode)s, lat = %(lat)s, lon = %(lon)s;
  const GW = 'https://apigateway.apollo247.in';
  const fail = (status) => ({status, serviceable: null, eta: null, products: []});
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  // Guest token: on load the page fetches a public token and keeps it in localStorage for 12 h. Reuse it. The token
  // endpoint is rate limited per IP (a few calls a minute; the 429 has no CORS headers, so fetch just fails).
  let tok = null, dev = null;
  const tries = () => performance.getEntriesByType('resource').filter((e) => e.name.includes('/auth-service/accessToken')).length;
  for (let i = 0; i < 40 && !tok; i++) {
    const t = localStorage.getItem('public_access_token'), exp = +localStorage.getItem('public_access_token_expiry') || 0;
    if (t && (!exp || exp > Date.now() + 60000)) { tok = t; dev = localStorage.getItem('public_device_id'); }
    else if (tries() >= 2) break; // the page is already retrying its token call: it is being rate limited
    else await sleep(150);
  }
  if (!tok) {
    // The page tried and got nothing: another call would only extend the rate limit.
    if (tries()) return fail(429);
    try {
      const r = await fetch(GW + '/auth-service/accessToken?_nonce=' + crypto.randomUUID());
      if (!r.ok) return fail(r.status);
      const j = await r.json(); tok = j.accessToken; dev = j.deviceId;
    } catch (e) { return fail(429); }
  }
  try {
    const where = pin ? 'pincode=' + encodeURIComponent(pin) : (lat && lon ? 'latitude=' + encodeURIComponent(lat) + '&longitude=' + encodeURIComponent(lon) : '');
    const svP = where ? fetch(GW + '/serviceability-api/v1/geocode/serviceable?' + where).then((r) => r.json()).catch(() => null) : Promise.resolve(null);
    let pinUsed = pin;
    if (!pinUsed) { const s0 = await svP; pinUsed = (s0 && s0.data && s0.data.data && s0.data.data.pincode) || ''; }
    const [sr, sv] = await Promise.all([
      fetch(GW + '/search-service/v5/fullSearch', {method: 'POST',
        headers: {'content-type': 'application/json', authorization: tok, 'x-device-id': dev || '', 'x-source-service': 'PHARMA_AP_IN', 'x-app-os': 'web'},
        body: JSON.stringify({query: q, page: 1, productsPerPage: 24, selSortBy: 'relevance', filters: [], pincode: pinUsed || ''})}),
      svP]);
    if (!sr.ok) return fail(sr.status);
    const sj = await sr.json();
    const svd = sv && sv.data && sv.data.data;
    const list = (sj.data && sj.data.productDetails && sj.data.productDetails.products) || [];
    const pack = (p) => {  // labels look like ["Rx", "15 Tablet", "Strip"] or ["375 gm Powder", "Vanilla", "Pack"]
      const l = ((p.additionalDetails && p.additionalDetails.labels) || []).filter((x) => x && x !== 'Rx');
      if (l.length < 2) return l[0] || null;
      return l[l.length - 1] + ' of ' + l[0] + (l.length > 2 ? ', ' + l.slice(1, -1).join(', ') : '');
    };
    // tatResponse seen: {EXPRESS, "4 hrs 15 mins"}, {NEXT_DAY, "Tom. 10:00 PM"}, {COURIER, "Wed, 14 Oct"}, {null, null}
    const etaOf = (p) => {
      const t = p.tatResponse;
      if (!t || !t.value) return null;
      const v = String(t.value).replace(/^Tom\.?\s*/i, 'tomorrow ').trim();
      const n = t.name ? String(t.name).replace(/_/g, ' ').toLowerCase() : '';
      const head = n ? n.charAt(0).toUpperCase() + n.slice(1) + ', ' : '';
      return head + (/^\d+\s*(hr|min)/i.test(v) ? 'in ' : 'by ') + v;
    };
    const products = list.filter((p) => p && p.name).map((p) => ({
      name: p.name, pack: pack(p),
      price: p.specialPrice != null ? p.specialPrice : p.price, mrp: p.price,
      available: p.status === 'in-stock', rx_required: p.isPrescriptionRequired === 1 || p.isPrescriptionRequired === true,
      id: p.sku || (p.id != null ? String(p.id) : null),
    }));
    const first = list.find((p) => p && p.status === 'in-stock' && etaOf(p));
    return {status: sr.status, serviceable: svd ? !!svd.isServiceable : null, eta: first ? etaOf(first) : null, products};
  } catch (e) { return fail(0); }
})()
