(async () => {
  const lat = %(lat)s, lon = %(lon)s, q = %(q)s;
  const API = 'https://bff-gateway.zepto.com';
  const readCk = () => Object.fromEntries(document.cookie.split('; ').filter(Boolean).map(s => { const i = s.indexOf('='); let v = s.slice(i + 1); try { v = decodeURIComponent(v); } catch (e) {} return [s.slice(0, i), v]; }));
  let ck = readCk();
  for (let i = 0; i < 40 && !(ck['XSRF-TOKEN'] && ck.device_id); i++) { await new Promise(r => setTimeout(r, 250)); ck = readCk(); }
  if (!ck['XSRF-TOKEN'] || !ck.device_id) return {status: 0, serviceable: null, error: 'zepto page not ready (no XSRF-TOKEN/device_id cookie yet)', products: []};
  const hex = async (s) => Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(s)))).map(b => b.toString(16).padStart(2, '0')).join('');
  const call = async (method, path, body, extra) => {
    const rid = crypto.randomUUID();
    const u = new URL(API + path);
    const signUrl = u.pathname + (u.searchParams.size ? '?' + u.searchParams.toString() : '');
    const bodyStr = body === undefined ? undefined : JSON.stringify(body);
    // the web app's own request integrity headers: sha256 of sorted body|deviceId|method|requestId|xsrf|url, and sha256 of that
    const sig = await hex([bodyStr, ck.device_id, method.toLowerCase(), rid, ck['XSRF-TOKEN'], signUrl].map(String).join('|'));
    const headers = Object.assign({
      accept: 'application/json, text/plain, */*', platform: 'WEB', app_sub_platform: 'WEB', tenant: 'ZEPTO',
      appversion: '17.2.2', app_version: '17.2.2', auth_revamp_flow: 'v2', source: 'DIRECT', auth_from_cookie: 'true',
      marketplace_type: ck.marketplace || 'SUPER_SAVER', device_id: ck.device_id, deviceid: ck.device_id,
      session_id: ck.session_id || '', sessionid: ck.session_id || '', request_id: rid, requestid: rid,
      'x-xsrf-token': ck['XSRF-TOKEN'], 'x-csrf-secret': ck.csrfSecret || '', 'request-signature': sig, 'x-timezone': await hex(sig),
    }, extra || {});
    if (bodyStr !== undefined) headers['content-type'] = 'application/json';
    const r = await fetch(u.toString(), {method, credentials: 'include', headers, body: bodyStr});
    let j = null; try { j = await r.json(); } catch (e) {}
    return {status: r.status, j};
  };
  let sv;
  try {
    sv = await call('GET', `/lms/api/v2/get_page?latitude=${encodeURIComponent(lat)}&longitude=${encodeURIComponent(lon)}&page_type=HOME&version=v2&show_new_eta_banner=true&page_size=1&enforce_platform_type=DESKTOP`);
  } catch (e) { return {status: 0, serviceable: null, error: 'serviceability call failed: ' + e, products: []}; }
  const ssr = sv.j && sv.j.storeServiceableResponse;
  if (sv.status !== 200 || !ssr) return {status: sv.status, serviceable: null, error: 'no serviceability answer', products: []};
  if (!ssr.serviceable || !ssr.storeId) return {status: sv.status, serviceable: false, products: []};
  const v2 = (sv.j.storeServiceableResponseV2 || []).filter(s => s && s.serviceable && s.storeId);
  const ids = v2.length ? v2.map(s => s.storeId) : [ssr.storeId].concat(ssr.secondaryStoreIds || []);
  const etas = '{' + ids.map(id => `"${id}":-1`).join(',') + '}';
  const mk = (sv.j.defaultMarketplace && sv.j.defaultMarketplace.name) || ck.marketplace || 'SUPER_SAVER';
  let sr;
  try {
    sr = await call('POST', '/user-search-service/api/v3/search',
      {query: q, pageNumber: 0, mode: 'SHOW_ALL_RESULTS', userSessionId: crypto.randomUUID()},
      {store_id: ids[0], storeid: ids[0], store_ids: ids.join(','), store_etas: etas, marketplace_type: mk, 'x-without-bearer': 'true'});
  } catch (e) { return {status: 0, serviceable: true, store_id: ids[0], error: 'search call failed: ' + e, products: []}; }
  if (sr.status !== 200 || !sr.j) return {status: sr.status, serviceable: true, store_id: ids[0], products: []};
  const rs = (p) => typeof p === 'number' ? Math.round(p) / 100 : null;
  const seen = new Set(), products = [];
  for (const w of sr.j.layout || []) {
    const items = w && w.data && w.data.resolver && w.data.resolver.data && w.data.resolver.data.items;
    if (!Array.isArray(items)) continue;
    for (const it of items) {
      const p = it && it.productResponse;
      if (!p || !p.product) continue;
      const id = (p.productVariant && p.productVariant.id) || p.id;
      if (seen.has(id)) continue;
      seen.add(id);
      products.push({
        name: p.product.name && p.product.name.replace(/\s+/g, ' ').trim(), pack: p.productVariant && p.productVariant.formattedPacksize,
        price: rs(p.discountedSellingPrice != null ? p.discountedSellingPrice : p.sellingPrice), mrp: rs(p.mrp),
        available: !p.outOfStock && (p.availableQuantity || 0) > 0, id, brand: p.product.brand || null, store_id: p.storeId,
        sponsored: ((p.meta && p.meta.tags) || []).some(t => t && t.type === 'SPONSORED'),
      });
    }
  }
  // Zepto's own relevance order, with paid placements (tagged SPONSORED) moved after the organic results
  return {status: sr.status, serviceable: true, store_id: ids[0], products: products.filter(p => !p.sponsored).concat(products.filter(p => p.sponsored))};
})()
