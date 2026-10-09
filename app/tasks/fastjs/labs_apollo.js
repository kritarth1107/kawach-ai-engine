(async () => {
  const q = %(q)s, pin = %(pincode)s, lat = %(lat)s, lon = %(lon)s;
  const API = 'https://api.apollo247.com/', GW = 'https://apigateway.apollo247.in', SITE = 'https://www.apollo247.com';
  const fail = (status, serviceable) => ({status, serviceable: serviceable == null ? null : serviceable, items: [], slots: null});
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  // Guest token: on load the page fetches a public token and keeps it in localStorage for 12 h. Reuse it. The token
  // endpoint is rate limited per IP (a few calls a minute; the 429 has no CORS headers, so fetch just fails).
  let tok = null, dev = null;
  const tries = () => performance.getEntriesByType('resource').filter((e) => e.name.includes('/auth-service/accessToken')).length;
  for (let i = 0; i < 50 && !tok; i++) {
    const t = localStorage.getItem('public_access_token'), exp = +localStorage.getItem('public_access_token_expiry') || 0;
    if (t && (!exp || exp > Date.now() + 60000)) { tok = t; dev = localStorage.getItem('public_device_id'); }
    else if (tries() >= 2) break; // the page is already retrying its token call: it is being rate limited
    else await sleep(150);
  }
  if (!tok) {
    if (tries()) return fail(429); // the page tried and got nothing: another call would only extend the block
    try {
      const ac = new AbortController(); const tm = setTimeout(() => ac.abort(), 4000);
      const r = await fetch(GW + '/auth-service/accessToken?_nonce=' + crypto.randomUUID(), {signal: ac.signal});
      clearTimeout(tm);
      if (!r.ok) return fail(r.status);
      const j = await r.json(); tok = j.accessToken; dev = j.deviceId;
    } catch (e) { return fail(429); }
  }
  // GraphQL on api.apollo247.com. Only allow-listed operation names are accepted (others: "Operation Not Supported").
  const gql = async (operationName, variables, query, ms) => {
    const ac = new AbortController(); const tm = setTimeout(() => ac.abort(), ms || 10000);
    try {
      const r = await fetch(API, {method: 'POST', signal: ac.signal,
        headers: {'content-type': 'application/json', authorization: tok, 'x-device-id': dev || '', 'x-app-os': 'web'},
        body: JSON.stringify({operationName, variables, query})});
      if (!r.ok) return {status: r.status};
      const j = await r.json();
      return {status: 200, data: j && j.data, errors: j && j.errors};
    } catch (e) { return {status: 0}; } finally { clearTimeout(tm); }
  };
  try {
    const la = parseFloat(lat), lo = parseFloat(lon);
    if (!isFinite(la) || !isFinite(lo)) return fail(400);
    const sv = await gql('getDiagnosticServiceability', {latitude: la, longitude: lo, source: 'WEB', postalCode: pin || null},
      'query getDiagnosticServiceability($latitude: Float!, $longitude: Float!, $source: DiagnosticsBookingSource, $postalCode: String) { getDiagnosticServiceability(latitude: $latitude, longitude: $longitude, source: $source, postalCode: $postalCode) { status serviceability { stateID state cityID city } } }', 8000);
    if (sv.status !== 200) return fail(sv.status);
    const svd = sv.data && sv.data.getDiagnosticServiceability;
    if (!svd) return fail(0);
    const city = svd.serviceability;
    if (!svd.status || !city || !city.cityID) return fail(200, false);
    const citySlug = String(city.city || '').toLowerCase().trim().replace(/[^a-z0-9]+/g, '-');
    const [sr, up] = await Promise.all([
      gql('searchDiagnosticItem', {keyword: q, cityId: city.cityID, size: 20, includeRadiology: false, radiologyCity: ''},
        'query searchDiagnosticItem($keyword: String!, $cityId: Int!, $size: Int, $includeRadiology: Boolean, $radiologyCity: String) { searchDiagnosticItem(keyword: $keyword, cityId: $cityId, size: $size, includeRadiology: $includeRadiology, radiologyCity: $radiologyCity) { data { diagnostic_item_id diagnostic_item_name diagnostic_item_itemType diagnostic_item_collectionType diagnostic_item_canonicalTag diagnostic_inclusions testParametersCount testParametersCountWithHeaderLogic radiologyItem diagnostic_item_price { price mrp groupPlan status } } matchedRecords } }'),
      // Earliest home-collection slot for the pincode ("Next slot available: 06:00 AM, Tomorrow")
      gql('getUpcomingSlotInfo', {latitude: la, longitude: lo, zipcode: pin || '', serviceability: {cityID: city.cityID, stateID: city.stateID || 0}},
        'query getUpcomingSlotInfo($latitude: Float!, $longitude: Float!, $zipcode: String!, $serviceability: DiagnosticsServiceability!) { getUpcomingSlotInfo(latitude: $latitude, longitude: $longitude, zipcode: $zipcode, serviceability: $serviceability) { status slotInfo slotMessage } }', 6000),
    ]);
    if (sr.status !== 200) return fail(sr.status, true);
    const res = sr.data && sr.data.searchDiagnosticItem;
    if (!res) return fail(0, true);
    const list = (res.data || []).filter((d) => d && d.diagnostic_item_name && !d.radiologyItem);
    const ids = list.map((d) => d.diagnostic_item_id).filter((x) => x != null);
    // Report times for all items in one call; preparation (fasting) is only on the item page query, one call per item,
    // so only the first few items get it.
    const PREP_N = 4;
    const [tat, ...preps] = await Promise.all([
      ids.length ? gql('getConfigurableReportTAT', {cityId: city.cityID, pincode: parseInt(pin, 10) || 0, itemIds: ids, latitude: la, longitude: lo},
        'query getConfigurableReportTAT($cityId: Int!, $pincode: Int!, $itemIds: [Int]!, $latitude: Float, $longitude: Float) { getConfigurableReportTAT(cityId: $cityId, pincode: $pincode, itemIds: $itemIds, latitude: $latitude, longitude: $longitude) { itemLevelReportTATs { itemId reportTATMessage preOrderReportTATMessage } } }', 6000) : Promise.resolve(null),
      ...list.slice(0, PREP_N).map((d) => gql('GetDiagnosticsByItemIdAndCanonicalTag', {canonicalTag: d.diagnostic_item_canonicalTag, cityId: city.cityID, itemId: d.diagnostic_item_id, pincode: parseInt(pin, 10) || null},
        'query GetDiagnosticsByItemIdAndCanonicalTag($canonicalTag: String, $cityId: Int!, $itemId: Int!, $pincode: Int) { getDiagnosticsByItemIdAndCanonicalTag(canonicalTag: $canonicalTag, cityId: $cityId, itemId: $itemId, pincode: $pincode) { itemId itemPreparationData } }', 5000)),
    ]);
    const tatById = {};
    const tl = (tat && tat.data && tat.data.getConfigurableReportTAT && tat.data.getConfigurableReportTAT.itemLevelReportTATs) || [];
    for (const t of tl) if (t && t.itemId != null && !tatById[t.itemId]) tatById[t.itemId] = t.preOrderReportTATMessage || t.reportTATMessage || null;
    const prepById = {};
    for (const p of preps) {
      const d = p && p.data && p.data.getDiagnosticsByItemIdAndCanonicalTag;
      if (d && d.itemId != null) prepById[d.itemId] = d.itemPreparationData == null ? '' : String(d.itemPreparationData);
    }
    // Item page text: "10- 12 Hr fasting is required" -> true; "No preparation required" -> false. Apollo leaves it empty
    // for packages: then true if an included test is a fasting one ("GLUCOSE, FASTING"), else unknown (null).
    const fastingOf = (txt, inc) => {
      const t = String(txt || '').toLowerCase().trim();
      if (t) {
        if (/no (special )?(fasting|preparation)|not required|fasting is not/.test(t)) return false;
        return /fast/.test(t);
      }
      return (inc || []).some((x) => /fasting/i.test(String(x))) ? true : null;
    };
    const items = list.map((d) => {
      const prices = (d.diagnostic_item_price || []).filter((p) => p && p.price != null);
      const pr = prices.find((p) => p.groupPlan === 'ALL' && (!p.status || p.status === 'active')) || prices[0] || {};
      const ct = d.diagnostic_item_collectionType;
      const tag = d.diagnostic_item_canonicalTag;
      return {
        name: d.diagnostic_item_name,
        kind: String(d.diagnostic_item_itemType || '').toUpperCase() === 'PACKAGE' ? 'package' : 'test',
        price: pr.price != null ? Number(pr.price) : null,
        mrp: pr.mrp != null ? Number(pr.mrp) : null,
        tests_included: d.testParametersCountWithHeaderLogic != null ? d.testParametersCountWithHeaderLogic : (d.testParametersCount != null ? d.testParametersCount : null),
        fasting: fastingOf(prepById[d.diagnostic_item_id], d.diagnostic_inclusions),
        report_time: tatById[d.diagnostic_item_id] || null,
        home_collection: ct ? /HC/i.test(ct) : null,
        id: String(d.diagnostic_item_id),
        url: tag ? SITE + '/lab-tests/' + tag + (citySlug ? '-c-' + citySlug : '') : null,
      };
    });
    const u = up && up.data && up.data.getUpcomingSlotInfo;
    const slotText = u && u.status && u.slotMessage ? String(u.slotMessage).replace(/^[^:]*slot[^:]*:\s*/i, '').trim() : '';
    return {status: 200, serviceable: true, items, slots: slotText ? [slotText] : null};
  } catch (e) { return fail(0); }
})()
