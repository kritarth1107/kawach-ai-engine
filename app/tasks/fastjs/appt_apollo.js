(async () => {
  const q = %(q)s, cityName = %(city)s, lat = %(lat)s, lon = %(lon)s;
  const API = 'https://api.apollo247.com/', GW = 'https://apigateway.apollo247.in', SITE = 'https://www.apollo247.com';
  const fail = (status) => ({status, items: []});
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
    let la = parseFloat(lat), lo = parseFloat(lon);
    if ((!isFinite(la) || !isFinite(lo)) && cityName) { // no coordinates: geocode the city like the site's city picker
      const g = await gql('GetGoogleMapsLocationDetails', {getLocationDetailsInput: {place: cityName, latitude: null, longitude: null}},
        'query GetGoogleMapsLocationDetails($getLocationDetailsInput: GetLocationDetailsInput) { getGoogleMapsLocationDetails(getLocationDetailsInput: $getLocationDetailsInput) { data } }', 6000);
      const loc = g.data && g.data.getGoogleMapsLocationDetails && g.data.getGoogleMapsLocationDetails.data;
      const p = loc && loc.results && loc.results[0] && loc.results[0].geometry && loc.results[0].geometry.location;
      if (p) { la = +p.lat; lo = +p.lng; }
    }
    const geo = isFinite(la) && isFinite(lo);
    // 1. Specialty, symptom or condition text -> specialty id (the site's own search box). "knee pain" gives a symptom
    // hit whose specialist is Orthopaedics; "cardiologist" gives the Cardiology specialty.
    const ds = await gql('doctorSearch', {input: {query: q, limit: 30, filters: geo ? {location: {lat: String(la), lon: String(lo)}} : {}}},
      'query doctorSearch($input: DoctorSearchInput) { doctorSearch(input: $input) { category search_field data } }', 8000);
    if (ds.status !== 200) return fail(ds.status);
    const hits = (ds.data && ds.data.doctorSearch) || [];
    if (!ds.data) return fail(0);
    const norm = (s) => String(s || '').toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
    const nq = norm(q);
    const symId = (h) => h && h.data && h.data.specialists && h.data.specialists.specialist_id;
    const spId = (h) => h && h.data && (h.data.specialtyId || h.data.id);
    const exactSym = hits.find((h) => h.category === 'symptom' && norm(h.search_field) === nq && symId(h));
    const firstSp = hits.find((h) => h.category === 'specialty' && spId(h));
    const anySym = hits.find((h) => h.category === 'symptom' && symId(h));
    const specialty = exactSym ? symId(exactSym) : firstSp ? spId(firstSp) : anySym ? symId(anySym) : null;
    if (!specialty) return {status: 200, items: []};
    // 2. Doctors of that specialty: near the location (clinic visit and/or online), and Apollo's online doctors from
    // anywhere (used when the city has few or no Apollo doctors, as the site itself does).
    const DL = 'query GetDoctorListV2($filterInput: FilterDoctorInput) { getDoctorListV2(filterInput: $filterInput) { doctors apolloDoctorCount partnerDoctorCount } }';
    const base = {experience: [], availability: [], fees: [], gender: [], language: [], doctorType: [], facilityType: [], sort: 'relevance',
      city: null, pageNo: 1, displaySpecialties: [], pageSize: 15, specialty, conditionSlugNames: [], procedureSlugNames: [],
      callSource: {appPlatform: 'DWEB', fromPage: '/specialties', source: 'WEB', webRender: 'CLIENT'}};
    const [near, online] = await Promise.all([
      geo ? gql('GetDoctorListV2', {filterInput: Object.assign({}, base, {consultMode: 'BOTH', isDigitalConsultation: true, isHospitalVisitConsultation: true,
        geolocation: {latitude: la, longitude: lo}, radius: 50, countSort: 'distance'})}, DL) : Promise.resolve(null),
      gql('GetDoctorListV2', {filterInput: Object.assign({}, base, {consultMode: 'ONLINE', isDigitalConsultation: true, isHospitalVisitConsultation: false})}, DL),
    ]);
    // Apollo's list also holds doctors of related specialties (e.g. general physicians under Cardiology): exact
    // specialty first, Apollo's order kept otherwise.
    const docs = (r) => ((r && r.data && r.data.getDoctorListV2 && r.data.getDoctorListV2.doctors) || [])
      .map((d, i) => [d, i]).sort((a, b) => ((b[0] && b[0].specialtyId === specialty) - (a[0] && a[0].specialtyId === specialty)) || a[1] - b[1]).map((x) => x[0]);
    if ((!near || near.status !== 200) && online.status !== 200) return fail((near && near.status) || online.status);
    const fmt = (iso) => {
      if (!iso) return null;
      const d = new Date(iso);
      if (isNaN(d)) return null;
      return d.toLocaleString('en-IN', {timeZone: 'Asia/Kolkata', weekday: 'short', day: 'numeric', month: 'short', hour: 'numeric', minute: '2-digit', hour12: true});
    };
    const num = (v) => (v == null || v === '' || isNaN(Number(v)) ? null : Number(v));
    const slug = (s) => String(s || '').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');
    const toItem = (d, local) => {
      const physical = local && d.consultMode !== 'ONLINE';
      const ns = d.doctorNextAvailSlots || {};
      const rec = d.doctorRecommendation;
      const city = d.doctorfacilityCity || null;
      return {
        doctor: d.displayName || null,
        specialty: d.specialistSingularTerm || d.specialtydisplayName || null,
        experience_years: num(d.experience),
        fee: num(physical ? (d.physicalConsultationFees != null ? d.physicalConsultationFees : d.fee) : (d.onlineConsultationFees != null ? d.onlineConsultationFees : d.fee)),
        clinic: d.doctorfacility || null,
        area: city && local && d.configuredDistanceFromUserLocation ? city + ', ' + d.configuredDistanceFromUserLocation : city,
        rating: rec && rec.ratingText ? rec.ratingText + ' recommend' + (rec.roundedTotalRecommendationsText ? ' (' + rec.roundedTotalRecommendationsText + ')' : '') : null,
        next_slot: fmt(physical ? (ns.physicalSlot || d.slot) : (ns.onlineSlot || d.slot)),
        video_consult: d.isVideoConsult === true || d.consultMode === 'ONLINE' || d.consultMode === 'BOTH' ? true : (d.consultMode ? false : null),
        id: d.id || null,
        url: d.id ? SITE + '/doctors/' + slug(d.displayName) + '-' + d.id : null,
      };
    };
    const seen = new Set(), items = [];
    for (const d of docs(near)) {
      if (items.length >= 10) break;
      if (d && d.id && !seen.has(d.id) && !d.disabled) { seen.add(d.id); items.push(toItem(d, true)); }
    }
    for (const d of docs(online)) {
      if (items.length >= 10) break;
      if (d && d.id && !seen.has(d.id) && !d.disabled) { seen.add(d.id); items.push(toItem(d, false)); }
    }
    return {status: 200, items};
  } catch (e) { return fail(0); }
})()
