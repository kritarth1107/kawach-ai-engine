(async () => {
  const q = %(q)s, city = %(city)s || 'Raipur', lat = %(lat)s, lon = %(lon)s;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  // Akamai sends some browsers an automatic check page first ("Challenge Validation"). It reloads into the real page
  // by itself (about 25 s), which ends this evaluation; the caller then runs it again on the real page. Only wait here.
  const onCheck = () => !!document.getElementById('sec-cpt-if') || /challenge validation/i.test(document.title);
  if (onCheck()) {
    for (let i = 0; i < 56 && onCheck(); i++) await sleep(500);
    if (onCheck()) return {status: 0, error: 'Akamai browser check still running', items: []};
  }
  const call = async (url, opts, ms) => {
    const ac = new AbortController();
    const t = setTimeout(() => ac.abort(), ms || 10000);
    try {
      const r = await fetch(url, Object.assign({credentials: 'include', signal: ac.signal}, opts || {}));
      const txt = await r.text();
      let j = null; try { j = JSON.parse(txt); } catch (e) {}
      return {status: r.status, j, html: !j && /<html/i.test(txt)};
    } catch (e) { return {status: 0, j: null, error: String(e)}; } finally { clearTimeout(t); }
  };
  // 1) the search box's own autocomplete maps the words to a Practo search term: specialty, symptom, service, common name
  let term = null, matched = null;
  const ap = new URLSearchParams({query: q, exclude: JSON.stringify(['locality', 'region', 'insurance_providers']), contexts: JSON.stringify({city})});
  const ac = await call('/client-api/v1/cerebro/v3/autocomplete?' + ap.toString(), {headers: {accept: 'application/json'}}, 8000);
  if (ac.html) return {status: 0, error: 'bot check page instead of data', items: []};
  const cands = ((((ac.j || {}).results || {}).default || {}).matches || [])
    .filter((m) => m && m.category && (!m.type || m.type === 'doctor') && !/_name$|locality|city|region|insurance/.test(m.category));
  const lq = q.trim().toLowerCase();
  const best = cands.find((m) => String(m.original || m.suggestion || '').toLowerCase() === lq) || cands[0];
  if (best) { term = {word: String(best.original || best.suggestion).toLowerCase(), autocompleted: true, category: best.category}; matched = best.suggestion + ' (' + best.category + ')'; }
  else term = {word: lq, autocompleted: false};
  // 2) the doctor listing the /search/doctors page loads (same JSON the server renders into window.__REDUX_STATE__)
  const sp = new URLSearchParams({results_type: 'doctor', q: JSON.stringify([term]), city, topaz: 'true', enable_partner_listing: 'true', with_ad: 'true', platform: 'desktop_web'});
  if (lat && lon) { sp.set('latitude', lat); sp.set('longitude', lon); sp.set('location_type', 'geo location'); }
  const sr = await call('/marketplace-api/dweb/search/provider?' + sp.toString(), {headers: {accept: 'application/json', 'api-version': '2'}});
  if (sr.html) return {status: 0, error: 'bot check page instead of data', items: []};
  if (sr.status !== 200 || !sr.j || !sr.j.doctors) return {status: sr.status || 0, error: sr.error || 'no doctor list', matched, items: []};
  const E = sr.j.doctors.entities || {};
  const docs = (sr.j.doctors.items || []).map((it) => E[it.id]).filter((d) => d && d.doctor_name);
  // 3) one batched call for the right-hand column of the cards: next bookable slot, video consult button
  const info = {};
  if (docs.length) {
    // the body the listing page sends (without page_type/search_meta/available_filters the server answers 500)
    const ld = sr.j.listing_data || {};
    const body = {doctors_info: docs.map((d) => ({fabric_doctor_id: d.doctor_id, fabric_practice_id: d.practice_id, fabric_relation_id: d.id, is_sensodyne_campaign_enabled: false})),
      page_type: 'listing', search_meta: {book_type: ld.book_type || '', total_results: ld.doctors_found || docs.length},
      available_filters: Array.isArray(sr.j.filters_data) ? sr.j.filters_data : []};
    const cp = new URLSearchParams({platform: 'desktop_web', page_source: 'Doctor Listing'});
    const ct = await call('/marketplace-api/dweb/provider/cta-info?' + cp.toString(), {method: 'POST', headers: {accept: 'application/json', 'content-type': 'application/json'}, body: JSON.stringify(body)}, 8000);
    for (const b of ((ct.j || {}).booking_info || [])) info[b.relation_id] = b;
  }
  const num = (v) => (v === null || v === undefined || v === '' || isNaN(+v) ? null : +v);
  const ist = (ts) => {
    const d = ts ? new Date(String(ts).replace(/\+0000$/, 'Z')) : null;
    return d && !isNaN(d) ? d.toLocaleString('en-IN', {timeZone: 'Asia/Kolkata', weekday: 'short', day: 'numeric', month: 'short', hour: 'numeric', minute: '2-digit', hour12: true}) : null;
  };
  const items = docs.map((d) => {
    const cta = (d.card_features && d.card_features.allowed_cta) || null;
    const b = info[d.id], c = (b && b.cta_data_v2) || {}, p = c.primary_cta || {}, s = c.secondary_cta || {};
    let next = null;
    if (b) {
      if (p.type === 'book' && p.availability_text) next = p.availability_text + ' (clinic visit)';
      else if (p.type === 'schedule_consult' && p.availability_text) next = p.availability_text + ' (video consult)';
    } else if (cta && cta.book && d.next_available_timestamp) next = ist(d.next_available_timestamp) + ' (clinic visit)';
    let video = cta ? !!(cta.scheduled_consult || cta.consult || cta.direct_consult) : null;
    if (b && (p.type === 'schedule_consult' || s.type === 'schedule_consult')) video = true;
    let url = null;
    try { const u = new URL(d.profile_url, location.origin); const pid = u.searchParams.get('practice_id'), spec = u.searchParams.get('specialization'); u.search = ''; if (pid) u.searchParams.set('practice_id', pid); if (spec) u.searchParams.set('specialization', spec); url = u.href; } catch (e) {}
    const pr = d.practice || {};
    return {
      doctor: d.doctor_name,
      specialty: d.specialization || ((d.specialties || [])[0] || {}).sub_specialty || null,
      experience_years: num(d.experience_years),
      fee: d.fee_unknown || d.show_consultation_fees === false ? null : num(d.consultation_fees),
      clinic: d.clinic_name || pr.name || null,
      area: d.locality || pr.locality || null,
      rating: num(d.recommendation_percent),
      next_slot: next,
      video_consult: video,
      id: String(d.id),
      url,
    };
  });
  return {status: sr.status, matched, items};
})()
