(async () => {
  // Swiggy food dish search, guest, no clicks. Run inside a tab already on https://www.swiggy.com/
  // (after the AWS WAF challenge reload, if any, has finished, i.e. window.___INITIAL_STATE__ exists).
  const q = String(%(q)s), lat = String(%(lat)s), lng = String(%(lon)s);
  const MAX = 100;
  const out = { status: 0, products: [] };
  // The first response for a new browser is an AWS WAF JS challenge page that reloads itself; the real Swiggy pages set
  // window.___INITIAL_STATE__ (food) or window.___INITIAL_STATE___ (Instamart). Give a page that is still loading a moment.
  for (let i = 0; i < 80 && !(window.___INITIAL_STATE__ || window.___INITIAL_STATE___); i++) await new Promise(r => setTimeout(r, 100));
  const waf = window.AwsWafIntegration;
  const F = waf && typeof waf.fetch === 'function' ? waf.fetch.bind(waf) : fetch;
  const H = { '__fetch_req__': 'true', 'platform': 'dweb', 'user-id': '0', 'Content-Type': 'application/json' };
  try {
    // Same delivery-point cookie the site writes when a user picks a location (keeps later page visits on this address).
    document.cookie = 'userLocation=' + encodeURIComponent(JSON.stringify({ lat, lng, address: '', area: '', showUserDefaultAddressHint: false })) + '; path=/; max-age=2592000';
    const uid = (crypto.randomUUID && crypto.randomUUID()) || ('' + Date.now() + Math.random()).replace('.', '');
    const qs = new URLSearchParams({ lat, lng, str: q, trackingId: 'undefined', submitAction: 'ENTER', queryUniqueId: uid, selectedPLTab: 'DISH' });
    const r = await F('/dapi/restaurants/search/v3?' + qs, { credentials: 'same-origin', headers: H });
    out.status = r.status;
    if (!r.ok) { out.error = 'search HTTP ' + r.status + (r.headers.get('x-amzn-waf-action') ? ' (waf ' + r.headers.get('x-amzn-waf-action') + ')' : ''); return out; }
    const j = await r.json();
    const groups = [];
    for (const c of (j.data && j.data.cards) || []) {
      const m = c.groupedCard && c.groupedCard.cardGroupMap;
      if (!m) continue;
      for (const tab of Object.keys(m)) for (const cc of m[tab].cards || []) {
        const g = cc.card && cc.card.card;
        if (g && g.restaurant && Array.isArray(g.dishes)) groups.push(g);
      }
    }
    if (!groups.length) {
      if (j.statusCode) { out.error = 'search statusCode ' + j.statusCode + ((j.data && j.data.statusMessage) ? ': ' + j.data.statusMessage : ''); return out; }
      // Empty answer: no match, or nothing delivers here. The restaurant listing for the point tells which.
      const r2 = await F('/dapi/restaurants/list/v5?' + new URLSearchParams({ lat, lng, 'is-seo-homepage-enabled': 'true', page_type: 'DESKTOP_WEB_LISTING' }), { credentials: 'same-origin', headers: H });
      const j2 = r2.ok ? await r2.json().catch(() => null) : null;
      if (j2 && j2.statusCode === 0) {
        out.serviceable = ((j2.data && j2.data.cards) || []).some(c => /restaurant_grid_listing/.test(String(c.card && c.card.card && c.card.card.id || '')));
      }
      out.error = out.serviceable === false ? 'no restaurants deliver to this location' : 'no dishes found for this search';
      return out;
    }
    out.serviceable = true;
    // Light re-rank: restaurants whose distinctive name words appear in the query go first (site order kept otherwise).
    const norm = s => String(s || '').toLowerCase().normalize('NFKD').replace(/['’`]/g, '').replace(/[^a-z0-9]+/g, ' ').trim();
    const GENERIC = new Set(['pizza', 'pizzas', 'pizzeria', 'restaurant', 'restro', 'cafe', 'kitchen', 'the', 'and', 'food', 'foods', 'house', 'by', 'of', 'bar', 'grill', 'dhaba', 'corner', 'point', 'express', 'shop', 'hub', 'wala', 'wale']);
    const qw = new Set(norm(q).split(' '));
    const hit = name => norm(name).split(' ').some(w => w.length >= 3 && !GENERIC.has(w) && qw.has(w));
    const ranked = groups.map((g, i) => ({ g, i, h: hit(((g.restaurant || {}).info || {}).name) ? 0 : 1 })).sort((a, b) => a.h - b.h || a.i - b.i);
    const rupees = p => (p == null || p === '') ? null : Number(p) / 100;
    for (const { g } of ranked) {
      const ri = (g.restaurant || {}).info || {};
      const sla = ri.sla || {};
      const open = !ri.availability || ri.availability.opened !== false;
      const serviceable = !sla.serviceability || sla.serviceability === 'SERVICEABLE';
      for (const d of g.dishes) {
        const i = d.info || {};
        if (!i.name) continue;
        const v2 = i.variantsV2 || {};
        const defs = (v2.variantGroups || []).map(vg => { const v = (vg.variations || []).find(x => x.default) || (vg.variations || [])[0]; return v ? v.name : ''; }).filter(Boolean);
        let list = rupees(i.price || i.defaultPrice);
        if (!list) {   // some menus price only the variants: take the cheapest variant price
          const vp = [].concat((v2.pricingModels || []).map(m => m.price), ...(v2.variantGroups || []).map(vg => (vg.variations || []).map(x => x.price))).filter(x => x > 0);
          list = vp.length ? rupees(Math.min(...vp)) : null;
        }
        const fin = rupees(i.finalPrice);
        out.products.push({
          name: i.name,
          pack: defs.join(', '),
          price: fin != null && fin > 0 ? fin : list,
          mrp: list,
          available: !!(Number(i.inStock) === 1 && open && serviceable),
          id: String(i.id || ''),
          restaurant: ri.name || '',
          eta: sla.slaString || (sla.deliveryTime ? sla.deliveryTime + ' MINS' : ''),
          restaurantId: String(ri.id || ''),
          closed: !open,
          opens: !open ? String((ri.availability || {}).nextOpenTimeMessage || (ri.availability || {}).nextOpenTime || '') : '',
          far: !serviceable,
        });
        if (out.products.length >= MAX) return out;
      }
    }
  } catch (e) { out.error = String(e && e.message || e); }
  return out;
})()
