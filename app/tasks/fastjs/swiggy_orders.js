(async () => {
  // Swiggy food: the logged-in account's latest orders (read-only, no clicks), for checking that a practice order placed nothing.
  const out = { status: 0, orders: [], logged_in: null };
  try {
    const r = await fetch('/dapi/order/all?order_id=', { credentials: 'same-origin', headers: { '__fetch_req__': 'true', 'platform': 'dweb' } });
    out.status = r.status;
    if (!r.ok) return out;
    const j = await r.json();
    if (j.statusCode && j.statusCode !== 0) { out.logged_in = false; out.error = String(j.statusMessage || j.statusCode); return out; }
    out.logged_in = true;
    const orders = (j.data && j.data.orders) || [];
    out.keys = orders[0] ? Object.keys(orders[0]).slice(0, 40) : [];
    out.orders = orders.slice(0, 5).map(o => ({
      id: String(o.order_id || o.orderId || ''),
      time: String(o.order_time || o.ordered_time || o.order_placed_time || o.created_at || ''),
      status: String(o.order_status || o.status || ''),
      total: o.order_total != null ? o.order_total : (o.net_total != null ? o.net_total : null),
      restaurant: String(o.restaurant_name || (o.restaurant || {}).name || ''),
    }));
  } catch (e) { out.error = String(e && e.message || e); }
  return out;
})()
