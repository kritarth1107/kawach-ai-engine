(async () => {
  // Blinkit: the account's order list (the orders page's own request), to find an order whose place step had no clear answer.
  const H = Object.assign(JSON.parse(%(page_headers)s), {'content-type': 'application/json'});
  const r = await fetch('/v1/layout/order_history', {method: 'POST', headers: H, body: '{}'});
  if (!r.ok) return {status: r.status};
  return {status: 200, body: await r.json()};
})()
