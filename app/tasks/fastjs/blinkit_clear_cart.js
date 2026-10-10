(() => {
  // After an order: the app's own cart copy still held the ordered items (lab 2026-10-10); empty it.
  try { const c = JSON.parse(localStorage.getItem('cart') || '{}') || {};
    localStorage.setItem('cart', JSON.stringify(Object.assign(c, { items: {}, cartItems: {}, count: 0, total: 0, uniqueSkuInCart: 0 }))); return true; } catch (e) { return false; }
})()
