def calculate_bill(cart: list[dict], customer: dict) -> int:
    """
    Compute the final total in cents for a shopping cart given the cart contents and customer information.

    Args:
        cart (list[dict]): A list of dictionaries, each representing an item in the cart.
            Each dictionary should have the keys "name" (str), "price" (int, cents), "qty" (int), and "category" (str).
        customer (dict): A dictionary containing customer information.
            It should have the keys "member" (bool), "coupon" (str or None), and "state" (str).

    Returns:
        int: The final total in cents.
    """

    # LINE TOTAL: Calculate the line total for each item and sum them up to get the subtotal
    line_totals = {}
    subtotal = 0
    for item in cart:
        line_total = item["price"] * item["qty"]
        line_totals[item["category"]] = line_totals.get(item["category"], 0) + line_total
        subtotal += line_total

    # BULK DISCOUNT: Apply a 10% discount to items with a quantity of 10 or more
    for item in cart:
        if item["qty"] >= 10:
            discount = (item["price"] * item["qty"]) * 0.1
            subtotal -= discount

    # MEMBER DISCOUNT: Apply a 5% discount if the customer is a member
    if customer["member"]:
        discount = subtotal * 0.05
        subtotal -= discount

    # COUPON: Apply the coupon discount if present
    if customer["coupon"]:
        if customer["coupon"] == "SAVE20" and subtotal >= 5000:
            discount = subtotal * 0.2
            subtotal -= discount
        elif customer["coupon"] == "FLAT500":
            subtotal -= 500
            if subtotal < 0:
                subtotal = 0

    # CATEGORY TAX: Apply tax based on the category mix
    category_tax = 0
    for category, total in line_totals.items():
        if category == "clothing":
            tax = (subtotal * (total / sum(line_totals.values()))) * 0.05
            category_tax += tax
        elif category == "electronics":
            tax = (subtotal * (total / sum(line_totals.values()))) * 0.1
            category_tax += tax
    subtotal += category_tax

    # STATE SURCHARGE: Add a flat 200 cent surcharge if the customer is from CA
    if customer["state"] == "CA":
        subtotal += 200

    # ROUNDING: Round the final total to the nearest 5 cents
    final_total = round(subtotal / 5) * 5

    return int(final_total)
