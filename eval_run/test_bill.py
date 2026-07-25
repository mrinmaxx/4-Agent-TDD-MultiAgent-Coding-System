from solution import calculate_bill

def check(label, got, expected):
    print(f"  [{'PASS' if got == expected else 'FAIL'}] {label}: got {got}, expected {expected}")

print("=== calculate_bill (dense-rule pricing engine) ===")

# CASE 1 (basic, no discounts/tax): single food item, non-member, no coupon, non-CA
# subtotal = 300*2 = 600; food tax-exempt; no rounding change (600 % 5 == 0) -> 600
check("basic food only",
      calculate_bill([{"name":"apple","price":300,"qty":2,"category":"food"}],
                     {"member":False,"coupon":None,"state":"NY"}), 600)

# CASE 2 (electronics tax): 1 item price 1000 qty 1, non-member, no coupon
# subtotal 1000; no discounts; tax: electronics 10% of its share (100% of total) = 100
# total 1100; round to nearest 5 -> 1100
check("electronics tax 10%",
      calculate_bill([{"name":"cable","price":1000,"qty":1,"category":"electronics"}],
                     {"member":False,"coupon":None,"state":"NY"}), 1100)

# CASE 3 (bulk discount): 1 item price 100 qty 10 -> line 1000, bulk 10% -> 900
# non-member, no coupon, clothing tax 5% of 900 = 45 -> 945; round nearest 5 -> 945
check("bulk discount qty>=10",
      calculate_bill([{"name":"sock","price":100,"qty":10,"category":"clothing"}],
                     {"member":False,"coupon":None,"state":"NY"}), 945)

# CASE 4 (member discount): electronics price 2000 qty 1, member True
# subtotal 2000; member 5% -> 1900; no coupon; tax electronics 10% of 1900 = 190 -> 2090
# round nearest 5 -> 2090
check("member 5% off",
      calculate_bill([{"name":"mouse","price":2000,"qty":1,"category":"electronics"}],
                     {"member":True,"coupon":None,"state":"NY"}), 2090)

# CASE 5 (coupon SAVE20 valid, total>=5000): food price 6000 qty 1, non-member, SAVE20
# subtotal 6000; SAVE20 valid (>=5000) -> 20% off -> 4800; food tax-exempt -> 4800
# round nearest 5 -> 4800
check("SAVE20 valid on food",
      calculate_bill([{"name":"cake","price":6000,"qty":1,"category":"food"}],
                     {"member":False,"coupon":"SAVE20","state":"NY"}), 4800)

# CASE 6 (coupon SAVE20 INVALID, total<5000): food price 3000, SAVE20 -> below 5000 -> ignored
# subtotal 3000; food exempt -> 3000
check("SAVE20 invalid under threshold",
      calculate_bill([{"name":"cake","price":3000,"qty":1,"category":"food"}],
                     {"member":False,"coupon":"SAVE20","state":"NY"}), 3000)

# CASE 7 (FLAT500 + CA surcharge): clothing price 1000 qty 1, non-member, FLAT500, CA
# subtotal 1000; FLAT500 -> 500; clothing tax 5% of 500 = 25 -> 525; CA +200 -> 725
# round nearest 5 -> 725
check("FLAT500 + CA surcharge",
      calculate_bill([{"name":"shirt","price":1000,"qty":1,"category":"clothing"}],
                     {"member":False,"coupon":"FLAT500","state":"CA"}), 725)

# CASE 8 (THE HARD ONE - mixed categories, proportional tax on discounted total):
# items: food 2000 (qty1), electronics 2000 (qty1). subtotal 4000.
# original shares: food 50%, electronics 50%.
# member True -> 5% off -> 3800. no coupon.
# tax: food share = 50% of 3800 = 1900 -> 0 tax; electronics share = 50% of 3800 = 1900 -> 10% = 190
# total 3800 + 190 = 3990; non-CA; round nearest 5 -> 3990
check("HARD: mixed-category proportional tax",
      calculate_bill([{"name":"rice","price":2000,"qty":1,"category":"food"},
                      {"name":"usb","price":2000,"qty":1,"category":"electronics"}],
                     {"member":True,"coupon":None,"state":"NY"}), 3990)

# CASE 9 (rounding half up): construct a total ending in .5 boundary
# clothing 999 qty1 -> subtotal 999; clothing tax 5% = 49.95 -> total 1048.95 cents??
# NOTE: tax in cents may be fractional; engine should round only at the END to nearest 5.
# 999 + 49.95 = 1048.95 -> nearest 5 -> 1050
check("rounding to nearest 5 (half up)",
      calculate_bill([{"name":"tie","price":999,"qty":1,"category":"clothing"}],
                     {"member":False,"coupon":None,"state":"NY"}), 1050)

print("\nDone.")
