"""
Static data: Product catalog, order database, and support policies.

In production these would come from a real database. For the demo
they are plain Python structures so you can see everything at a glance.
"""

from __future__ import annotations

# ── Product Catalog (used by RAG) ───────────────────────────
SNACKSTACK_MENU: list[dict] = [
{
        "id": "DISH001",
        "name": "Margherita Pizza",
        "category": "Main Course",
        "cuisine": "Italian",
        "price (INR)": 299,
        "rating": 4.7,
        "dietary_tags": ["Veg"],
        "description": "Classic thin crust with tomato, mozzarella, basil",
        "availability": True
    },
    {
        "id": "DISH002",
        "name": "Vegan Pasta Primavera",
        "category": "Main Course",
        "cuisine": "Italian",
        "price (INR)": 349,
        "rating": 4.5,
        "dietary_tags": ["Vegan"],
        "description": "Penne with seasonal vegetables, olive oil, garlic",
        "availability": True
    },
    {
        "id": "DISH003",
        "name": "Butter Chicken",
        "category": "Main Course",
        "cuisine": "Indian",
        "price (INR)": 379,
        "rating": 4.9,
        "dietary_tags": ["GF"],
        "description": "Creamy tomato curry with tender chicken and naan",
        "availability": True
    },
    {
        "id": "DISH004",
        "name": "Vegan Buddha Bowl",
        "category": "Main Course",
        "cuisine": "Fusion",
        "price (INR)": 319,
        "rating": 4.6,
        "dietary_tags": ["Vegan", "GF"],
        "description": "Quinoa, chickpeas, avocado, greens, tahini",
        "availability": True
    },
    {
        "id": "DISH005",
        "name": "Classic Cheeseburger",
        "category": "Main Course",
        "cuisine": "American",
        "price (INR)": 259,
        "rating": 4.4,
        "dietary_tags": [],
        "description": "Beef patty, cheddar, lettuce, tomato, brioche bun",
        "availability": True
    },
    {
        "id": "DISH006",
        "name": "Paneer Tikka",
        "category": "Starter",
        "cuisine": "Indian",
        "price (INR)": 199,
        "rating": 4.8,
        "dietary_tags": ["Veg", "GF"],
        "description": "Tandoor-grilled cottage cheese with peppers",
        "availability": True
    },
    {
        "id": "DISH007",
        "name": "Aglio e Olio",
        "category": "Main Course",
        "cuisine": "Italian",
        "price (INR)": 279,
        "rating": 4.5,
        "dietary_tags": ["Vegan"],
        "description": "Spaghetti with garlic, chilli, olive oil, parsley",
        "availability": True
    },
    {
        "id": "DISH008",
        "name": "Mango Lassi",
        "category": "Beverage",
        "cuisine": "Indian",
        "price (INR)": 99,
        "rating": 4.7,
        "dietary_tags": ["Veg", "GF"],
        "description": "Blended yogurt with Alphonso mango, cardamom",
        "availability": True
    },
]


# ── Order Database ───────────────────────────────────────────
ORDER_DATABASE: dict[str, dict] = {
    "ORD-201": {
        "item_id": "DISH003",
        "item_name": "Butter Chicken",
        "customer_name": "Priya Nair",
        "customer_email": "priya@example.com",
        "status": "Out for Delivery",
        "price": 379,
        "order_date": "2026-05-12",
        "estimated_delivery": "2026-05-13",
        "tracking_id": "SS201TRK"
    },
    "ORD-202": {
        "item_id": "DISH001",
        "item_name": "Margherita Pizza",
        "customer_name": "Arjun Mehta",
        "customer_email": "arjun@example.com",
        "status": "Placed",
        "price": 299,
        "order_date": "2026-05-13",
        "estimated_delivery": "2026-05-13",
        "tracking_id": "SS202TRK"
    },
    "ORD-203": {
        "item_id": "DISH005",
        "item_name": "Classic Cheeseburger",
        "customer_name": "Sneha Roy",
        "customer_email": "sneha@example.com",
        "status": "Preparing",
        "price": 259,
        "order_date": "2026-05-13",
        "estimated_delivery": "2026-05-13",
        "tracking_id": "SS203TRK"
    },
    "ORD-204": {
        "item_id": "DISH004",
        "item_name": "Vegan Buddha Bowl",
        "customer_name": "Rahul Das",
        "customer_email": "rahul@example.com",
        "status": "Delivered",
        "price": 319,
        "order_date": "2026-05-13",
        "estimated_delivery": "2026-05-13",
        "tracking_id": "SS204TRK"
    },
    "ORD-205": {
        "item_id": "DISH006",
        "item_name": "Paneer Tikka",
        "customer_name": "Kavya Sharma",
        "customer_email": "kavya@example.com",
        "status": "Placed",
        "price": 199,
        "order_date": "2026-05-13",
        "estimated_delivery": "2026-05-13",
        "tracking_id": "SS205TRK"
    }
}

ORDER_BY_TRACKING: dict[str, dict] = {
    order["tracking_id"]: {**order, "order_id": oid}
    for oid, order in ORDER_DATABASE.items()
}
ORDER_BY_EMAIL: dict[str, dict] = {
    order["customer_email"]: {**order, "order_id": oid}
    for oid, order in ORDER_DATABASE.items()
}