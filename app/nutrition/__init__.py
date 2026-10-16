"""Kavach nutrition agent (NUTRITION_PLAN_2026-10-14): a worker inside the engine that turns what people say they ate into
counted, cited nutrition. Saheli stays the only voice: the agent never messages anyone; it answers Saheli's tools
(log_food, food_log, change_food, nutrition_target, pantry, nutrition_settings, nutrition_report) and asks through her.

Modules: models (tables), nutrients (keys, checks, maths), portions (units → grams), lookup (the nutrition lookup chain),
vision (meal and label photos), entries (the food log, versions and undo), targets and settings (care facts), pantry
(food at home from orders), jobs, turn and agent (the state machine), report (day, week, month, doctor), nudge_hooks (what
Kavach NUdge reads), backfill (old meal events → entries), photos (the asia-south1 bucket, when configured).
"""
