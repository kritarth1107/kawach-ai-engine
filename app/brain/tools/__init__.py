"""Saheli's care tools, one module per area. Memory tools act on Postgres directly; world tools go through the host.

base       the framework: TurnCtx, @tool, run(), specs
places     Places and pins: saved places, map links, WhatsApp locations, where a ride starts and ends.
orders     Orders and rides through the task runtime (app/tasks): start, inputs, cancel, status, feedback, boundaries.
memory     Memory tools: remember, stop, notes, recall, confirm a change, habits.
reminders  Reminders, activities and loops: set and change reminders, the day's schedule, open and close loops.
health     Health tools: doses, readings and events of the day, alerts to the family and the care team, records.
views      Care views: refills, emergency card, care team, reports, wellbeing, family tasks, spending.
food       Food tools (the nutrition agent is app/nutrition): log_food, the food log, settings, pantry.
people     People and setup: skills, voice and language, setup, add a person, names, songs, the daily brief.
trips      Trips and look-ups: web search, ways there, flights (note, accept, read), doctors and lab tests.
outcomes   Outcomes, feedback and learning consent.
history    Memory history: what changed, undo, forget.
"""

from __future__ import annotations

from app.brain.tools.base import (
    _CORE as _CORE,
)
from app.brain.tools.base import (
    _GROUPS as _GROUPS,
)
from app.brain.tools.base import (
    _HEALTH_SITUATIONS as _HEALTH_SITUATIONS,
)
from app.brain.tools.base import (
    _TOOLS as _TOOLS,
)
from app.brain.tools.base import (
    _WHEN_SYSTEM as _WHEN_SYSTEM,
)
from app.brain.tools.base import (
    _WHEN_TASK as _WHEN_TASK,
)
from app.brain.tools.base import (  # noqa: F401
    ABOUT as ABOUT,
)
from app.brain.tools.base import (
    ABOUT_OWN as ABOUT_OWN,
)
from app.brain.tools.base import (
    FOR_WHOM as FOR_WHOM,
)
from app.brain.tools.base import (
    SAVES_FOR_SPEAKER as SAVES_FOR_SPEAKER,
)
from app.brain.tools.base import (
    TOOL_GROUPS as TOOL_GROUPS,
)
from app.brain.tools.base import (
    Handler as Handler,
)
from app.brain.tools.base import (
    ToolRefused as ToolRefused,
)
from app.brain.tools.base import (
    TurnCtx as TurnCtx,
)
from app.brain.tools.base import (
    _release_record_locks as _release_record_locks,
)
from app.brain.tools.base import (
    acts_catalog_line as acts_catalog_line,
)
from app.brain.tools.base import (
    arg_problems as arg_problems,
)
from app.brain.tools.base import (
    groups_for as groups_for,
)
from app.brain.tools.base import (
    logger as logger,
)
from app.brain.tools.base import (
    names as names,
)
from app.brain.tools.base import (
    need_tools as need_tools,
)
from app.brain.tools.base import (
    run as run,
)
from app.brain.tools.base import (
    specs as specs,
)
from app.brain.tools.base import (
    specs_for as specs_for,
)
from app.brain.tools.base import (
    tool as tool,
)
from app.brain.tools.food import (
    _WHOSE_REMINDER as _WHOSE_REMINDER,
)
from app.brain.tools.food import (  # noqa: F401
    FOOD_FOR_WHOM as FOOD_FOR_WHOM,
)
from app.brain.tools.food import (
    FOOD_UNITS as FOOD_UNITS,
)
from app.brain.tools.food import (
    REMINDER_ABOUT as REMINDER_ABOUT,
)
from app.brain.tools.food import (
    REMINDER_SCHEDULE as REMINDER_SCHEDULE,
)
from app.brain.tools.food import (
    _change_medicine_time as _change_medicine_time,
)
from app.brain.tools.food import (
    _clock_times as _clock_times,
)
from app.brain.tools.food import (
    _delivered_subject as _delivered_subject,
)
from app.brain.tools.food import (
    _food_day as _food_day,
)
from app.brain.tools.food import (
    _food_household as _food_household,
)
from app.brain.tools.food import (
    _food_may_touch as _food_may_touch,
)
from app.brain.tools.food import (
    _food_qty as _food_qty,
)
from app.brain.tools.food import (
    _food_subject as _food_subject,
)
from app.brain.tools.food import (
    _food_who as _food_who,
)
from app.brain.tools.food import (
    _may_manage_reminders as _may_manage_reminders,
)
from app.brain.tools.food import (
    _reminder_line as _reminder_line,
)
from app.brain.tools.food import (
    _reminder_schedule as _reminder_schedule,
)
from app.brain.tools.food import (
    _reminder_subject as _reminder_subject,
)
from app.brain.tools.food import (
    activity as activity,
)
from app.brain.tools.food import (
    change_food as change_food,
)
from app.brain.tools.food import (
    change_reminder as change_reminder,
)
from app.brain.tools.food import (
    food_log as food_log,
)
from app.brain.tools.food import (
    log_activity as log_activity,
)
from app.brain.tools.food import (
    log_food as log_food,
)
from app.brain.tools.food import (
    nutrition_report as nutrition_report,
)
from app.brain.tools.food import (
    nutrition_settings as nutrition_settings,
)
from app.brain.tools.food import (
    nutrition_target as nutrition_target,
)
from app.brain.tools.food import (
    pantry_tool as pantry_tool,
)
from app.brain.tools.food import (
    reminder_log as reminder_log,
)
from app.brain.tools.food import (
    reminders_tool as reminders_tool,
)
from app.brain.tools.food import (
    set_reminder as set_reminder,
)
from app.brain.tools.food import (
    today_schedule as today_schedule,
)
from app.brain.tools.health import (  # noqa: F401
    VITAL_KINDS as VITAL_KINDS,
)
from app.brain.tools.health import (
    _crisis_told_lately as _crisis_told_lately,
)
from app.brain.tools.health import (
    _follow_up_alert as _follow_up_alert,
)
from app.brain.tools.health import (
    _not_theirs as _not_theirs,
)
from app.brain.tools.health import (
    alert_caregiver as alert_caregiver,
)
from app.brain.tools.health import (
    fact_still_true as fact_still_true,
)
from app.brain.tools.health import (
    health_record as health_record,
)
from app.brain.tools.health import (
    lab_trend as lab_trend,
)
from app.brain.tools.health import (
    log_dose as log_dose,
)
from app.brain.tools.health import (
    log_event as log_event,
)
from app.brain.tools.health import (
    log_vital as log_vital,
)
from app.brain.tools.health import (
    note_habit as note_habit,
)
from app.brain.tools.health import (
    search_records as search_records,
)
from app.brain.tools.history import (  # noqa: F401
    _apply_fact_plan as _apply_fact_plan,
)
from app.brain.tools.history import (
    _history_subjects as _history_subjects,
)
from app.brain.tools.history import (
    _may_touch as _may_touch,
)
from app.brain.tools.history import (
    memory_changes as memory_changes,
)
from app.brain.tools.history import (
    undo_change as undo_change,
)
from app.brain.tools.memory import (  # noqa: F401
    _after_condition_stop as _after_condition_stop,
)
from app.brain.tools.memory import (
    _change_was_asked as _change_was_asked,
)
from app.brain.tools.memory import (
    _follows_the_chat as _follows_the_chat,
)
from app.brain.tools.memory import (
    _sync_medicine as _sync_medicine,
)
from app.brain.tools.memory import (
    confirm_change as confirm_change,
)
from app.brain.tools.memory import (
    note as note,
)
from app.brain.tools.memory import (
    recall as recall,
)
from app.brain.tools.memory import (
    remember as remember,
)
from app.brain.tools.memory import (
    stop as stop,
)
from app.brain.tools.orders import (  # noqa: F401
    ADDRESS_OK_FOR as ADDRESS_OK_FOR,
)
from app.brain.tools.orders import (
    FEEDBACK_ITEM as FEEDBACK_ITEM,
)
from app.brain.tools.orders import (
    _agent as _agent,
)
from app.brain.tools.orders import (
    _and as _and,
)
from app.brain.tools.orders import (
    _care_limits as _care_limits,
)
from app.brain.tools.orders import (
    _confirm_address_first as _confirm_address_first,
)
from app.brain.tools.orders import (
    _create_task as _create_task,
)
from app.brain.tools.orders import (
    _for_whom as _for_whom,
)
from app.brain.tools.orders import (
    _moved as _moved,
)
from app.brain.tools.orders import (
    _named_place_problem as _named_place_problem,
)
from app.brain.tools.orders import (
    _names as _names,
)
from app.brain.tools.orders import (
    _not_their_place as _not_their_place,
)
from app.brain.tools.orders import (
    _own_home_elsewhere as _own_home_elsewhere,
)
from app.brain.tools.orders import (
    _person as _person,
)
from app.brain.tools.orders import (
    _restricted as _restricted,
)
from app.brain.tools.orders import (
    _start_compare as _start_compare,
)
from app.brain.tools.orders import (
    _store_named_with_address as _store_named_with_address,
)
from app.brain.tools.orders import (
    _task as _task,
)
from app.brain.tools.orders import (
    _tell_who_cancelled as _tell_who_cancelled,
)
from app.brain.tools.orders import (
    boundaries_read as boundaries_read,
)
from app.brain.tools.orders import (
    cancel_task as cancel_task,
)
from app.brain.tools.orders import (
    done_in_app as done_in_app,
)
from app.brain.tools.orders import (
    order_feedback as order_feedback,
)
from app.brain.tools.orders import (
    set_boundaries as set_boundaries,
)
from app.brain.tools.orders import (
    set_task_agent as set_task_agent,
)
from app.brain.tools.orders import (
    start_task as start_task,
)
from app.brain.tools.orders import (
    task_agent as task_agent,
)
from app.brain.tools.orders import (
    task_input as task_input,
)
from app.brain.tools.orders import (
    task_status as task_status,
)
from app.brain.tools.orders import (
    whats_pending as whats_pending,
)
from app.brain.tools.outcomes import (  # noqa: F401
    feedback as feedback,
)
from app.brain.tools.outcomes import (
    forget_tool as forget_tool,
)
from app.brain.tools.outcomes import (
    learning_consent as learning_consent,
)
from app.brain.tools.outcomes import (
    log_outcome as log_outcome,
)
from app.brain.tools.outcomes import (
    offer_buttons as offer_buttons,
)
from app.brain.tools.people import (  # noqa: F401
    SETUP_ITEMS as SETUP_ITEMS,
)
from app.brain.tools.people import (
    SONG_MAX_LINES as SONG_MAX_LINES,
)
from app.brain.tools.people import (
    _caregiver_by_role as _caregiver_by_role,
)
from app.brain.tools.people import (
    _save_language as _save_language,
)
from app.brain.tools.people import (
    add_person as add_person,
)
from app.brain.tools.people import (
    daily_brief as daily_brief,
)
from app.brain.tools.people import (
    finish_setup as finish_setup,
)
from app.brain.tools.people import (
    forget_skill as forget_skill,
)
from app.brain.tools.people import (
    language_preference as language_preference,
)
from app.brain.tools.people import (
    list_skills as list_skills,
)
from app.brain.tools.people import (
    my_name as my_name,
)
from app.brain.tools.people import (
    save_family_skill as save_family_skill,
)
from app.brain.tools.people import (
    service_status as service_status,
)
from app.brain.tools.people import (
    setup_progress as setup_progress,
)
from app.brain.tools.people import (
    sing as sing,
)
from app.brain.tools.people import (
    voice_replies as voice_replies,
)
from app.brain.tools.places import (  # noqa: F401
    MAPS_HOSTS as MAPS_HOSTS,
)
from app.brain.tools.places import (
    PIN_KEPT_FOR as PIN_KEPT_FOR,
)
from app.brain.tools.places import (
    PIN_SAVE_WITHIN_KM as PIN_SAVE_WITHIN_KM,
)
from app.brain.tools.places import (
    RIDE_PIN_ASK_FOR as RIDE_PIN_ASK_FOR,
)
from app.brain.tools.places import (
    _delivery_place as _delivery_place,
)
from app.brain.tools.places import (
    _follow as _follow,
)
from app.brain.tools.places import (
    _km as _km,
)
from app.brain.tools.places import (
    _maps_meta as _maps_meta,
)
from app.brain.tools.places import (
    _maps_point as _maps_point,
)
from app.brain.tools.places import (
    _pickup_words_asked as _pickup_words_asked,
)
from app.brain.tools.places import (
    _pin_place as _pin_place,
)
from app.brain.tools.places import (
    _ride_end_place as _ride_end_place,
)
from app.brain.tools.places import (
    _ride_pin_first as _ride_pin_first,
)
from app.brain.tools.places import (
    _save_pin as _save_pin,
)
from app.brain.tools.places import (
    _shared_pin as _shared_pin,
)
from app.brain.tools.places import (
    ask_location as ask_location,
)
from app.brain.tools.places import (
    map_link as map_link,
)
from app.brain.tools.places import (
    maps_link_pin as maps_link_pin,
)
from app.brain.tools.places import (
    places as places,
)
from app.brain.tools.places import (
    remove_place as remove_place,
)
from app.brain.tools.places import (
    save_place as save_place,
)
from app.brain.tools.places import (
    saved_place_args as saved_place_args,
)
from app.brain.tools.reminders import (  # noqa: F401
    SEND_COOLDOWN as SEND_COOLDOWN,
)
from app.brain.tools.reminders import (
    _approval_just_asked as _approval_just_asked,
)
from app.brain.tools.reminders import (
    _clock_label as _clock_label,
)
from app.brain.tools.reminders import (
    _first_name as _first_name,
)
from app.brain.tools.reminders import (
    _grounded_send as _grounded_send,
)
from app.brain.tools.reminders import (
    _no_answer_alert_loop as _no_answer_alert_loop,
)
from app.brain.tools.reminders import (
    _order_for as _order_for,
)
from app.brain.tools.reminders import (
    _order_notice as _order_notice,
)
from app.brain.tools.reminders import (
    _refill_who as _refill_who,
)
from app.brain.tools.reminders import (
    _wake_reminder as _wake_reminder,
)
from app.brain.tools.reminders import (
    close_loop_tool as close_loop_tool,
)
from app.brain.tools.reminders import (
    delivery_kind as delivery_kind,
)
from app.brain.tools.reminders import (
    first_line as first_line,
)
from app.brain.tools.reminders import (
    gap_asked as gap_asked,
)
from app.brain.tools.reminders import (
    open_loop_tool as open_loop_tool,
)
from app.brain.tools.reminders import (
    send_message as send_message,
)
from app.brain.tools.reminders import (
    send_problems as send_problems,
)
from app.brain.tools.reminders import (
    template_hint as template_hint,
)
from app.brain.tools.reminders import (
    wake_from_clock as wake_from_clock,
)
from app.brain.tools.trips import (  # noqa: F401
    _AIRLINES as _AIRLINES,
)
from app.brain.tools.trips import (
    _NUTRIENT_WORDS as _NUTRIENT_WORDS,
)
from app.brain.tools.trips import (
    _airline_service as _airline_service,
)
from app.brain.tools.trips import (
    _asks_nutrient as _asks_nutrient,
)
from app.brain.tools.trips import (
    _care_place as _care_place,
)
from app.brain.tools.trips import (
    _logged_nutrient as _logged_nutrient,
)
from app.brain.tools.trips import (
    _surname as _surname,
)
from app.brain.tools.trips import (
    _with_browser as _with_browser,
)
from app.brain.tools.trips import (
    accept_trip as accept_trip,
)
from app.brain.tools.trips import (
    find_doctor as find_doctor,
)
from app.brain.tools.trips import (
    find_lab_test as find_lab_test,
)
from app.brain.tools.trips import (
    note_trip as note_trip,
)
from app.brain.tools.trips import (
    read_trip as read_trip,
)
from app.brain.tools.trips import (
    ways_there as ways_there,
)
from app.brain.tools.trips import (
    web_search as web_search,
)
from app.brain.tools.trips import (
    websearch_limit as websearch_limit,
)
from app.brain.tools.views import (  # noqa: F401
    DASHBOARD_URL as DASHBOARD_URL,
)
from app.brain.tools.views import (
    _name_of as _name_of,
)
from app.brain.tools.views import (
    add_doctor_question as add_doctor_question,
)
from app.brain.tools.views import (
    assign_family_task as assign_family_task,
)
from app.brain.tools.views import (
    care_team_tool as care_team_tool,
)
from app.brain.tools.views import (
    emergency_card as emergency_card,
)
from app.brain.tools.views import (
    family_tasks_tool as family_tasks_tool,
)
from app.brain.tools.views import (
    medicine_stock as medicine_stock,
)
from app.brain.tools.views import (
    past_orders as past_orders,
)
from app.brain.tools.views import (
    patterns_tool as patterns_tool,
)
from app.brain.tools.views import (
    set_stock_tool as set_stock_tool,
)
from app.brain.tools.views import (
    spending_tool as spending_tool,
)
from app.brain.tools.views import (
    weekly_report as weekly_report,
)
from app.brain.tools.views import (
    wellbeing_tool as wellbeing_tool,
)

from app.brain.tools.exact import (  # noqa: E402,F401
    compute as compute,
)

# The order the model sees the tools in (as before the split).
_ORDER = ['need_tools', 'remember', 'stop', 'note', 'recall', 'confirm_change', 'fact_still_true', 'note_habit', 'log_dose', 'log_vital', 'log_event', 'log_food', 'food_log', 'change_food', 'nutrition_target', 'pantry', 'nutrition_settings', 'nutrition_report', 'reminder_log', 'set_reminder', 'activity', 'log_activity', 'reminders', 'change_reminder', 'today_schedule', 'open_loop', 'gap_asked', 'close_loop', 'send_message', 'alert_caregiver', 'search_records', 'lab_trend', 'health_record', 'start_task', 'places', 'save_place', 'map_link', 'ask_location', 'remove_place', 'task_input', 'whats_pending', 'find_lab_test', 'web_search', 'ways_there', 'note_trip', 'accept_trip', 'read_trip', 'find_doctor', 'boundaries', 'set_boundaries', 'cancel_task', 'done_in_app', 'task_status', 'order_feedback', 'set_stock', 'medicine_stock', 'emergency_card', 'care_team', 'add_doctor_question', 'weekly_report', 'wellbeing', 'assign_family_task', 'family_tasks', 'spending', 'past_orders', 'patterns', 'log_outcome', 'offer_buttons', 'feedback', 'learning_consent', 'forget', 'memory_changes', 'undo_change', 'list_skills', 'save_family_skill', 'forget_skill', 'voice_replies', 'sing', 'language_preference', 'setup_progress', 'daily_brief', 'add_person', 'my_name', 'finish_setup', 'service_status']
_TOOLS.update({k: _TOOLS.pop(k) for k in [*_ORDER, *[k for k in list(_TOOLS) if k not in _ORDER]] if k in _TOOLS})
