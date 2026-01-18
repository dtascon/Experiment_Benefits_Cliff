from os import environ

SESSION_CONFIGS = [
    dict(
        name="benefit_cliff_enhanced",
        display_name="Enhanced Benefit Cliff Experiment",
        app_sequence=["logic_experiment"],
        num_demo_participants=10,
    ),
]

SESSION_CONFIG_DEFAULTS = dict(
    real_world_currency_per_point=1.00,
    participation_fee=0.00,
    doc=""
)

# PARTICIPANT FIELDS - Store Prolific and Qualtrics IDs
PARTICIPANT_FIELDS = [
    'prolific_pid',           # Prolific participant ID
    'qualtrics_response_id',  # Qualtrics response ID
    'completion_code',        # Unique code for Prolific completion
]

SESSION_FIELDS = []

LANGUAGE_CODE = 'en'
REAL_WORLD_CURRENCY_CODE = 'USD'
USE_POINTS = True

ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = environ.get('OTREE_ADMIN_PASSWORD')
DEMO_PAGE_INTRO_HTML = """ """
SECRET_KEY = '5347758300814'